import os

from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import wandb
import torch

from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
    AutoModelForCausalLM,
    TrainingArguments,
    pipeline,
    HfArgumentParser,
    logging,
)

from trl import (
    AutoModelForCausalLMWithValueHead,
    PPOConfig,
    PPOTrainer,
    create_reference_model,
    set_seed,
)
from trl.core import LengthSampler

from datasets import load_dataset

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_int8_training,
    AutoPeftModelForCausalLM,
)

from torch.optim import Adam

from accelerate import Accelerator

from tqdm import tqdm

from lib import (
    untar_file,
    get_file_from_aws,
    set_env_variables,
    make_tarfile,
    upload_to_aws,
    collator,
)

"""
python rlhf_llama.py \


accelerate launch rlhf_llama.py \
--llm_model_name llama-2-7b-instruct \
--llm_model_tar_file llama-2-7b-instruct.tar.gz \
--trained_llm_model_save_path llm-model-output \
--trained_llm_model_filename llama-2-7b-instruct-rl-deontology.tar.gz \
--reward_model_tar_file reward_model-deontology-one-epoch.tar.gz \
--ppo_steps 2 \
--ppo_mini_batch_size 1 \
--ppo_batch_size 2 \
--gradient_accumulation_steps 2 \
--aws_access_key_id  \
--aws_secret_access_key  \
--hf_auth_token \
--wandb_api_key  \
--wandb_project llama-2-7b-instruct-rl-deontology \
--wandb_notes "First test run of instruct tuned Llama 2 with deontology reward model"


# 1. Pull the preference reward model checkpoint from S3
# 2. Untar and load the model checkpoint
# 3. Pull the pretrained PPO LLM from S3
# 4. Untar the model and prepare with PEFT and 4Bit parameter loading
# 5. Get the prompt dataset that will be used for to training
# 6. Preprocess the prompt dataset with the LLM tokenizer
# 7. Set up RL process
# 8. Run the prompt data through the reinforcement learning process
# 9. Merge the PEFT layers into the PPO LLM model
# 10. Tar the saved model and push to S3

"""


@dataclass
class ScriptArguments:
    #### Models and dataset
    llm_model_name: str = field(
        metadata={"help": "the model which will be fine-tuned with RLHF"}
    )
    llm_model_tar_file: str = field(
        metadata={"help": "the model which will be fine-tuned with RLHF"}
    )
    trained_llm_model_save_path: str = field(
        metadata={"help": "where to save the model to disk after RLHF fine-tuning"}
    )
    trained_llm_model_filename: str = field(
        metadata={"help": "the name of the tar.gz file in AWS"}
    )
    reward_model_tar_file: str = field(
        metadata={"help": "the reward model that will be used in fine-tuning"}
    )
    prompt_dataset: Optional[str] = field(
        default="yahma/alpaca-cleaned",
        metadata={"help": "the dataset used to prompt the model"},
    )
    prompt_dataset_text_field: Optional[str] = field(
        default="text", metadata={"help": "the text field of the dataset"}
    )

    #### PPO training config
    use_4bit: Optional[bool] = field(
        default=True,
        metadata={"help": "Activate 4bit precision base model loading"},
    )
    use_nested_quant: Optional[bool] = field(
        default=False,
        metadata={"help": "Activate nested quantization for 4bit base models"},
    )
    bnb_4bit_compute_dtype: Optional[str] = field(
        default="bfloat16",
        metadata={"help": "Compute dtype for 4bit base models"},
    )
    bnb_4bit_quant_type: Optional[str] = field(
        default="nf4",
        metadata={"help": "Quantization type fp4 or nf4"},
    )
    fp16: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables fp16 training."},
    )
    bf16: Optional[bool] = field(
        default=True,
        metadata={"help": "Enables bf16 training."},
    )
    gradient_accumulation_steps: Optional[int] = field(
        default=16, metadata={"help": "the number of gradient accumulation steps"}
    )
    ppo_train_epochs: int = field(
        default=1,
        metadata={"help": "Number of training epochs"},
    )
    ppo_save_model_steps: int = field(
        default=100,
        metadata={"help": "model save frequency"},
    )
    ppo_input_min_text_length: int = field(
        default=0,
        metadata={"help": "min prompt length"},
    )
    ppo_input_max_text_length: int = field(
        default=1024,
        metadata={"help": "min prompt length"},
    )
    ppo_output_min_text_length: int = field(
        default=0,
        metadata={"help": "min prompt length"},
    )
    ppo_output_max_text_length: int = field(
        default=1024,
        metadata={"help": "min prompt length"},
    )
    ppo_steps: int = field(
        default=10000,
        metadata={"help": "how many steps should the trainer run for"},
    )
    ppo_mini_batch_size: int = field(
        default=2,
        metadata={"help": "minibatch size"},
    )
    ppo_batch_size: int = field(
        default=4,
        metadata={"help": "main batch size"},
    )
    seed: int = field(
        default=0,
        metadata={"help": "random seed value"},
    )

    #### PEFT
    merge_peft_weights: Optional[bool] = field(
        default=True,
        metadata={"help": "Merge and push weights after training"},
    )
    peft_lora_r: Optional[int] = field(
        default=64, metadata={"help": "the r parameter of the LoRA adapters"}
    )
    peft_lora_alpha: Optional[int] = field(
        default=16, metadata={"help": "the alpha parameter of the LoRA adapters"}
    )
    peft_lora_dropout: Optional[float] = field(default=0.1)

    ##### External services
    aws_bucket_name: str = field(
        default="msc-project-rlhf", metadata={"help": "the model name"}
    )
    aws_access_key_id: str = field(default="", metadata={"help": "AWS access key"})
    aws_secret_access_key: str = field(
        default="", metadata={"help": "AWS secret access key"}
    )
    hf_auth_token: Optional[str] = field(default="", metadata={"help": "HF auth token"})
    wandb_api_key: Optional[str] = field(
        default="", metadata={"help": "Weights and Biases API key"}
    )
    wandb_project: Optional[str] = field(
        default="", metadata={"help": "Weights and Biases project"}
    )
    wandb_notes: Optional[str] = field(
        default="", metadata={"help": "Weights and Biases project"}
    )


def get_args():
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    return script_args


def get_checkpoint_folder(save_dir):
    """
    https://discuss.huggingface.co/t/loading-a-model-from-local-with-best-checkpoint/1707/10
    """
    ckpt_dirs = os.listdir(save_dir)
    ckpt_dirs = [name for name in ckpt_dirs if "-" in name]
    ckpt_dirs = sorted(ckpt_dirs, key=lambda x: int(x.split("-")[1]))
    last_ckpt = ckpt_dirs[-1]

    return last_ckpt


def load_reward_model(args):
    # Pull the tar file from AWS
    get_file_from_aws(args.aws_bucket_name, args.reward_model_tar_file)

    # Untar the file
    model_folder = untar_file(args.reward_model_tar_file)

    checkpoint_folder_name = get_checkpoint_folder(model_folder)

    pretrained_reward_model_path = f"{model_folder}/{checkpoint_folder_name}"

    # Load the entire model on the GPU 0
    # switch to `device_map = "auto"` for multi-GPU
    device_map = {"": 0}

    reward_model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_reward_model_path,
        device_map=device_map,
    )
    reward_tokenizer = AutoTokenizer.from_pretrained(pretrained_reward_model_path)
    reward_config = AutoConfig.from_pretrained(pretrained_reward_model_path)

    # # Quick test of the model
    # reward_pipeline = pipeline(
    #     "text-classification",
    #     model=reward_model,
    #     config=reward_config,
    #     tokenizer=reward_tokenizer,
    # )
    # print(reward_pipeline("hello how are you?"))

    # Return the reward model and reward model tokenizer
    return reward_model, reward_tokenizer


def load_ppo_model(args):
    # Get the file from S3
    # get_file_from_aws(args.aws_bucket_name, args.llm_model_tar_file)

    # Untar the file
    # model_folder = untar_file(args.llm_model_tar_file)
    model_folder = "output"

    checkpoint_folder_name = get_checkpoint_folder(model_folder)

    # Build the path to the model and the path to the tokenizer
    ppo_model_path = f"./{model_folder}/final_merged_checkpoint"
    ppo_tokenizer_path = f"./{model_folder}/{checkpoint_folder_name}"

    current_device = Accelerator().local_process_index

    compute_dtype = getattr(torch, args.bnb_4bit_compute_dtype)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,  # args.use_4bit,
        bnb_4bit_quant_type="nf4",  # fp4 or nf4,  args.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype="bfloat16",  # compute_dtype,
        bnb_4bit_use_double_quant=False,  # args.use_nested_quant,
    )

    if compute_dtype == torch.float16 and args.use_4bit:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            print("=" * 80)
            print(
                "Your GPU supports bfloat16, you can accelerate training with the argument --bf16"
            )
            print("=" * 80)

    # Load the entire model on the GPU 0
    # switch to `device_map = "auto"` for multi-GPU
    # device_map = {"": 0}

    # Apply LoRA config for peft library
    peft_config = LoraConfig(
        lora_alpha=args.peft_lora_alpha,
        lora_dropout=args.peft_lora_dropout,
        r=args.peft_lora_r,
        bias="none",
        task_type="CAUSAL_LM",
    )

    ppo_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        ppo_model_path,
        load_in_8bit=True,
        # quantization_config=bnb_config,
        device_map={"": current_device},
        peft_config=peft_config,
        # load_in_4bit=True
    )

    ppo_model.config.pretraining_tp = 1

    ppo_tokenizer = AutoTokenizer.from_pretrained(ppo_tokenizer_path)

    # https://github.com/huggingface/transformers/issues/24843
    num_tokens = len(ppo_tokenizer)
    ppo_model.pretrained_model.resize_token_embeddings(num_tokens)

    return ppo_model, ppo_tokenizer


def build_dataset(args, tokenizer):
    print(f"Loading dataset: {args.prompt_dataset}")
    tokenizer.pad_token = tokenizer.eos_token

    # Load the dataset
    ds = load_dataset(args.prompt_dataset, split="train")
    # ds = ds.rename_columns({"text": "review"})
    # ds = ds.filter(lambda x: len(x["prompt"]) > 200, batched=False)

    input_size = LengthSampler(
        args.ppo_input_min_text_length, args.ppo_input_max_text_length
    )

    def tokenize(sample):
        full_string = sample["instruction"]

        if len(sample["input"]) > 0:
            full_string = f'{sample["instruction"]}:{sample["input"]}'

        sample["input_ids"] = tokenizer.encode(full_string)[: input_size()]
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")
    # ds.
    return ds


def main(args):
    # 1. Pull the preference reward model checkpoint from S3
    # 2. Untar and load the model checkpoint
    reward_model, reward_tokenizer = load_reward_model(args)

    # 3. Pull the pretrained PPO LLM from S3
    # 4. Untar the model and prepare with PEFT and 4Bit parameter loading
    ppo_model, ppo_tokenizer = load_ppo_model(args)

    # 5. Get the prompt dataset that will be used for to training
    # 6. Preprocess the prompt dataset with the LLM tokenizer
    prompt_dataset = build_dataset(args, ppo_tokenizer)

    # 7. Set up RL process
    current_device = Accelerator().local_process_index

    # Create the reference model
    ppo_ref_model = create_reference_model(ppo_model, num_shared_layers=20)

    ppo_config = PPOConfig(
        model_name=args.llm_model_name,
        learning_rate=(1.47e-5) * 2,
        log_with=["wandb"],
        tracker_project_name=args.wandb_project,
        tracker_kwargs={
            "wandb": {
                "notes": args.wandb_notes,
            }
        },
        ppo_epochs=args.ppo_train_epochs,
        mini_batch_size=args.ppo_mini_batch_size,
        batch_size=args.ppo_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

    # We make sure to use `Adam` optimizer on the model parameters that require gradients.
    optimizer = Adam(
        filter(lambda p: p.requires_grad, ppo_model.parameters()),
        lr=ppo_config.learning_rate,
    )

    # if getattr(ppo_tokenizer, "pad_token", None) is None:
    #     ppo_tokenizer.pad_token = ppo_tokenizer.eos_token

    # We then build the PPOTrainer, passing the model, the reference model, the tokenizer
    ppo_trainer = PPOTrainer(
        ppo_config,
        ppo_model,
        ref_model=ppo_ref_model,
        tokenizer=ppo_tokenizer,
        dataset=prompt_dataset,
        data_collator=collator,
        optimizer=optimizer,
    )

    reward_model.to(ppo_trainer.accelerator.device)

    # Text generation from the PPO model
    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": ppo_tokenizer.eos_token_id,
    }

    output_length_sampler = LengthSampler(
        args.ppo_output_min_text_length, args.ppo_output_max_text_length
    )

    current_step = 0

    # 8. Run the prompt data through the reinforcement learning process
    for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
        query_tensors = batch["input_ids"]

        # query_tensors.to("cuda:0")
        # query_tensors = {torch.tensor(v).to(ppo_trainer.accelerator.device) for v in query_tensors}
        # query_tensors = query_tensors.to(ppo_trainer.accelerator.device)

        # Get response from the policy model
        response_tensors = ppo_trainer.generate(
            query_tensors,
            return_prompt=False,
            # length_sampler=output_length_sampler,
            **generation_kwargs,
        )
        batch["response"] = ppo_tokenizer.batch_decode(
            response_tensors, skip_special_tokens=True
        )

        # Compute sentiment score # noqa
        texts = batch["response"]
        inputs_rewards = reward_tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt"
        ).to(ppo_trainer.accelerator.device)
        logits = reward_model(**inputs_rewards).logits.float()
        labels = (logits[:, 0]).tolist()

        rewards = [
            torch.tensor(output).to(ppo_trainer.accelerator.device) for output in labels
        ]

        # Run PPO step
        # NOTE THis line errs
        print(query_tensors)
        print(response_tensors)
        print(rewards)

        # rewards = rewards.to(ppo_trainer.accelerator.device)

        # print(f"query_tensors: {query_tensors.is_cuda}")
        # print(f"response_tensors: {response_tensors.is_cuda}")
        # print(f"rewards: {rewards.is_cuda}")

        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)

        # Save model every 100 steps
        if epoch % args.ppo_save_model_steps == 0:
            if ppo_trainer.accelerator.is_main_process:
                ppo_trainer.save_pretrained(args.trained_llm_model_save_path)

        current_step = current_step + 1

        # Have we hit the required number of steps
        if current_step > args.ppo_steps:
            break

    # 9. Merge the PEFT layers into the PPO LLM model
    if args.merge_peft_weights:
        output_dir = os.path.join(args.trained_llm_model_save_path, "final_checkpoints")
        ppo_trainer.model.save_pretrained(args.trained_llm_model_save_path)

        # Free memory for merging weights
        del ppo_model
        torch.cuda.empty_cache()

        model = AutoPeftModelForCausalLM.from_pretrained(
            output_dir, device_map="auto", torch_dtype=torch.bfloat16
        )
        model = model.merge_and_unload()

        output_merged_dir = os.path.join(args.output_dir, "final_merged_checkpoint")
        model.save_pretrained(output_merged_dir, safe_serialization=True)

    # 10. Tar the saved model and push to S3
    make_tarfile(args.trained_llm_model_filename, args.trained_llm_model_save_path)
    upload_to_aws(
        args.trained_llm_model_filename,
        args.aws_bucket_name,
        args.trained_llm_model_filename,
    )


if __name__ == "__main__":
    args = get_args()

    set_env_variables(args)
    set_seed(args.seed)

    logging.set_verbosity_info()

    main(args)

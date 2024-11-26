import argparse
import os

from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    logging,
    set_seed,
)

from trl import SFTTrainer
from trl.trainer import ConstantLengthDataset

from lib import set_env_variables

"""
Fine-Tune Llama 2 on SE paired dataset

Adapted from the example code found here:
https://github.com/lvwerra/trl/blob/main/examples/research_projects/stack_llama/scripts/supervised_finetuning.py

"""


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument(
        "--dataset_name", type=str, default="timdettmers/openassistant-guanaco"
    )

    parser.add_argument("--seq_length", type=int, default=1024)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--eos_token_id", type=int, default=49152)

    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--num_warmup_steps", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0.05)

    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--no_fp16", action="store_false")
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument(
        "--no_gradient_checkpointing", action="store_false", default=False
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--log_freq", default=1, type=int)
    parser.add_argument("--eval_freq", default=100, type=int)
    parser.add_argument("--save_freq", default=1000, type=int)

    parser.add_argument("--hf_auth_token", default=None, type=str)
    parser.add_argument("--wandb_api_key", default=None, type=str)
    parser.add_argument("--wandb_project", default=None, type=str)
    parser.add_argument("--wandb_notes", default=None, type=str)

    return parser.parse_args()


def chars_token_ratio(dataset, tokenizer, nb_examples=400):
    """
    Estimate the average number of characters per token in the dataset.
    """
    total_characters, total_tokens = 0, 0
    for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):
        text = example["text"]
        total_characters += len(text)
        if tokenizer.is_fast:
            total_tokens += len(tokenizer(text).tokens())
        else:
            total_tokens += len(tokenizer.tokenize(text))

    return total_characters / total_tokens


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def create_datasets(tokenizer, args):
    dataset = load_dataset(args.dataset_name, use_auth_token=True)

    # print(dataset)

    train_data = dataset["train"]  # type: ignore

    # def map_fn(example):
    #     return example['text']

    # train_data = train_data.map(map_fn)

    valid_data = dataset["test"]  # type: ignore

    # valid_data = valid_data.map(map_fn)

    print(
        f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}"
    )

    chars_per_token = chars_token_ratio(train_data, tokenizer)
    print(f"The character to token ratio of the dataset is: {chars_per_token:.2f}")

    train_dataset = ConstantLengthDataset(
        tokenizer,
        train_data,
        infinite=False,
        seq_length=args.seq_length,
        chars_per_token=chars_per_token,
        dataset_text_field="text",
    )
    valid_dataset = ConstantLengthDataset(
        tokenizer,
        valid_data,
        infinite=False,
        seq_length=args.seq_length,
        chars_per_token=chars_per_token,
        dataset_text_field="text",
    )
    return train_dataset, valid_dataset


def run_training(args, train_data, eval_data):
    print("Loading the model")

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    train_data.start_iteration = 0

    print("Starting main loop")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        dataloader_drop_last=True,
        evaluation_strategy="steps",
        max_steps=args.max_steps,
        eval_steps=args.eval_freq,
        save_steps=args.save_freq,
        logging_steps=args.log_freq,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.num_warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        fp16=not args.no_fp16,
        bf16=args.bf16,
        weight_decay=args.weight_decay,
        report_to="wandb",  # type: ignore
        ddp_find_unused_parameters=False,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, load_in_8bit=True, device_map={"": Accelerator().process_index}
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        peft_config=lora_config,  # type: ignore
        packing=True,
    )

    print_trainable_parameters(trainer.model)

    print("Training...")
    trainer.train()

    print("Saving last checkpoint of the model")
    trainer.model.save_pretrained(os.path.join(args.output_dir, "final_checkpoint/"))  # type: ignore


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    train_dataset, eval_dataset = create_datasets(tokenizer, args)
    run_training(args, train_dataset, eval_dataset)


if __name__ == "__main__":
    args = get_args()
    assert args.model_path != "", "Please provide the llama model path"

    os.environ["HUGGING_FACE_HUB_TOKEN"] = args.hf_auth_token
    os.environ["WANDB_API_KEY"] = args.wandb_api_key
    os.environ["WANDB_PROJECT"] = args.wandb_project
    os.environ["WANDB_NOTES"] = args.wandb_notes

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    logging.set_verbosity_error()

    main(args)

# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    logging,
    set_seed,
)

from trl import SFTTrainer

from lib import make_tarfile, upload_to_aws, set_env_variables

from accelerate import Accelerator

accelerator = Accelerator()

tqdm.pandas()

"""
This script is adapted to fine-tune the Llama2 with the timdettmers/openassistant-guanaco
dataset. Functions for setting up Weights and biases and AWS credentials have been added.


python llama_sft.py \
--model_name meta-llama/Llama-2-7b-hf \
--load_in_4bit \
--use_peft \
--batch_size 4 \
--gradient_accumulation_steps 2 \
--use_auth_token false \
--hf_auth_token \
--wandb_project facebook-opt-125m-fine-tune-test \
--wandb_api_key  \
--aws_access_key_id  \
--aws_secret_access_key 
--aws_filename facebook-opt-125m-fine-tune-test.tar.gz
"""


# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with SFTTrainer
    """

    model_name: Optional[str] = field(
        default="meta-llama/Llama-2-13b-hf", metadata={"help": "the model name"}
    )
    dataset_name: Optional[str] = field(
        default="timdettmers/openassistant-guanaco", metadata={"help": "the model name"}
    )
    dataset_text_field: Optional[str] = field(
        default="text", metadata={"help": "the text field of the dataset"}
    )
    learning_rate: Optional[float] = field(
        default=1.41e-5, metadata={"help": "the learning rate"}
    )
    max_grad_norm: Optional[float] = field(default=0.3)
    local_rank: Optional[int] = field(
        default=-1, metadata={"help": "Used for multi-gpu"}
    )
    per_device_train_batch_size: Optional[int] = field(default=4)
    per_device_eval_batch_size: Optional[int] = field(default=1)
    # batch_size: Optional[int] = field(default=64, metadata={"help": "the batch size"})
    max_steps: int = field(
        default=10000, metadata={"help": "How many optimizer update steps to take"}
    )
    gradient_accumulation_steps: Optional[int] = field(
        default=16, metadata={"help": "the number of gradient accumulation steps"}
    )
    use_4bit: Optional[bool] = field(
        default=True,
        metadata={"help": "Activate 4bit precision base model loading"},
    )
    use_nested_quant: Optional[bool] = field(
        default=False,
        metadata={"help": "Activate nested quantization for 4bit base models"},
    )
    bnb_4bit_compute_dtype: Optional[str] = field(
        default="float16",
        metadata={"help": "Compute dtype for 4bit base models"},
    )
    bnb_4bit_quant_type: Optional[str] = field(
        default="nf4",
        metadata={"help": "Quantization type fp4 or nf4"},
    )
    num_train_epochs: Optional[int] = field(
        default=1,
        metadata={"help": "The number of training epochs for the reward model."},
    )
    trust_remote_code: Optional[bool] = field(
        default=True, metadata={"help": "Enable `trust_remote_code`"}
    )
    output_dir: Optional[str] = field(
        default="output", metadata={"help": "the output directory"}
    )
    peft_lora_r: Optional[int] = field(
        default=64, metadata={"help": "the r parameter of the LoRA adapters"}
    )
    peft_lora_alpha: Optional[int] = field(
        default=16, metadata={"help": "the alpha parameter of the LoRA adapters"}
    )
    peft_lora_dropout: Optional[float] = field(default=0.1)
    logging_steps: Optional[int] = field(
        default=1, metadata={"help": "the number of logging steps"}
    )
    use_auth_token: Optional[bool] = field(
        default=True, metadata={"help": "Use HF auth token to access the model"}
    )
    hf_auth_token: Optional[str] = field(default="", metadata={"help": "HF auth token"})
    seed: Optional[int] = field(default=0, metadata={"help": "Random value seed"})
    wandb_api_key: Optional[str] = field(
        default="", metadata={"help": "Weights and Biases API key"}
    )
    wandb_project: Optional[str] = field(
        default="", metadata={"help": "Weights and Biases project"}
    )
    aws_access_key_id: Optional[str] = field(
        default="", metadata={"help": "AWS access key"}
    )
    aws_secret_access_key: Optional[str] = field(
        default="", metadata={"help": "AWS secret access key"}
    )
    aws_filename: Optional[str] = field(default="", metadata={"help": "AWS filename"})
    max_seq_length: Optional[int] = field(
        default=1024, metadata={"help": "max_seq_length"}
    )
    merge_peft_weights: Optional[bool] = field(
        default=False,
        metadata={"help": "Merge and push weights after training"},
    )
    optim: Optional[str] = field(
        default="paged_adamw_32bit",
        metadata={"help": "The optimizer to use."},
    )
    fp16: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables fp16 training."},
    )
    bf16: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables bf16 training."},
    )
    packing: Optional[bool] = field(
        default=False,
        metadata={"help": "Use packing dataset creating."},
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True,
        metadata={"help": "Enables gradient checkpointing."},
    )
    lr_scheduler_type: str = field(
        default="constant",
        metadata={
            "help": "Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis"
        },
    )
    warmup_ratio: float = field(
        default=0.03, metadata={"help": "Fraction of steps to do a warmup for"}
    )
    group_by_length: bool = field(
        default=True,
        metadata={
            "help": "Group sequences into batches with same length. Saves memory and speeds up training considerably."
        },
    )
    save_steps: int = field(
        default=10, metadata={"help": "Save checkpoint every X updates steps."}
    )


def get_args():
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    print(f"Model: {script_args.model_name}")
    print(f"Data: {script_args.dataset_name}")

    return script_args


def create_and_prepare_model(args):
    compute_dtype = getattr(torch, args.bnb_4bit_compute_dtype)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=args.use_4bit,
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=args.use_nested_quant,
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
    device_map = {"": 0}

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        use_auth_token=args.use_auth_token,
    )

    # check: https://github.com/huggingface/transformers/pull/24906
    model.config.pretraining_tp = 1

    peft_config = LoraConfig(
        lora_alpha=args.peft_lora_alpha,
        lora_dropout=args.peft_lora_dropout,
        r=args.peft_lora_r,
        bias="none",
        task_type="CAUSAL_LM",
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    return model, peft_config, tokenizer


def main(args):
    model, peft_config, tokenizer = create_and_prepare_model(args)
    model.config.use_cache = False

    # Step 2: Load the dataset
    dataset = load_dataset(args.dataset_name, split="train")

    # Fix weird overflow issue with fp16 training
    tokenizer.padding_side = "right"

    # Step 3: Define the training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        optim=args.optim,
        logging_steps=args.logging_steps,
        fp16=args.fp16,
        bf16=args.bf16,
        learning_rate=args.learning_rate,
        save_steps=args.save_steps,
        max_grad_norm=args.max_grad_norm,
        # max_steps=args.max_steps,
        warmup_ratio=args.warmup_ratio,
        group_by_length=args.group_by_length,
        lr_scheduler_type=args.lr_scheduler_type,
        num_train_epochs=args.num_train_epochs,
        report_to="wandb",
        save_strategy="epoch",
    )

    # Step 5: Define the Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        dataset_text_field=args.dataset_text_field,
        peft_config=peft_config,
        max_seq_length=args.max_seq_length,
    )

    trainer.train()

    if args.merge_peft_weights:
        output_dir = os.path.join(args.output_dir, "final_checkpoints")
        trainer.model.save_pretrained(output_dir)

        # Free memory for merging weights
        del model
        torch.cuda.empty_cache()

        from peft import AutoPeftModelForCausalLM

        model = AutoPeftModelForCausalLM.from_pretrained(
            output_dir, device_map="auto", torch_dtype=torch.bfloat16
        )
        model = model.merge_and_unload()

        output_merged_dir = os.path.join(args.output_dir, "final_merged_checkpoint")
        model.save_pretrained(output_merged_dir, safe_serialization=True)


if __name__ == "__main__":
    args = get_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    set_env_variables(args)

    logging.set_verbosity_info()

    main(args)

    # Save to AWS
    # zip and upload the trained model to s3
    make_tarfile(args.aws_filename, args.output_dir)
    upload_to_aws(args.aws_filename, "msc-project-rlhf", args.aws_filename)

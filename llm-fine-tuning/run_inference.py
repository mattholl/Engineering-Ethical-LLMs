import os
import boto3
import tarfile
import torch
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Optional

from transformers import (
    HfArgumentParser,
    logging,
    LlamaForCausalLM,
    LlamaTokenizer,
    AutoTokenizer,
    AutoModelForCausalLM,
    LlamaTokenizerFast,
)

from lib import untar_file, get_file_from_aws, set_some_env_variables

tqdm.pandas()

"""
python run_inference.py \
--model_tar_file llama-2-7b-instruct.tar.gz \
--prompt "What is a llama?" \
--aws_access_key_id  \
--aws_secret_access_key 

- 1. Get the gzipped model from AWS bucket
- 2. Untar the file
- 3. Load the fine-tuned Llama 2 model and tokenizer
- 4. Build a pipeline to run with a prompt
"""


@dataclass
class ScriptArguments:
    model_tar_file: str = field(metadata={"help": "the model name"})
    load_from_hub: bool = field(
        metadata={"help": "should the model be loaded from the hub or from disk"}
    )
    aws_bucket_name: str = field(
        default="msc-project-rlhf", metadata={"help": "the model name"}
    )
    prompt: str = field(
        default="What is a llama?", metadata={"help": "the prompt for the model"}
    )
    aws_access_key_id: str = field(default="", metadata={"help": "AWS access key"})
    aws_secret_access_key: str = field(
        default="", metadata={"help": "AWS secret access key"}
    )
    hf_auth_token: Optional[str] = field(default="", metadata={"help": "HF auth token"})


def get_args():
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    return script_args


def load_model(model_folder):
    print("load_model")
    print(model_folder)
    # Return the model and tokenizer to use for inference.
    model = AutoModelForCausalLM.from_pretrained(model_folder)

    # tokenizer = LlamaTokenizerFast.from_pretrained("meta-llama/Llama-2-7b-hf")
    tokenizer = AutoTokenizer.from_pretrained("output/checkpoint-153")

    return model, tokenizer


def format_prompt(prompt_text):
    # Prepend with "### Human {prompt}. ### Assistant: "
    return f"### Human: {prompt_text}.### Assistant:"


def main(args):
    if not args.load_from_hub:
        # Retrieve model - get from AWS and untar
        get_file_from_aws(args.aws_bucket_name, args.model_tar_file, "./")
        model_folder = untar_file(args.model_tar_file)

        # Load the model
        checkpoint_folder = f"./{model_folder}/final_merged_checkpoint"
        model, tokenizer = load_model(checkpoint_folder)
    else:
        # Get the model and tokenizer using the the model_folder name
        # to pull from huggingface hub
        pass

    # format the prompt
    formatted_prompt = format_prompt(args.prompt)
    print(formatted_prompt)

    # Run the prompt through the model pipeline
    # pipline etc.
    inputs = tokenizer(formatted_prompt, return_tensors="pt")

    # Generate
    generate_ids = model.generate(inputs.input_ids, max_length=200)
    response = tokenizer.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    print(f"\n\nResponse: {response}")


if __name__ == "__main__":
    args = get_args()

    set_some_env_variables(args)

    logging.set_verbosity_info()

    main(args)

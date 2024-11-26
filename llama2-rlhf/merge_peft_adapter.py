import os
from dataclasses import dataclass, field
from typing import Optional

import torch
from peft import PeftConfig, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
)
from lib import upload_to_aws, upload_dir_s3


@dataclass
class ScriptArguments:
    """
    Adapted from: https://github.com/lvwerra/trl/blob/main/examples/research_projects/stack_llama/scripts/merge_peft_adapter.py
    My modifications:
    - optional upload to hugging face
    - optional upload to S3
    """

    adapter_model_name: Optional[str] = field(
        default=None, metadata={"help": "the adapter model name"}
    )
    base_model_name: Optional[str] = field(
        default=None, metadata={"help": "the base model name"}
    )
    output_name: Optional[str] = field(
        default=None, metadata={"help": "the output model name"}
    )
    push_to_hub: Optional[bool] = field(
        default=False,
        metadata={"help": "should the model be pushed to the Hugging face hub"},
    )
    hugging_face_api_key: Optional[str] = field(
        default=None, metadata={"help": "HF API key"}
    )
    push_to_s3: Optional[bool] = field(
        default=False, metadata={"help": "should the model be pushed to S3"}
    )
    aws_access_key_id: Optional[str] = field(
        default="", metadata={"help": "AWS access key"}
    )
    aws_secret_access_key: Optional[str] = field(
        default="", metadata={"help": "AWS secret access key"}
    )
    s3_bucket_name: str = field(
        default="msc-project-rlhf", metadata={"help": "the bucket name"}
    )
    s3_object_prefix: str = field(default="", metadata={"help": "S3 object name"})


parser = HfArgumentParser(ScriptArguments)  # type: ignore
script_args = parser.parse_args_into_dataclasses()[0]
assert (
    script_args.adapter_model_name is not None
), "please provide the name of the Adapter you would like to merge"
assert (
    script_args.base_model_name is not None
), "please provide the name of the Base model"
assert (
    script_args.output_name is not None
), "please provide the output name of the merged model"


# os.environ["HUGGING_FACE_HUB_TOKEN"] = script_args.hugging_face_api_key
os.environ["AWS_ACCESS_KEY_ID"] = script_args.aws_access_key_id
os.environ["AWS_SECRET_ACCESS_KEY"] = script_args.aws_secret_access_key

print("Loading PEFT config...\n")
peft_config = PeftConfig.from_pretrained(script_args.adapter_model_name)

print("Loading model...\n")
if peft_config.task_type == "SEQ_CLS":
    # peft is for reward model so load sequence classification
    model = AutoModelForSequenceClassification.from_pretrained(
        script_args.base_model_name, num_labels=1, torch_dtype=torch.bfloat16
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        script_args.base_model_name, return_dict=True, torch_dtype=torch.bfloat16
    )

print("Loading tokenizer...\n")
tokenizer = AutoTokenizer.from_pretrained(script_args.base_model_name)

# Load the Lora model
print("Load the Lora model...\n")
model = PeftModel.from_pretrained(model, script_args.adapter_model_name)
model.eval()

print("Merge...\n")
model = model.merge_and_unload()  # type: ignore

print("Save the model and tokenizer...\n")
model.save_pretrained(f"{script_args.output_name}")
tokenizer.save_pretrained(f"{script_args.output_name}")


if script_args.push_to_hub:
    print("Pushing to huggingface hub...\n")
    model.push_to_hub(
        f"{script_args.output_name}",
        use_temp_dir=False,
        use_auth_token=script_args.hugging_face_api_key,
    )

if script_args.push_to_s3:
    print("Pushing to S3...\n")
    # upload_to_aws(
    #     script_args.output_name,
    #     script_args.s3_bucket_name,
    #     script_args.s3_object_prefix,
    # )
    upload_dir_s3(
        script_args.output_name,
        script_args.s3_bucket_name,
        script_args.s3_object_prefix,
    )

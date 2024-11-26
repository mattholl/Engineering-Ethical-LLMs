from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Optional

from transformers import (
    HfArgumentParser,
    logging,
    LlamaForCausalLM,
    LlamaTokenizerFast,
)

tqdm.pandas()

"""
python run_inference.py \
--local_model ./rlm-llama-2-13b-rl-harmless-step_400 \
--prompt "What is a llama?"
"""


@dataclass
class ScriptArguments:
    local_model: str = field(metadata={"help": "the model name"})
    prompt: str = field(
        default="What is a llama?", metadata={"help": "the prompt for the model"}
    )


def get_args():
    parser = HfArgumentParser(ScriptArguments)  # type: ignore
    script_args = parser.parse_args_into_dataclasses()[0]

    return script_args


def load_model(model_folder):
    print("load_model")
    print(model_folder)
    # Return the model and tokenizer to use for inference.
    model = LlamaForCausalLM.from_pretrained(
        model_folder, device_map="auto", load_in_8bit=True
    )
    tokenizer = LlamaTokenizerFast.from_pretrained(
        model_folder, device_map="auto", load_in_8bit=True
    )

    return model, tokenizer


def format_prompt(prompt_text):
    # Prepend with "### Human {prompt}. ### Assistant: "
    return f"### Human: {prompt_text}.### Assistant:"


def main(args):
    model, tokenizer = load_model(args.local_model)

    # format the prompt
    formatted_prompt = format_prompt(args.prompt)
    print(formatted_prompt)

    # Run the prompt through the model pipeline
    # pipline etc.
    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    inputs = inputs.to("cuda")

    # Generate
    generate_ids = model.generate(inputs.input_ids, max_length=128)  # type: ignore
    response = tokenizer.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    print(f"\n\nGenerated response: {response}")


if __name__ == "__main__":
    args = get_args()
    logging.set_verbosity_info()

    main(args)

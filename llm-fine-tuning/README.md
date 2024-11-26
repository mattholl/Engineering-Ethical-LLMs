# LLM fine-tuning

## Supervised fine-tuning with instruct dataset

https://gist.github.com/younesbelkada/9f7f75c94bdc1981c8ca5cc937d4a4da


1. Apply the supervised fine-tuning to Llama2 model from huggingface
2. apply the peft layers
3. save the pretrained model to disk
4. tar and push to S3

test the smaller model on a cloud GPU
update to use accelerate
run the full training on a cloud GPU

```
pip install trl peft bitsandbytes datasets tqdm transformers boto3 wandb sentencepiece
```

```
accelerate launch llama_sft.py \
--model_name meta-llama/Llama-2-7b-hf \
--bf16 \
--bnb_4bit_compute_dtype bfloat16 \
--use_auth_token \
--merge_peft_weights \
--hf_auth_token '' \
--wandb_project llama-2-7b-instruct \
--wandb_api_key '' \
--aws_access_key_id '' \
--aws_secret_access_key '' \
--aws_filename llama-2-7b-instruct.tar.gz
```


```
accelerate launch llama_sft.py \
--model_name meta-llama/Llama-2-13b-hf \
--per_device_train_batch_size 8 \
--bf16 \
--bnb_4bit_compute_dtype bfloat16 \
--use_auth_token \
--merge_peft_weights \
--hf_auth_token '' \
--wandb_project llama-2-13b-instruct \
--wandb_api_key '' \
--aws_access_key_id '' \
--aws_secret_access_key '' \
--aws_filename llama-2-13b-instruct.tar.gz
```

## Test supervised fine-tuning with model inference

```
python run_inference.py \
--model_tar_file llama-2-7b-instruct.tar.gz \
--prompt "What is a llama?" \
--aws_access_key_id '' \
--aws_secret_access_key ''
```

## RLHF with ethical reward models

### 1. Supervised fine-tuning

```
supervised_finetuning.py

os.environ["HUGGING_FACE_HUB_TOKEN"] = ""
os.environ["WANDB_API_KEY"] = ""

```

Llama2

```
torchrun --nnodes 1 --nproc_per_node 1 llama_sft.py --model_path=meta-llama/Llama-2-7b-hf --streaming --no_gradient_checkpointing --learning_rate 1e-5 --max_steps 5000 --output_dir ./llama-2-7b-sft
```

### 2. Merge PEFT layers

Llama2

```
python examples/research_projects/stack_llama/scripts/merge_peft_adapter.py --adapter_model_name=./llama-2-7b-se/final_checkpoint --base_model_name=meta-llama/Llama-2-7b-hf --output_name=llama-2-7b-se-sft
```

### 3. Train reward model

Llama2

```
torchrun --nnodes 1 --nproc_per_node 1 examples/research_projects/stack_llama/scripts/reward_modeling.py --model_name=./llama-2-7b-se-sft --per_device_train_batch_size 2
```

Cancel without 1 epoch, this will take two days.

### 4. Merge layers again

Llama2

```
python examples/research_projects/stack_llama/scripts/merge_peft_adapter.py \
--adapter_model_name=./llama-2-7b-se-sft_peft_stack-exchange-paired_rmts__100000_2e-05/checkpoint-4500 \
--base_model_name=huggyllama/llama-7b \
--output_name=llama-2-7b-se-reward
```

### 5. Run the RL fine-tuning

Run overnight.

Llama 2

```
accelerate launch \
--num_machines 1  \
--num_processes 1 \
examples/research_projects/stack_llama/scripts/rl_training.py \
--log_with=wandb \
--model_name=./llama-2-7b-se-sft \
--reward_model_name=./llama-2-7b-se-reward \
--adafactor=False \
--tokenizer_name=./llama-2-7b-se-reward \
--save_freq=100 \
--output_max_length=128 \
--batch_size=8 \
--gradient_accumulation_steps=8 \
--batched_gen=True \
--ppo_epochs=4 \
--seed=0 \
--learning_rate=1.4e-5 \
--early_stopping=True \
--output_dir=llama-se-rl-finetune-128-8-8-1.4e-5_adam
```

### Merge again

```
python examples/research_projects/stack_llama/scripts/merge_peft_adapter.py \
--adapter_model_name=./llama-se-rl-finetune-128-8-8-1.4e-5_adamstep_500 \
--base_model_name=./llama-2-7b-se-sft \
--output_name=llama-2-7b-rl-fine-tuned
```

## Test inference

# Llama2 RLHF with ethical datasets

- supervised fine-tuning
    - check the dataset for "###Human: ..." prompts
- train a reward model for each dataset
    - 1 epoch only
- fine-tune the SFT base model with each dataset for reward models
    - push models to S3
- apply the reward model to the supervised fine-tuned base model

This project contains the Python scripts and Notebooks used in my MSc project exploring the efficacy of various ethical theories in mitigating bias and toxicity in generative language models. The steps below are covered by these scripts:

1. Train a base models (Llama 2 13B) with an human / assistnat oriented instruct dataset
2. From this base model train a reward preference model for each ethical theory
3. Use the ethical reward models to fine-tune the instruct tuned base model with reinforcement learning


The Python scripts here are adapted from ...

Set up on a new machine, clone the project and `pip install`:

```
git clone https://gitlab+deploy-token-2319714:fNHftsv7PWoFRxwubDxF@gitlab.com/msc-final-project/llama2-rlhf.git
pip install trl peft bitsandbytes datasets tqdm transformers wandb sentencepiece boto3 evaluate
```

NOTE upload to S3 llama-2/13b/



## 1. Supervised fine-tuning with instruct dataset

```bash
torchrun \
--nnodes 1 \
--nproc_per_node 2 \
supervised_finetuning.py \
--model_path=meta-llama/Llama-2-13b-hf \
--no_gradient_checkpointing \
--learning_rate=1e-5 \
--max_steps=5000 \
--output_dir=./llama-2-13b-adapter \
--hf_auth_token=\
--wandb_project=llama-2-13b__supervised_finetuning \
--wandb_api_key= \
--wandb_notes="Fine-tuning the base Llama 2 13B model with a instruct oriented dataset"
```

## 2. Merge the PEFT layers to get a standalone model and upload to S3

```bash
python merge_peft_adapter.py \
--adapter_model_name=./llama-2-13b-adapter/checkpoint-5000 \
--base_model_name=meta-llama/Llama-2-13b-hf \
--output_name=llama-2-13b-instruct-fine-tuned \
--push_to_s3 \
--aws_access_key_id= \
--aws_secret_access_key= \
--s3_object_prefix=llama-2/13b \
--hugging_face_api_key=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
```

This results in a supervised fine-tuned base model. From the base models the various reward models from ethical theories with be trained.

## 3. Reward modelling

### Dataset generation

Datasets are prepared with `data/explore-ethics-dataset.ipynb`.

### Train the reward models from each dataset

### Deontology

```bash
torchrun \
--nnodes 1 \
--nproc_per_node 2 \
reward_modelling.py \
--model_name=./llama-2-13b-instruct-fine-tuned \
--output_name=llama-2-13b-deontology-reward-model \
--local_rank=2 \
--eval_first_step \
--train_dataset=./data/reward-training-data/deontology_train.csv \
--eval_dataset=./data/reward-training-data/deontology_test.csv \
--wandb_project=llama-2-13b-deontology-reward-model \
--wandb_api_key= \
--wandb_notes="Deontology dataset reward model based on llama-2-13b-instruct-fine-tuned base model"
```

### Harmless

```bash
torchrun \
--nnodes 1 \
--nproc_per_node 2 \
reward_modelling.py \
--model_name=./llama-2-13b-instruct-fine-tuned \
--output_name=llama-2-13b-harmless-reward-model-adapter \
--local_rank=2 \
--eval_first_step \
--per_device_train_batch_size=4 \
--train_dataset=./data/reward-training-data/harmless_train_sample.csv \
--eval_dataset=./data/reward-training-data/harmless_test.csv \
--wandb_project=llama-2-13b-harmless-reward-model \
--wandb_api_key= \
--wandb_notes="Harmless dataset reward model based on llama-2-13b-instruct-fine-tuned base model"
```

### Utilitarianism

```bash
torchrun \
--nnodes 1 \
--nproc_per_node 2 \
reward_modelling.py \
--model_name=./llama-2-13b-instruct-fine-tuned \
--output_name=llama-2-13b-utilitarianism-reward-model-adapter \
--local_rank=2 \
--eval_first_step \
--per_device_train_batch_size=4 \
--train_dataset=./data/reward-training-data/utilitarianism_train.csv \
--eval_dataset=./data/reward-training-data/utilitarianism_test.csv \
--wandb_project=llama-2-13b-utilitarianism-reward-model \
--wandb_api_key= \
--wandb_notes="Utilitarianism dataset reward model based on llama-2-13b-instruct-fine-tuned base model"
```


### Virtue

```bash
torchrun \
--nnodes 1 \
--nproc_per_node 2 \
reward_modelling.py \
--model_name=./llama-2-13b-instruct-fine-tuned \
--output_name=llama-2-13b-virtue-reward-model-adapter \
--local_rank=2 \
--eval_first_step \
--per_device_train_batch_size=4 \
--train_dataset=./data/reward-training-data/virtue_train.csv \
--eval_dataset=./data/reward-training-data/virtue_test.csv \
--wandb_project=llama-2-13b-virtue-reward-model \
--wandb_api_key= \
--wandb_notes="Virtue dataset reward model based on llama-2-13b-instruct-fine-tuned base model"
```


## 4. Merge the PEFT layers and save the final models

### Deontology

```bash
python merge_peft_adapter.py \
--adapter_model_name=./llama-2-13b-deontology-reward-model_peft_last_checkpoint \
--base_model_name=./llama-2-13b-instruct-fine-tuned \
--output_name=./llama-2-13b-deontology-reward-model \
--push_to_s3 \
--aws_access_key_id= \
--aws_secret_access_key= \
--s3_object_prefix=llama-2/13b/
```

### Harmless

```bash
python merge_peft_adapter.py \
--adapter_model_name=./llama-2-13b-harmless-reward-model-adapter_peft_last_checkpoint \
--base_model_name=./llama-2-13b-instruct-fine-tuned \
--output_name=./llama-2-13b-harmless-reward-model \
--push_to_s3 \
--aws_access_key_id= \
--aws_secret_access_key= \
--s3_object_prefix=llama-2/13b/llama-2-13b-harmless-reward-model/
```

### Utilitarianism

```bash
python merge_peft_adapter.py \
--adapter_model_name=./llama-2-13b-utilitarianism-reward-model-adapter_peft_last_checkpoint \
--base_model_name=./llama-2-13b-instruct-fine-tuned \
--output_name=./llama-2-13b-utilitarianism-reward-model \
--push_to_s3 \
--aws_access_key_id= \
--aws_secret_access_key= \
--s3_object_prefix=llama-2/13b/llama-2-13b-utilitarianism-reward-model/
```

### Virtue

```bash
python merge_peft_adapter.py \
--adapter_model_name=./llama-2-13b-virtue-reward-model-adapter_peft_last_checkpoint \
--base_model_name=./llama-2-13b-instruct-fine-tuned \
--output_name=./llama-2-13b-virtue-reward-model \
--push_to_s3 \
--aws_access_key_id= \
--aws_secret_access_key= \
--s3_object_prefix=llama-2/13b/llama-2-13b-virtue-reward-model/
```



## 5. Reinforcement learning to create fine-tuned language models

### Deontology

```bash
accelerate launch \
--multi_gpu \
--num_machines 1 \
--num_processes 2 \
rl_training.py \
--model_name=./llama-2-13b-instruct-fine-tuned \
--reward_model_name=./llama-2-13b-deontology-reward-model \
--adafactor=False \
--tokenizer_name=./llama-2-13b-instruct-fine-tuned \
--save_freq=100 \
--output_max_length=128 \
--batch_size=8 \
--gradient_accumulation_steps=8 \
--batched_gen=True \
--ppo_epochs=4 \
--seed=0 \
--steps=4000 \
--learning_rate=1.4e-5 \
--early_stopping=False \
--log_with=wandb \
--wandb_project=rlm-llama-2-13b-rl-deontology \
--wandb_api_key= \
--wandb_notes="Reinforcement Learning Model, deontology reward model applied to llama-2-13b-instruct-fine-tuned" \
--output_dir=./rlm-llama-2-13b-rl-deontology \
--hugging_face_api_key=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
```

### Harmless

```bash
accelerate launch \
--multi_gpu \
--num_machines 1 \
--num_processes 2 \
rl_training.py \
--model_name=./llama-2-13b-instruct-fine-tuned \
--reward_model_name=./llama-2-13b-harmless-reward-model \
--adafactor=False \
--tokenizer_name=./llama-2-13b-instruct-fine-tuned \
--save_freq=100 \
--output_max_length=128 \
--batch_size=8 \
--gradient_accumulation_steps=8 \
--batched_gen=True \
--ppo_epochs=4 \
--seed=0 \
--steps=500 \
--learning_rate=1.4e-5 \
--early_stopping=False \
--log_with=wandb \
--wandb_project=rlm-llama-2-13b-rl-harmless \
--wandb_api_key= \
--wandb_notes="Reinforcement Learning Model, harmless reward model applied to llama-2-13b-instruct-fine-tuned" \
--output_dir=./rlm-llama-2-13b-rl-harmless \
--hugging_face_api_key=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
```


### Utilitarianism

```bash
accelerate launch \
--multi_gpu \
--num_machines 1 \
--num_processes 2 \
rl_training.py \
--model_name=./llama-2-13b-instruct-fine-tuned \
--reward_model_name=./llama-2-13b-utilitarianism-reward-model \
--adafactor=False \
--tokenizer_name=./llama-2-13b-instruct-fine-tuned \
--save_freq=100 \
--output_max_length=128 \
--batch_size=8 \
--gradient_accumulation_steps=8 \
--batched_gen=True \
--ppo_epochs=4 \
--seed=0 \
--steps=4000 \
--learning_rate=1.4e-5 \
--early_stopping=False \
--log_with=wandb \
--wandb_project=rlm-llama-2-13b-rl-utilitarianism \
--wandb_api_key= \
--wandb_notes="Reinforcement Learning Model, utilitarianism reward model applied to llama-2-13b-instruct-fine-tuned" \
--output_dir=./rlm-llama-2-13b-rl-utilitarianism \
--hugging_face_api_key=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
```

### Harmless

```bash
accelerate launch \
--multi_gpu \
--num_machines 1 \
--num_processes 2 \
rl_training.py \
--model_name=./llama-2-13b-instruct-fine-tuned \
--reward_model_name=./llama-2-13b-harmless-reward-model \
--adafactor=False \
--tokenizer_name=./llama-2-13b-instruct-fine-tuned \
--save_freq=100 \
--output_max_length=128 \
--batch_size=8 \
--gradient_accumulation_steps=8 \
--batched_gen=True \
--ppo_epochs=4 \
--seed=0 \
--steps=4000 \
--learning_rate=1.4e-5 \
--early_stopping=False \
--log_with=wandb \
--wandb_project=rlm-llama-2-13b-rl-harmless \
--wandb_api_key= \
--wandb_notes="Reinforcement Learning Model, harmless reward model applied to llama-2-13b-instruct-fine-tuned" \
--output_dir=./rlm-llama-2-13b-rl-harmless \
--hugging_face_api_key=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
```



### Virtue

```bash
accelerate launch \
--multi_gpu \
--num_machines 1 \
--num_processes 2 \
rl_training.py \
--model_name=./llama-2-13b-instruct-fine-tuned \
--reward_model_name=./llama-2-13b-virtue-reward-model \
--adafactor=False \
--tokenizer_name=./llama-2-13b-instruct-fine-tuned \
--save_freq=100 \
--output_max_length=128 \
--batch_size=8 \
--gradient_accumulation_steps=8 \
--batched_gen=True \
--ppo_epochs=4 \
--seed=0 \
--steps=4000 \
--learning_rate=1.4e-5 \
--early_stopping=False \
--log_with=wandb \
--wandb_project=rlm-llama-2-13b-rl-virtue \
--wandb_api_key= \
--wandb_notes="Reinforcement Learning Model, virtue reward model applied to llama-2-13b-instruct-fine-tuned" \
--output_dir=./rlm-llama-2-13b-rl-virtue \
--hugging_face_api_key=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
```


## 6. Merge the PEFT layers and save the final models

### Deontology

```bash
python merge_peft_adapter.py \
--adapter_model_name=./rlm-llama-2-13b-rl-deontologystep_400 \
--base_model_name=./llama-2-13b-instruct-fine-tuned \
--output_name=./rlm-llama-2-13b-rl-deontology-step_400 \
--push_to_s3 \
--aws_access_key_id= \
--aws_secret_access_key= \
--s3_object_prefix=llama-2/13b/rlm-llama-2-13b-rl-deontology-step_400/
```

### Utilitarianism

```bash
python merge_peft_adapter.py \
--adapter_model_name=./rlm-llama-2-13b-rl-utilitarianismstep_400 \
--base_model_name=./llama-2-13b-instruct-fine-tuned \
--output_name=./rlm-llama-2-13b-rl-utilitarianism-step_400 \
--push_to_s3 \
--aws_access_key_id= \
--aws_secret_access_key= \
--s3_object_prefix=llama-2/13b/rlm-llama-2-13b-rl-utilitarianism-step_400/
```

### Harmless

```bash
python merge_peft_adapter.py \
--adapter_model_name=./rlm-llama-2-13b-rl-harmlessstep_400 \
--base_model_name=./llama-2-13b-instruct-fine-tuned \
--output_name=./rlm-llama-2-13b-rl-harmless-step_400 \
--push_to_s3 \
--aws_access_key_id= \
--aws_secret_access_key= \
--s3_object_prefix=llama-2/13b/rlm-llama-2-13b-rl-harmless-step_400/
```

### Virtue

```bash
python merge_peft_adapter.py \
--adapter_model_name=./rlm-llama-2-13b-rl-virtuestep_400 \
--base_model_name=./llama-2-13b-instruct-fine-tuned \
--output_name=./rlm-llama-2-13b-rl-virtue-step_400 \
--push_to_s3 \
--aws_access_key_id= \
--aws_secret_access_key= \
--s3_object_prefix=llama-2/13b/rlm-llama-2-13b-rl-virtue-step_400/
```


From S3:
msc-project-rlhf/llama-2/13b/llama-2-13b-instruct-fine-tuned
msc-project-rlhf/llama-2/13b/rlm-llama-2-13b-rl-deontology-step_400/
msc-project-rlhf/llama-2/13b/rlm-llama-2-13b-rl-utilitarianism-step_400/
msc-project-rlhf/llama-2/13b/rlm-llama-2-13b-rl-harmless-step_400/
msc-project-rlhf/llama-2/13b/rlm-llama-2-13b-rl-virtue-step_400/



Instances

export NCCL_P2P_DISABLE=1
accelerate test

deontology
ssh -p 48105 root@89.37.121.214

utilitarianism
ssh -p 48171 root@89.37.121.214

harmless
ssh -p 44178 root@46.214.200.185

virtue
ssh -p 40113 root@93.114.160.254

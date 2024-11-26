# Benchmarks

# Benchmarks

ssh -p 20627 root@213.108.196.111

git clone gitlab.com/msc-final-project/llama2-rlhf.git
pip install accelerate
pip install bitsandbytes

curl -Ok https://storage.googleapis.com/ai2-mosaic-public/projects/scruples/v1.0/data/anecdotes.tar.gz
tar -xvf anecdotes.tar.gz



Transfer from S3 to the instance:

msc-project-rlhf/llama-2/13b/llama-2-13b-instruct-fine-tuned
/root/models/llama-2-13b-instruct-fine-tuned

/root/results/rlm-llama-2-13b-rl-utilitarianism-step_400_scruples.gz
msc-project-rlhf/benchmarks/results/rlm-llama-2-13b-rl-utilitarianism-step_400_scruples.gz


/workspace/rlm-llama-2-13b-rl-utilitarianism-step_400_scruples-predictions.gz
msc-project-rlhf/benchmarks/results/rlm-llama-2-13b-rl-utilitarianism-step_400_scruples-predictions.gz



For each model:

```

huggingface-cli login

python3 scruples_anecdotes.py \
--model_path meta-llama/Llama-2-13b-hf \
--file_out /workspace/llama-2-13b-hf_scruples-predictions \
--num_samples 500


msc-project-rlhf/llama-2/13b/rlm-llama-2-13b-rl-virtue-step_400
/workspace/rlm-llama-2-13b-rl-virtue-step_400


python3 scruples_anecdotes.py \
--model_path /workspace/rlm-llama-2-13b-rl-virtue-step_400 \
--file_out /workspace/rlm-llama-2-13b-rl-virtue-step_400_predictions \
--num_samples 500



llama-2-13b
llama-2-13b-instruct-fine-tuned - DONE
rlm-llama-2-13b-rl-deontology-step_400 - DONE
rlm-llama-2-13b-rl-harmless-step_400 - DONE
rlm-llama-2-13b-rl-utilitarianism-step_400 - DONE
rlm-llama-2-13b-rl-virtue-step_400 - DONE


```

12 hours per model?
how balanced is the test set between right and wrong?




calculate accuracy - of those that the model did not swerve on the answer
calculate Matthews Correlation Coefficient
calculate how significant the diference?

# Other benchmarks

## Eleuther AI - LM evaluation harness

crowspairs - bias
toxigen - toxicity
wsc273 - bias

msc-project-rlhf/llama-2/13b/rlm-llama-2-13b-rl-virtue-step_400
/workspace/rlm-llama-2-13b-rl-virtue-step_400


# results_crows_pairs_english_wsc273

python main.py \
--model hf \
--model_args pretrained=meta-llama/Llama-2-13b-hf,parallelize=False,load_in_8bit=True \
--tasks crows_pairs_english,wsc273 \
--output_path /workspace/results/llama-2-13b-hf \
--device cuda:0

meta-llama/Llama-2-13b-hf
llama-2-13b-instruct-fine-tuned - DONE
rlm-llama-2-13b-rl-deontology-step_400 - DONE
rlm-llama-2-13b-rl-harmless-step_400 - DONE
rlm-llama-2-13b-rl-utilitarianism-step_400 - DONE
rlm-llama-2-13b-rl-virtue-step_400 - DONE



# toxigen

msc-project-rlhf/llama-2/13b/rlm-llama-2-13b-rl-virtue-step_400
/workspace/rlm-llama-2-13b-rl-virtue-step_400

python main.py \
--model hf \
--model_args pretrained=/workspace/rlm-llama-2-13b-rl-virtue-step_400,parallelize=False,load_in_8bit=True \
--tasks toxigen \
--output_path /workspace/results_toxigen/rlm-llama-2-13b-rl-virtue-step_400 \
--device cuda:0


meta-llama/Llama-2-13b-hf - DONE
llama-2-13b-instruct-fine-tuned - DONE
rlm-llama-2-13b-rl-deontology-step_400 - DONE
rlm-llama-2-13b-rl-harmless-step_400 - DONE
rlm-llama-2-13b-rl-utilitarianism-step_400 - DONE
rlm-llama-2-13b-rl-virtue-step_400 - DONE



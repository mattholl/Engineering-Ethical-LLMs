# Engineering Ethical LLMs: Analysing the Effectiveness of Datasets To Steer Generative Language Models

This repo contains code that I developed to support my investigation of LLM steering with reinforcement learning. The project was submitted in fulfillment of MSc Data science and Artificial Intelligence.

## Abstract

> Most of the recent high profile developments in artificial intelligence (AI) have been in relation to large language models. At their most basic, these are transformer-based neural networks that are trained to predict the next word in a sequence, but when scaled up, the model is endowed with the ability to generate text indistinguishable from a human. These impressive abilities come at a price: language models freely generate toxic content and propagates biases, all of which can be found somewhere in the vast, web-scale, collections of their training data. As language models gain exponentially in influence the question of engineering their ethical behaviour will become crucial in their usability.
>Conversational assistant AI services like ChatGPT are steered using techniques like reinforce- ment learning from human feedback (RLHF). Recent research emphasises the concepts of ‘helpful- ness’ and ‘harmlessness’ as targets for AI alignment.
>This paper reports on a project which was carried out to analyse and quantitatively compare a selection of ethically motivated datasets. The Llama 2 13B pre-trained model was fine-tuned into four versions each with a deontology, harmlessness, utilitarianism, or virtue ethics dataset providing the reward signal. The feedback model training data was sourced from the ETHICS dataset.

>The performance of these four reward models was measured on a binary labelling task using the Scruples dataset. By calculating the Matthews Correlation Coefficient (MCC) value of each as a classifier it will be shown that none of the datasets performs significantly better than another. However, the base LLM does learn to elicit a higher reward value from harmlessness model. Per- formance of each model on the ToxiGen and Winograd benchmarks are almost equal.
> This paper argues that the collection of real human preference data is paramount to effectively steer a LLM, and paves the way for further investigation into the influence of concepts from ethics on LLM alignment.


## llm-fine-tuning

Supervised fine-tuning of Llama2 7B base model using the [OpenAssistant Conversations Dataset (OASST1) ](https://huggingface.co/datasets/timdettmers/openassistant-guanaco).

## llama2-rlhf

LLM fine-tuning with RLHF using the trl library.

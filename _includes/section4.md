
In pre-training, the goal is basically "predict the next token on random internet text". While the resulting "base" models are still useful in some contexts, their outputs are often chaotic or "unaligned", and they may not respect the format of a back-and-forth conversation. Here we'll look at a set of techniques for going from these base models to ones resembling the friendly chatbots and assistants we're more familiar with. A great companion resource, especially for this section, is Maxime Labonne's interactive [LLM course](https://github.com/mlabonne/llm-course?tab=readme-ov-file) on Github.

<h2>Instruct Fine-Tuning</h2>

Instruct fine-tuning (or "instruction tuning", or "supervised finetuning", or "chat tuning" -- the boundaries here are a bit fuzzy) is the primary technique used (at least initially) for coaxing LLMs to conform to a particular style or format. Here, data is presented as a sequence of (input, output) pairs where the input is a user question to answer, and the model's goal is to predict the output -- typically this also involves adding special "start"/"stop"/"role" tokens and other masking techniques, enabling the model to "understand" the difference between the user's input and its own outputs. This technique is also widely used for task-specific finetuning on datasets with a particular kind of problem structure (e.g. translation, math, general question-answering).

See this [blog post](https://newsletter.ruder.io/p/instruction-tuning-vol-1) from Sebastian Ruder or this [video](https://www.youtube.com/watch?v=YoVek79LFe0) from Shayne Longpre for short overviews.

<h2>Low-Rank Adapters (LoRA)</h2>

While pre-training (and "full finetuning") requires applying gradient updates to all parameters of a model, this is typically impractical on consumer GPUs or home setups; fortunately, it's often possible to significantly reduce the compute requirements by using parameter-efficient finetuning (PEFT) techniques like Low-Rank Adapters (LoRA). This can enable competitive performance even with relatively small datasets, particularly for application-specific use cases. The main idea behind LoRA is to train each weight matrix in a low-rank space by "freezing" the base matrix and training a factored representation with much smaller inner dimension, which is then added to the base matrix.

Resources:
- LoRA paper walkthrough [(video, part 1)](https://youtu.be/dA-NhCtrrVE?si=TpJkPfYxngQQ0iGj)
- LoRA code demo [(video, part 2)](https://youtu.be/iYr1xZn26R8?si=aG0F8ws9XslpZ4ur)
- ["Parameter-Efficient LLM Finetuning With Low-Rank Adaptation"](https://sebastianraschka.com/blog/2023/llm-finetuning-lora.html ) by Sebastian Raschka
- ["Practical Tips for Finetuning LLMs Using LoRA"](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms) by Sebastian Raschka

Additionally, an "decomposed" LoRA variant called DoRA has been gaining popularity in recent months, often yielding performance improvements; see this [post](https://magazine.sebastianraschka.com/p/lora-and-dora-from-scratch) from Sebastian Raschka for more details.


<h2>Reward Models and RLHF</h2>

One of the most prominent techniques for "aligning" a language model is Reinforcement Learning from Human Feedback (RLHF); here, we typically assume that an LLM has already been instruction-tuned to respect a chat style, and that we additionally have a "reward model" which has been trained on human preferences. Given pairs of differing outputs to an input, where a preferred output has been chosen by a human, the learning objective of the reward model is to predict the preferred output, which involves implicitly learning preference "scores". This allows bootstrapping a general representation of human preferences (at least with respect to the dataset of output pairs), which can be used as a "reward simulator" for continual training of a LLM using RL policy gradient techniques like PPO. 

For overviews, see the posts ["Illustrating Reinforcement Learning from Human Feedback (RLHF)"](https://huggingface.co/blog/rlhf) from Hugging Face and ["Reinforcement Learning from Human Feedback"](https://huyenchip.com/2023/05/02/rlhf.html) from Chip Huyen, and/or this [RLHF talk](https://www.youtube.com/watch?v=2MBJOuVq380) by Nathan Lambert. Further, this [post](https://sebastianraschka.com/blog/2024/research-papers-in-march-2024.html) from Sebastian Raschka dives into RewardBench, and how reward models themselves can be evaluated against each other by leveraging ideas from Direct Preference Optimization, another prominent approach for aligning LLMs with human preference data. 

<h2>Direct Preference Optimization Methods</h2>

The space of alignment algorithms seems to be following a similar trajectory as we saw with stochastic optimization algorithms a decade ago. In this an analogy, RLHF is like SGD --- it works, it's the original, and it's also become kind of a generic "catch-all" term for the class of algorithms that have followed it. Perhaps DPO is AdaGrad, and in the year since its release there's been a rapid wave of further algorithmic developments along the same lines (KTO, IPO, ORPO, etc.), whose relative merits are still under active debate. Maybe a year from now, everyone will have settled on a standard approach which will become the "Adam" of alignment.


For an overview of the theory behind DPO see this [blog post](https://towardsdatascience.com/understanding-the-implications-of-direct-preference-optimization-a4bbd2d85841) Matthew Gunton; this [blog post](https://huggingface.co/blog/dpo-trl) from Hugging Face features some code and demonstrates how to make use of DPO in practice. Another [blog post](https://huggingface.co/blog/pref-tuning) from Hugging Face also discusses tradeoffs between a number of the DPO-flavored methods which have emerged in recent months. 


<h2>Context Scaling</h2>

Beyond task specification or alignment, another common goal of finetuning is to increase the effective context length of a model, either via additional training, adjusting parameters for positional encodings, or both. Even if adding more tokens to a model's context can "type-check", training on additional longer examples is generally necessary if the model may not have seen such long sequences during pretraining.

Resources:
- ["Scaling Rotational Embeddings for Long-Context Language Models"](https://gradient.ai/blog/scaling-rotational-embeddings-for-long-context-language-models) by Gradient AI
- ["Extending the RoPE"](https://blog.eleuther.ai/yarn/) by Eleuther AI, introducing the YaRN method for increased context via attention temperature scaling 
- ["Everything About Long Context Fine-tuning"](https://huggingface.co/blog/wenbopan/long-context-fine-tuning) by Wenbo Pan



<h2>Distillation and Merging</h2>

Here we'll look at two very different methods of consolidating knowledge across LLMs --- distillation and merging. Distillation was first popularized for BERT models, where the goal is to "distill" the knowledge and performance of a larger model into a smaller one (at least for some tasks) by having it serve as a "teacher" during the smaller model's training, bypassing the need for large quantities of human-labeled data. 

Some resources on distillation:
- ["Smaller, faster, cheaper, lighter: Introducing DistilBERT, a distilled version of BERT"](https://medium.com/huggingface/distilbert-8cf3380435b5) from Hugging Face
- ["LLM distillation demystified: a complete guide"](https://snorkel.ai/llm-distillation-demystified-a-complete-guide/) from Snorkel AI 
- [“Distilling Step by Step” blog](https://blog.research.google/2023/09/distilling-step-by-step-outperforming.html) from Google Research 

Merging, on the other hand, is much more of a "wild west" technique, largely used by open-source engineers who want to combine the strengths of multiple finetuning efforts. It's kind of wild to me that it works at all, and perhaps grants some credence to "linear representation hypotheses" (which will appear in the next section when we discuss interpretability). The idea is basically to take two different finetunes of the same base model and just average their weights. No training required. Technically, it's usually "spherical interpolation" (or "slerp"), but this is pretty much just fancy averaging with a normalization step. For more details, see the post [Merge Large Language Models with mergekit](https://huggingface.co/blog/mlabonne/merge-models) by Maxime Labonne.
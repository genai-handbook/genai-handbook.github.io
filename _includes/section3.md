
In this section, we'll explore a number of concepts which will take us from the decoder-only Transformer architecture towards understanding the implementation choices and tradeoffs behind many of today's frontier LLMs. If you first want a birds-eye view the of topics in section and some of the following ones, the post ["Understanding Large Language Models"](https://magazine.sebastianraschka.com/p/understanding-large-language-models) by Sebastian Raschka is a nice summary of what the LLM landscape looks like (at least up through mid-2023). 

<h2>Tokenization</h2>

Character-level tokenization (like in several of the Karpathy videos) tends to be inefficient for large-scale Transformers vs. word-level tokenization, yet naively picking a fixed "dictionary" (e.g. Merriam-Webster) of full words runs the risk of encountering unseen words or misspellings at inference time. Instead, the typical approach is to use subword-level tokenization to "cover" the space of possible inputs, while maintaining the efficiency gains which come from a larger token pool, using algorithms like Byte-Pair Encoding (BPE) to select the appropriate set of tokens. If you've ever seen Huffman coding in an introductory algorithms class I think it's a somewhat useful analogy for BPE here, although the input-output format is notably different, as we don't know the set of "tokens" in advance. I'd recommend watching Andrej Karpathy's [video](https://www.youtube.com/watch?v=zduSFxRajkE) on tokenization and checking out this tokenization [guide](https://blog.octanove.org/guide-to-subword-tokenization/) from Masato Hagiwara.

<h2>Positional Encoding</h2>

As we saw in the past section, Transformers don't natively have the same notion of adjacency or position within a context windows (in contrast to RNNs), and position must instead represented with some kind of vector encoding. While this could be done naively with something like one-hot encoding, this is impractical for context-scaling and suboptimal for learnability, as it throws away notions of ordinality. Originally, this was done with sinusoidal positional encodings, which may feel reminiscent of Fourier features if you're familiar; the most popular implementation of this type of approach nowadays is likely Rotary Positional Encoding, or RoPE, which tends to be more stable and faster to learn during training.

Resources:
- [blog post](https://harrisonpim.com/blog/understanding-positional-embeddings-in-transformer-models) by Harrison Pim on intution for positional encodings
- [blog post](https://machinelearningmastery.com/a-gentle-introduction-to-positional-encoding-in-transformer-models-part-1/) by Mehreen Saeed on the original Transformer positional encodings
- [blog post](https://blog.eleuther.ai/rotary-embeddings/) on RoPE from Eleuther AI
original Transformer: https://kazemnejad.com/blog/transformer_architecture_positional_encoding/ 
https://machinelearningmastery.com/a-gentle-introduction-to-positional-encoding-in-transformer-models-part-1/
- [animated video](https://www.youtube.com/watch?v=GQPOtyITy54) from DeepLearning Hero

<h2>Pretraining Recipes</h2>

Once you've committed to pretraining a LLM of a certain general size on a particular corpus of data (e.g Common Crawl, FineWeb), there are still a number of choices to make before you're ready to go:

- Attention mechanisms (multi-head, multi-query, grouped-query)
- Activations (ReLU, GeLU, SwiGLU)
- Optimizers, learning rates, and schedulers (AdamW, warmup, cosine decay)
- Dropout?
- Hyperparameter choices and search strategies
- Batching, parallelization strategies, gradient accumulation
- How long to train for, how often to repeat data
- ...and many other axes of variation

As far as I can tell, there's not a one-size-fits-all rule book for how to go about this, but I'll share a handful of worthwhile resources to consider, depending on your interests:

- While it predates the LLM era, the blog post ["A Recipe for Training Neural Networks"](https://karpathy.github.io/2019/04/25/recipe/) is a great starting point for framing this problem, as many of these questions are relevant throughout deep learning.
- ["The Novice's LLM Training Guide"](https://rentry.org/llm-training) by Alpin Dale, discussing hyperparameter choices in practice, as well as the finetuning techniques we'll see in future sections. 
- ["How to train your own Large Language Models"](https://blog.replit.com/llm-training) from Replit has some nice discussions on data pipelines and evaluations for training.
- For understanding attention mechanism tradeoffs, see the post ["Navigating the Attention Landscape: MHA, MQA, and GQA Decoded"](https://iamshobhitagarwal.medium.com/navigating-the-attention-landscape-mha-mqa-and-gqa-decoded-288217d0a7d1) by Shobhit Agarwal.
- For discussion of "popular defaults", see the post ["The Evolution of the Modern Transformer: From ‘Attention Is All You Need’ to GQA, SwiGLU, and RoPE"](https://deci.ai/blog/evolution-of-modern-transformer-swiglu-rope-gqa-attention-is-all-you-need/) from Deci AI.
- For details on learning rate scheduling, see [Chapter 12.11](https://d2l.ai/chapter_optimization/lr-scheduler.html) from the d2l.ai book.
- For discussion of some controversy surrounding reporting of "best practices", see this [post](https://blog.eleuther.ai/nyt-yi-34b-response/) from Eleuther AI.


<h2>Distributed Training and FSDP</h2>

There are a number of additional challenges associated with training models which are too large to fit on individual GPUs (or even multi-GPU machines), typically necessitating the use of distributed training protocols like Fully Sharded Data Parallelism (FSDP), in which models can be co-located across machines during training. It's probably worth also understanding its precursor Distributed Data Parallelism (DDP), which is covered in the first post linked below.
 
Resources:
- official FSDP [blog post](https://engineering.fb.com/2021/07/15/open-source/fsdp/) from Meta (who pioneered the method)
https://sumanthrh.com/post/distributed-and-efficient-finetuning/ 
- [blog post](https://blog.clika.io/fsdp-1/) on FSDP by Bar Rozenman, featuring many excellent visualizations
- [report](https://www.yitay.net/blog/training-great-llms-entirely-from-ground-zero-in-the-wilderness) from Yi Tai on the challenges of pretraining a model in a startup environment
- [technical blog](https://www.answer.ai/posts/2024-03-14-fsdp-qlora-deep-dive.html) from Answer.AI on combining FSDP with parameter-efficient finetuning techniques for use on consumer GPUs


<h2>Scaling Laws</h2>

It's useful to know about scaling laws as a meta-topic which comes up a lot in discussions of LLMs (most prominently in reference to the "Chinchilla" [paper](https://arxiv.org/abs/2203.15556)), more so than any particular empirical finding or technique. In short, the performance which will result from scaling up the model, data, and compute used for training a language model results in fairly reliable predictions for model loss. This then enables calibration of optimal hyperparameter settings without needing to run expensive grid searches. 

Resources:
- [Chinchilla Scaling Laws for Large Language Models](https://medium.com/@raniahossam/chinchilla-scaling-laws-for-large-language-models-llms-40c434e4e1c1) (blog overview by Rania Hossam)
- [New Scaling Laws for LLMs](https://www.lesswrong.com/posts/midXmMb2Xg37F2Kgn/new-scaling-laws-for-large-language-models) (discussion on LessWrong)
- [Chinchilla's Wild Implications](https://www.lesswrong.com/posts/6Fpvch8RR29qLEWNH/chinchilla-s-wild-implications) (post on LessWrong)
- [Chinchilla Scaling: A Replication Attempt](https://epochai.org/blog/chinchilla-scaling-a-replication-attempt) (potential issues with Chinchilla findings)
- [Scaling Laws and Emergent Properties](https://cthiriet.com/blog/scaling-laws) (blog post by Clément Thiriet)
- ["Scaling Language Models"](https://www.youtube.com/watch?v=UFem7xa3Q2Q) (video lecture, Stanford CS224n)

<h2>Mixture-of-Experts</h2>

While many of the prominent LLMs (such as Llama3) used today are "dense" models (i.e. without enforced sparsification), Mixture-of-Experts (MoE) architectures are becoming increasingly popular for navigating tradeoffs between "knowledge" and efficiency, used perhaps most notably in the open-weights world by Mistral AI's "Mixtral" models (8x7B and 8x22B), and [rumored](https://the-decoder.com/gpt-4-architecture-datasets-costs-and-more-leaked/) to be used for GPT-4. In MoE models, only a fraction of the parameters are "active" for each step of inference, with trained router modules for selecting the parallel "experts" to use at each layer. This allows models to grow in size (and perhaps "knowlege" or "intelligence") while remaining efficient for training or inference compared to a comparably-sized dense model. 

See this [blog post](https://huggingface.co/blog/moe) from Hugging Face for a technical overview, and this [video](https://www.youtube.com/watch?v=0U_65fLoTq0) from Trelis Research for a visualized explainer.



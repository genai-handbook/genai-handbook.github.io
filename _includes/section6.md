
Here we'll look at a handful of techniques for improving the speed and efficiency of inference from pre-trained Transformer language models, most of which are fairly widely used in practice. It's worth first reading this short Nvidia [blog post](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/) for a crash course in several of the topics we'll look at (and a number of others).

<h2>Parameter Quantization</h2>

With the rapid increase in parameter counts for leading LLMs and difficulties (both in cost and availability) in acquiring GPUs to run models on, there's been a growing interest in quantizing LLM weights to use fewer bits each, which can often yield comparable output quality with a 50-75% (or more) reduction in required memory. Typically this shouldn't be done naively; Tim Dettmers, one of the pioneers of several modern quantization methods (LLM.int8(), QLoRA, bitsandbytes) has a great [blog post](https://timdettmers.com/2022/08/17/llm-int8-and-emergent-features/) for understanding quantization principles, and the need for mixed-precision quantization as it relates to emergent features in large-model training. Other popular methods and formats are GGUF (for llama.cpp), AWQ, HQQ, and GPTQ; see this [post](https://www.tensorops.ai/post/what-are-quantized-llms) from TensorOps for an overview, and this [post](https://www.maartengrootendorst.com/blog/quantization/) from Maarten Grootendorst for a discussion of their tradeoffs.


In addition to enabling inference on smaller machines, quantization is also popular for parameter-efficient training; in QLoRA, most weights are quantized to 4-bit precision and frozen, while active LoRA adapters are trained in 16-bit precision. See this [talk](https://www.youtube.com/watch?v=fQirE9N5q_Y) from Tim Dettmers, or this [blog](https://huggingface.co/blog/4bit-transformers-bitsandbytes) from Hugging Face for overviews. This [blog post](https://www.answer.ai/posts/2024-03-14-fsdp-qlora-deep-dive.html) from Answer.AI also shows how to combine QLoRA with FSDP for efficient finetuning of 70B+ parameter models on consumer GPUs.


<h2>Speculative Decoding</h2>

The basic idea behind speculative decoding is to speed up inference from a larger model by primarily sampling tokens from a much smaller model and occasionally applying corrections (e.g. every _N_ tokens) from the larger model whenever the output distributions diverge. These batched consistency checks tend to be much faster than sampling _N_ tokens directly, and so there can be large overall speedups if the token sequences from smaller model only diverge periodically.

See this [blog post](https://jaykmody.com/blog/speculative-sampling/) from Jay Mody for a walkthrough of the original paper, and this PyTorch [article](https://pytorch.org/blog/hitchhikers-guide-speculative-decoding/) for some evaluation results. There's a nice [video](https://www.youtube.com/watch?v=hm7VEgxhOvk) overview from Trelis Research as well.

<h2>FlashAttention</h2>

Computing attention matrices tends to be a primary bottleneck in inference and training for Transformers, and FlashAttention has become one of the most widely-used techniques for speeding it up. In contrast to some of the techniques we'll see in [Section 7](#s7) which _approximate_ attention with a more concise representation (occurring some representation error as a result), FlashAttention is an _exact_ representation whose speedup comes from hardware-aware impleemntation. It applies a few tricks --- namely, tiling and recomputation --- to decompose the expression of attention matrices, enabling significantly reduced memory I/O and faster wall-clock performance (even with slightly increasing the required FLOPS).

Resources:
- [Talk](https://www.youtube.com/watch?v=gMOAud7hZg4) by Tri Dao (author of FlashAttention)
- [ELI5: FlashAttention](https://gordicaleksa.medium.com/eli5-flash-attention-5c44017022ad ) by Aleksa Gordić


<h2>Key-Value Caching and Paged Attention</h2>

As noted in the [NVIDIA blog](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/) referenced above, key-value caching is fairly standard in Transformer implementation matrices to avoid redundant recomputation of attention. This enables a tradeoff between speed and resource utilization, as these matrices are kept in GPU VRAM. While managing this is fairly straightforward for a single "thread" of inference, a number of complexities arise when considering parallel inference or multiple users for a single hosted model instance. How can you avoid recomputing values for system prompts and few-shot examples? When should you evict cache elements for a user who may or may not want to continue a chat session? PagedAttention and its popular implementation [vLLM](https://docs.vllm.ai/en/stable/) addresses this by leveraging ideas from classical paging in operating systems, and has become a standard for self-hosted multi-user inference servers.

Resources:
- [The KV Cache: Memory Usage in Transformers](https://www.youtube.com/watch?v=80bIUggRJf4) (video, Efficient NLP)
- [Fast LLM Serving with vLLM and PagedAttention](https://www.youtube.com/watch?v=5ZlavKF_98U) (video, Anyscale)
- vLLM [blog post](https://blog.vllm.ai/2023/06/20/vllm.html)


<h2>CPU Offloading</h2>

The primary method used for running LLMs either partially or entirely on CPU (vs. GPU) is llama.cpp. See [here](https://www.datacamp.com/tutorial/llama-cpp-tutorial) for a high-level overview; llama.cpp serves as the backend for a number of popular self-hosted LLM tools/frameworks like LMStudio and Ollama. Here's a [blog post](https://justine.lol/matmul/) with some technical details about CPU performance improvements. 
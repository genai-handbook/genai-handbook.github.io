
A major bottleneck in scaling both the size and context length of Transformers is the quadratic nature of attention, in which all pairs of token interactions are considered. Here we'll look at a number of approaches for circumventing this, ranging from those which are currently widely used to those which are more exploratory (but promising) research directions.

<h2>Sliding Window Attention</h2>

Introduced in the "Longformer" [paper](https://arxiv.org/abs/2004.05150), sliding window attention acts as a sub-quadratic drop-in replacement for standard attention which allows attending only to a sliding window (shocking, right?) of recent tokens/states rather than the entire context window, under the pretense that vectors for these states have already attended to earlier ones and thus have sufficient representational power to encode relevant pieces of early context. Due to its simplicity, it's become one of the more widely adopted approaches towards sub-quadratic scaling, and is used in Mistral's popular Mixtral-8x7B model (among others).

Resources:
- ["What is Sliding Window Attention?"](https://klu.ai/glossary/sliding-window-attention) (blog post by Stephen M. Walker)
- ["Sliding Window Attention"](https://medium.com/@manojkumal/sliding-window-attention-565f963a1ffd) (blog post by Manoj Kumal)
- ["Longformer: The Long-Document Transformer"](https://www.youtube.com/watch?v=_8KNb5iqblE) (video by Yannic Kilcher)

<h2>Ring Attention</h2>

Another modification to standard attention mechanisms, Ring Attention enables sub-quadratic full-context interaction via incremental computation with a "message-passing" structure, wherein "blocks" of context communicate with each other over a series of steps rather than all at once. Within each block, the technique is essentially classical attention. While largely a research direction rather than standard technique at least within the open-weights world, Google's Gemini is [rumored](https://www.reddit.com/r/MachineLearning/comments/1arj2j8/d_gemini_1m10m_token_context_window_how/) to possibly be using Ring Attention in order to enable its million-plus-token context.

Resources:

- ["Breaking the Boundaries: Understanding Context Window Limitations and the idea of Ring Attention"](https://medium.com/@tanuj22july/breaking-the-boundaries-understanding-context-window-limitations-and-the-idea-of-ring-attention-170e522d44b2) (blog post, Tanuj Sharma)
- ["Understanding Ring Attention: Building Transformers With Near-Infinite Context"](https://www.e2enetworks.com/blog/understanding-ring-attention-building-transformers-with-near-infinite-context) (blog post, E2E Networks)
- ["Ring Attention Explained"](https://www.youtube.com/watch?v=jTJcP8iyoOM) (video)


<h2>Linear Attention (RWKV)</h2>

The Receptance-Weighted Key Value (RWKV) architecture is a return to the general structure of RNN models (e.g LSTMs), with modifications to enable increased scaling and a _linear_ attention-style mechanism which supports recurrent "unrolling" of its representation (allowing constant computation per output token as context length scales). 
 
Resources:
- [](https://huggingface.co/blog/rwkv) (Huggingface blog)
- ["The RWKV language model: An RNN with the advantages of a transformer" - Pt. 1](https://johanwind.github.io/2023/03/23/rwkv_overview.html) (blog post, Johan Wind)
- ["How the RWKV language model works" - Pt. 2](https://johanwind.github.io/2023/03/23/rwkv_details.html)
- ["RWKV: Reinventing RNNs for the Transformer Era (Paper Explained)"](https://www.youtube.com/watch?v=x8pW19wKfXQ) (video, Yannic Kilcher)

<h2>Structured State Space Models</h2>

Structured State Space Models (SSMs) have become one of the most popular alternatives to Transformers in terms of current research focus, with several notable variants (S4, Hyena, Mamba/S6, Jamba, Mamba-2), but are somewhat notorious for their complexity. 
The architecture draws inspiration from classical control theory and linear time-invariant systems, with a number of optimizations to translate from continuous to discrete time, and to avoid dense representations of large matrices. They support both recurrent and convolutional representations, which allows efficiency gains both for training and at inference, and many variants require carefully-conditioned "hidden state matrix" representations to support "memorization" of context without needing all-pairs attention. SSMs also seem to be becoming more practical at scale, and have recently resulted in breakthrough speed improvements for high-quality text to speech (via [Cartesia AI](https://www.cartesia.ai/), founded by the inventors of SSMs).

The best explainer out there is likely ["The Annotated S4"](https://srush.github.io/annotated-s4/), focused on the S4 paper from which SSMs originated. The post ["A Visual Guide to Mamba and State Space Models"](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mamba-and-state) is great for intuitions and visuals with slightly less math, and Yannic Kilcher has a nice [video](https://www.youtube.com/watch?v=9dSkvxS2EB0) on SSMs as well.


Recently, the Mamba authors released their follow-up "Mamba 2" paper, and their accompanying series of blog posts discusses some newly-uncovered connections between SSM representations and linear attention which may be interesting:

- [State Space Duality (Mamba-2) Part I - The Model](https://tridao.me/blog/2024/mamba2-part1-model/)
- [State Space Duality (Mamba-2) Part II - The Theory](https://tridao.me/blog/2024/mamba2-part2-theory/)
- [State Space Duality (Mamba-2) Part III - The Algorithm](https://tridao.me/blog/2024/mamba2-part3-algorithm/)
- [State Space Duality (Mamba-2) Part IV - The Systems](https://tridao.me/blog/2024/mamba2-part4-systems/) 

<h2>HyperAttention</h2>

Somewhat similar to RWKV and SSMs, HyperAttention is another proposal for achieving near-linear scaling for attention-like mechanisms, relying on locality-sensitive hashing (think vector DBs) rather than recurrent representations. I don't see it discussed as much as the others, but it may be worth being aware of nonetheless. 

For an overview, see this [blog post](https://medium.com/@yousra.aoudi/linear-time-magic-how-hyperattention-optimizes-large-language-models-b691c0e2c2b0) by Yousra Aoudi and short explainer [video](https://www.youtube.com/watch?v=uvix7XwAjOg) by Tony Shin.
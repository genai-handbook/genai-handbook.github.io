
Here we'll be looking at a handful of topics related to improving or modifying the performance of language models without additional training, as well as techniques for measuring and understanding their performance.


Before diving into the individual chapters, I'd recommend these two high-level overviews, which touch on many of the topics we'll examine here:
- ["Building LLM applications for production"](https://huyenchip.com/2023/04/11/llm-engineering.html) by Chip Huyen
- ["What We Learned from a Year of Building with LLMs" Part 1](https://www.oreilly.com/radar/what-we-learned-from-a-year-of-building-with-llms-part-i/) and [Part 2](https://www.oreilly.com/radar/what-we-learned-from-a-year-of-building-with-llms-part-ii/) from O'Reilly (several authors)

These web courses also have a lot of relevant interactive materials:
- ["Large Language Model Course"](https://github.com/mlabonne/llm-course) from Maxime Labonne
- ["Generative AI for Beginners"](https://microsoft.github.io/generative-ai-for-beginners/) from Microsoft

<h2>Benchmarking</h2>

Beyond the standard numerical performance measures used during LLM training like cross-entropy loss and [perplexity](https://medium.com/@priyankads/perplexity-of-language-models-41160427ed72), the true performance of frontier LLMs is more commonly judged according to a range of benchmarks, or "evals". Common types of these are:

- Human-evaluated outputs (e.g. [LMSYS Chatbot Arena](https://chat.lmsys.org/))
- AI-evaluated outputs (as used in [RLAIF](https://argilla.io/blog/mantisnlp-rlhf-part-4/))
- Challenge question sets (e.g. those in HuggingFace's [LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard))

See the [slides](https://web.stanford.edu/class/cs224n/slides/cs224n-spr2024-lecture11-evaluation-yann.pdf) from Stanford's CS224n for an overview.
This [blog post](https://www.jasonwei.net/blog/evals) by Jason Wei and [this one](https://humanloop.com/blog/evaluating-llm-apps?utm_source=newsletter&utm_medium=sequence&utm_campaign=) by Peter Hayes do a nice job discussing the challenges and tradeoffs associated with designing good evaluations, and highlighting a number of the most prominent ones used today. The documentation for the open source framework [inspect-ai](https://ukgovernmentbeis.github.io/inspect_ai/) also features some useful discussion around designing benchmarks and reliable evaluation pipelines.

<h2>Sampling and Structured Outputs</h2>

While typical LLM inference samples tokens one at a time, there are number of parameters controlling the token distribution (temperature, top_p, top_k) which can be modified to control the variety of responses, as well as non-greedy decoding strategies that allow some degree of "lookahead". This [blog post](https://towardsdatascience.com/decoding-strategies-in-large-language-models-9733a8f70539) by Maxime Labonne does a nice job discussing several of them.

Sometimes we also want our outputs to follow a particular structure, particularly if we are using LLMs as a component of a larger system rather than as just a chat interface. Few-shot prompting works okay, but not all the time, particularly as output schemas become more complicated. For schema types like JSON, Pydantic and Outlines are popular tools for constraining the output structure from LLMs. Some useful resources:

- [Pydantic Concepts](https://docs.pydantic.dev/latest/concepts/models/)
- [Outlines for JSON](https://outlines-dev.github.io/outlines/reference/json/)
- [Outlines review](https://michaelwornow.net/2023/12/29/outlines-demo) by Michael Wornow 


<h2>Prompting Techniques</h2>

There are many prompting techniques, and many more prompt engineering guides out there, featuring methods for coaxing more desirable outputs from LLMs. Some of the classics:

- Few-Shot Examples
- Chain-of-Thought
- Retrieval-Augmented Generation (RAG)
- ReAct

This [blog post](https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/) by Lilian Weng discusses several of the most dominant approaches, [guide](https://www.promptingguide.ai/techniques) has decent coverage and examples for a wider range of the prominent techniques used today. Keyword-searching on Twitter/X or LinkedIn will give you plenty more. We'll also dig deeper into RAG and agent methods in later chapters.


<h2>Vector Databases and Reranking</h2>

RAG systems require the ability to quickly retrieve relevant documents from large corpuses. Relevancy is typically determined by similarity measures for semantic [embedding](#embeddings-and-topic-modeling) vectors of both queries and documents, such as cosine similarity or Euclidean distance. If we have just a handful of documents, this can be computed between a query and each document, but this quickly becomes intractable when the number of documents grows large. This is the problem addressed by vector databases, which allow retrieval of the _approximate_ top-K matches (significantly faster than checking all pairs) by maintaining high-dimensional indices over vectors which efficiently encode their geometric structure.  These [docs](https://www.pinecone.io/learn/series/faiss/) from Pinecone do a nice job walking through a few different methods for vector storage, like Locality-Sensitive Hashing and Hierarchical Navigable Small Worlds, which can be implemented with the popular FAISS open-source library. This [talk](https://www.youtube.com/watch?v=W-i8bcxkXok) by Alexander Chatzizacharias gives a nice overview as well.


Another related application of vector retrieval is the "reranking" problem, wherein a model can optimize for other metrics beyond query similarity, such as diversity within retrieved results. See these [docs](https://www.pinecone.io/learn/series/rag/rerankers/) from Pinecone for an overview. We'll see more about how retrieved results are actually used by LLMs in the next chapter.

<h2>Retrieval-Augmented Generation</h2>

One of the most buzzed-about uses of LLMs over the past year, retrieval-augmented generation (RAG) is how you can "chat with a PDF" (if larger than a model's context) and how applications like Perplexity and Arc Search can "ground" their outputs using web sources. This retrieval is generally powered by embedding each document for storage in a vector database + querying with the relevant section of a user's input.

Some overviews:
- ["Deconstructing RAG"](https://blog.langchain.dev/deconstructing-rag/) from Langchain
- ["Building RAG with Open-Source and Custom AI Models"](https://www.bentoml.com/blog/building-rag-with-open-source-and-custom-ai-models) from Chaoyu Yang

The [Advanced RAG](https://learn.deeplearning.ai/courses/building-evaluating-advanced-rag/lesson/1/introduction) video course from DeepLearning.AI may also be useful for exploring variants on the standard setup. 

<h2>Tool Use and "Agents"</h2>

The other big application buzzwords you've most likely encountered in some form are "tool use" and "agents", or "agentic programming". This typically starts with the ReAct framework we saw in the prompting section, then gets extended to elicit increasingly complex behaviors like software engineering (see the much-buzzed-about "Devin" system from Cognition, and several related open-source efforts like Devon/OpenDevin/SWE-Agent). There are many programming frameworks for building agent systems on top of LLMs, with Langchain and LlamaIndex being two of the most popular. There also seems to be some value in having LLMs rewrite their own prompts + evaluate their own partial outputs; this observation is at the heart of the DSPy framework (for "compiling" a program's prompts, against a reference set of instructions or desired outputs) which has recently been seeing a lot of attention.

Resources:
- ["LLM Powered Autonomous Agents" (post)](https://lilianweng.github.io/posts/2023-06-23-agent/ ) from Lilian Weng
- ["A Guide to LLM Abstractions" (post)](https://www.twosigma.com/articles/a-guide-to-large-language-model-abstractions/) from Two Sigma
- ["DSPy Explained! (video)"](https://www.youtube.com/watch?v=41EfOY0Ldkc) by Connor Shorten

Also relevant are more narrowly-tailored (but perhaps more practical) applications related to databases --- see these two [blog](https://neo4j.com/developer-blog/knowledge-graphs-llms-multi-hop-question-answering/) [posts](https://neo4j.com/blog/unifying-llm-knowledge-graph/) from Neo4J for discussion on applying LLMs to analyzing or constructing knowledge graphs, or this [blog post](https://numbersstation.ai/data-wrangling-with-fms-2/) from Numbers Station about applying LLMs to data wrangling tasks like entity matching.


<h2>LLMs for Synthetic Data</h2>

An increasing number of applications are making use of LLM-generated data for training or evaluations, including distillation, dataset augmentation, AI-assisted evaluation and labeling, self-critique, and more. This [post](https://www.promptingguide.ai/applications/synthetic_rag) demonstrates how to construct such a synthetic dataset (in a RAG context), and this [post](https://argilla.io/blog/mantisnlp-rlhf-part-4/) from Argilla gives an overview of RLAIF, which is often a popular alternative to RLHF, given the challenges associated with gathering pairwise human preference data. AI-assisted feedback is also a central component of the "Constitutional AI" alignment method pioneered by Anthropic (see their [blog](https://www.anthropic.com/news/claudes-constitution) for an overview). 

<h2>Representation Engineering</h2>

Representation Engineering is a new and promising technique for fine-grained steering of language model outputs via "control vectors". Somewhat similar to LoRA adapters, it has the effect of adding low-rank biases to the weights of a network which can elicit particular response styles (e.g. "humorous", "verbose", "creative", "honest"), yet is much more computationally efficient and can be implemented without any training required. Instead, the method simply looks at differences in activations for pairs of inputs which vary along the axis of interest (e.g. honesty), which can be generated synthetically, and then performs dimensionality reduction.


See this short [blog post](https://www.safe.ai/blog/representation-engineering-a-new-way-of-understanding-models) from Center for AI Safety (who pioneered the method) for a brief overview, and this [post](https://vgel.me/posts/representation-engineering/) from Theia Vogel for a technical deep-dive with code examples. Theia also walks through the method in this [podcast episode](https://www.youtube.com/watch?v=PkA4DskA-6M). 


<h2>Mechanistic Interpretability</h2>

Mechanistic Interpretability (MI) is the dominant paradigm for understanding the inner workings of LLMs by identifying sparse representations of "features" or "circuits" encoded in model weights. Beyond enabling potential modification or explanation of LLM outputs, MI is often viewed as an important step towards potentially "aligning" increasingly powerful systems. Most of the references here will come from [Neel Nanda](https://www.neelnanda.io), a leading researcher in the field who's created a large number of useful educational resources about MI across a range of formats:

- ["A Comprehensive Mechanistic Interpretability Explainer & Glossary"](https://www.neelnanda.io/mechanistic-interpretability/glossary) 
- ["An Extremely Opinionated Annotated List of My Favourite Mechanistic Interpretability Papers"](https://www.neelnanda.io/mechanistic-interpretability/favourite-papers) 
- ["Mechanistic Interpretability Quickstart Guide"](https://www.lesswrong.com/posts/jLAvJt8wuSFySN975/mechanistic-interpretability-quickstart-guide) (Neel Nanda on LessWrong)
- ["How useful is mechanistic interpretability?"](https://www.lesswrong.com/posts/tEPHGZAb63dfq2v8n/how-useful-is-mechanistic-interpretability) (Neel and others, discussion on LessWrong)
- ["200 Concrete Problems In Interpretability"](https://docs.google.com/spreadsheets/d/1oOdrQ80jDK-aGn-EVdDt3dg65GhmzrvBWzJ6MUZB8n4/edit#gid=0) (Annotated spreadsheet of open problems from Neel)

Additionally, the articles ["Toy Models of Superposition"](https://transformer-circuits.pub/2022/toy_model/index.html) and ["Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet"](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html) from Anthropic are on the longer side, but feature a number of great visualizations and demonstrations of these concepts.

<h2>Linear Representation Hypotheses</h2> 

An emerging theme from several lines of interpretability research has been the observation that internal representations of features in Transformers are often "linear" in high-dimensional space (a la Word2Vec). On one hand this may appear initially surprising, but it's also essentially an implicit assumption for techniques like similarity-based retrieval, merging, and the key-value similarity scores used by attention. See this [blog post](https://www.beren.io/2023-04-04-DL-models-are-secretly-linear/) by Beren Millidge, this [talk](https://www.youtube.com/watch?v=ko1xVcyDt8w) from Kiho Park, and perhaps at least skim the paper ["Language Models Represent Space and Time"](https://arxiv.org/pdf/2310.02207) for its figures. 
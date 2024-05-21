
Here, more than any other section of the handbook, I’ll defer mostly wholesale to existing sequences of instructional resources. This material has been covered quite well by many people in a variety of formats, and there's no need to reinvent the wheel. 

There are a couple different routes you can take from the basics of neural networks towards Transformers (the dominant architecture for most frontier LLMs in 2024). Once we cover the basics, I'll mostly focus on "deep sequence learning" methods like RNNs. Many deep learning books and courses will more heavily emphasize convolutional neural nets (CNNs), which are quite important for image-related applications and historically were one of the first areas where "scaling" was particularly successful, but technically they're fairly disconnected from Transformers. They’ll make an appearance when we discuss [state-space models](#structured-state-space-models) and are definitely important for vision applications, but you'll mostly be okay skipping them for now. However, if you're in a rush and just want to get to the new stuff, you could consider diving right into decoder-only Transformers once you're comfortable with feed-forward neural nets --- this the approach taken by the excellent ["Let's build GPT"](https://www.youtube.com/watch?v=kCc8FmEb1nY) video from Andrej Karpathy, casting them as an extension of neural n-gram models for next-token prediction. That's probably your single best bet for speedrunning Transformers in under 2 hours. But if you've got a little more time, understanding the history of RNNs, LSTMs, and encoder-decoder Transformers is certainly worthwhile.

This section is mostly composed of signposts to content from the following sources (along with some blog posts):

- The ["Dive Into Deep Learning" (d2l.ai)](http://d2l.ai) interactive textbook (nice graphics, in-line code, some theory)
- 3Blue1Brown's ["Neural networks"](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) video series (lots of animations)
- Andrej Karpathy's ["Zero to Hero"](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) video series (live coding + great intuitions)
- ["StatQuest with Josh Starmer"](https://www.youtube.com/@statquest) videos
- The Goodfellow et al. ["Deep Learning"](https://www.deeplearningbook.org/) textbook (theory-focused, no Transformers)

If your focus is on applications, you might find the interactive ["Machine Learning with PyTorch and Scikit-Learn"](https://github.com/rasbt/machine-learning-book/tree/main) book useful, but I'm not as familiar with it personally. 

For these topics, you can also probably get away with asking conceptual questions to your preferred LLM chat interface. This likely won't be true for later sections --- some of those topics were introduced after the knowledge cutoff dates for many current LLMs, and there's also just a lot less text on the internet about them, so you end up with more "hallucinations".

<h2>Statistical Prediction with Neural Networks</h2>

I'm not actually sure where I first learned about neural nets --- they're pervasive enough in technical discussions and general online media that I'd assume you've picked up a good bit through osmosis even if you haven't studied them formally. Nonetheless, there are many worthwhile explainers out there, and I'll highlight some of my favorites.

- The first 4 videos in 3Blue1Brown's ["Neural networks"](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) series will take you from basic definitions up through the mechanics of backpropagation.  
- This [blog post](https://karpathy.github.io/neuralnets/) from Andrej Karpathy (back when he was a PhD student) is a solid crash-course, well-accompanied by his [video](https://www.youtube.com/watch?v=VMj-3S1tku0) on building backprop from scratch.
- This [blog post](https://colah.github.io/posts/2015-08-Backprop/) from Chris Olah has a nice and concise walk-through of the math behind backprop for neural nets. 
- Chapters 3-5 of the [d2l.ai](http://d2l.ai) book are great as a "classic textbook" presentation of deep nets for regression + classification, with code examples and visualizations throughout.


<h2>Recurrent Neural Networks</h2>

RNNs are where we start adding "state" to our models (as we process increasingly long sequences), and there are some high-level similarities to hidden Markov models. This blog post from [Andrej Karpathy](https://karpathy.github.io/2015/05/21/rnn-effectiveness/) is a good starting point. [Chapter 9](https://d2l.ai/chapter_recurrent-neural-networks/index.html) of the [d2l.ai](http://d2l.ai) book is great for main ideas and code; check out [Chapter 10](https://www.deeplearningbook.org/contents/rnn.html) of "Deep Learning" if you want more theory.

For videos, [here](https://www.youtube.com/watch?v=AsNTP8Kwu80)'s a nice one from StatQuest.

<h2>LSTMs and GRUs</h2>

Long Short-Term Memory (LSTM) networks and Gated Recurrent Unit (GRU) networks build upon RNNs with more specialized mechanisms for state representation (with semantic inspirations like "memory", "forgetting", and "resetting"), which have been useful for improving performance in more challenging data domains (like language).   

[Chapter 10](https://d2l.ai/chapter_recurrent-modern/index.html) of [d2l.ai](http://d2l.ai) covers both of these quite well (up through 10.3). The ["Understanding LSTM Networks"](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) blog post from Chris Olah is also excellent. This [video](https://www.youtube.com/watch?v=8HyCNIVRbSU) from "The AI Hacker" gives solid high-level coverage of both; StatQuest also has a video on [LSTMs](https://www.youtube.com/watch?v=YCzL96nL7j0), but not GRUs. GRUs are essentially a simplified alternative to LSTMs with the same basic objective, and it's up to you if you want to cover them specifically. 

Neither LSTMs or GRUs are really prerequisites for Transformers, which are "stateless", but they're useful for understanding the general challenges neural sequence of neural sequence and contextualizing the Transformer design choices. They'll also help motivate some of the approaches towards addressing the "quadratic scaling problem" in [Section VII](#s7). 

<h2>Embeddings and Topic Modeling</h2>

Before digesting Transformers, it's worth first establishing a couple concepts which will be useful for reasoning about what's going on under the hood inside large language models. While deep learning has led to a large wave of progress in NLP, it's definitely a bit harder to reason about than some of the "old school" methods which deal with word frequencies and n-gram overlaps; however, even though these methods don't always scale to more complex tasks, they're useful mental models for the kinds of "features" that neural nets might be learning. For example, it's certainly worth knowing about Latent Dirichlet Allocation for topic modeling ([blog post](https://towardsdatascience.com/latent-dirichlet-allocation-lda-9d1cd064ffa2)) and [tf-idf](https://jaketae.github.io/study/tf-idf/) to get a feel for what numerical similarity or relevance scores can represent for language.

Thinking about words (or tokens) as high-dimensional "meaning" vectors is quite useful, and the Word2Vec embedding method illustrates this quite well --- you may have seen the classic "King - Man + Woman = Queen" example referenced before. ["The Illustrated Word2Vec"](https://jalammar.github.io/illustrated-word2vec/) from Jay Alammar is great for building up this intuition, and these [course notes](https://web.stanford.edu/class/cs224n/readings/cs224n_winter2023_lecture1_notes_draft.pdf) from Stanford's CS224n are excellent as well. Here's also a nice [video](https://www.youtube.com/watch?v=f7o8aDNxf7k) on Word2Vec from ritvikmath, and another fun one [video](https://www.youtube.com/watch?v=gQddtTdmG_8) on neural word embeddings from Computerphile.

Beyond being a useful intuition and an element of larger language models, standalone neural embedding models are also widely used today. Often these are encoder-only Transformers, trained via "contrastive loss" to construct high-quality vector representations of text inputs which are useful for retrieval tasks (like [RAG](#retrieval-augmented-generation)). See this [post+video](https://docs.cohere.com/docs/text-embeddings) from Cohere for a brief overview, and this [blog post](https://lilianweng.github.io/posts/2021-05-31-contrastive/) from Lilian Weng for more of a deep dive.

<h2>Encoders and Decoders</h2>

Up until now we've been pretty agnostic as to what the inputs to our networks are --- numbers, characters, words --- as long as it can be converted to a vector representation somehow. Recurrent models can be configured to both input and output either a single object (e.g. a vector) or an entire sequence. This observation enables the sequence-to-sequence encoder-decoder architecture, which rose to prominence for machine translation, and was the original design for the Transformer in the famed ["Attention is All You Need"](https://arxiv.org/abs/1706.03762) paper. Here, the goal is to take an input sequence (e.g. an English sentence), "encode" it into a vector object which captures its "meaning", and then "decode" that object into another sequence (e.g. a French sentence). [Chapter 10](https://d2l.ai/chapter_recurrent-modern/index.html) in [d2l.ai](http://d2l.ai) (10.6-10.8) covers this setup as well, which sets the stage for the encoder-decoder formulation of Transformers in [Chapter 11](https://d2l.ai/chapter_attention-mechanisms-and-transformers/index.html) (up through 11.7). For historical purposes you should certainly at least skim the original paper, though you might get a bit more out of the presentation of its contents via ["The Annotated Transformer"](https://nlp.seas.harvard.edu/annotated-transformer/), or perhaps ["The Illustrated Transformer"](https://jalammar.github.io/illustrated-transformer/) if you want more visualizations. These [notes](https://web.stanford.edu/class/cs224n/readings/cs224n-self-attention-transformers-2023_draft.pdf) from Stanford's CS224n are great as well.

There are videos on [encoder-decoder](https://www.youtube.com/watch?v=L8HKweZIOmg) architectures and [Attention](https://www.youtube.com/watch?v=PSs6nxngL6k) from StatQuest, a full walkthrough of the original Transformer by [The AI Hacker](https://www.youtube.com/watch?v=4Bdc55j80l8). 

However, note that these encoder-decoder Transformers differ from most modern LLMs, which are typically "decoder-only" -- if you're pressed for time, you may be okay jumping right to these models and skipping the history lesson.

<h2>Decoder-Only Transformers</h2>

There's a lot of moving pieces inside of Transformers --- multi-head attention, skip connections, positional encoding, etc. --- and it can be tough to appreciate it all the first time you see it. Building up intuitions for why some of these choices are made helps a lot, and here I'll recommend to pretty much anyone that you watch a video or two about them (even if you're normally a textbook learner), largely because there are a few videos which are really excellent:

- 3Blue1Brown's ["But what is a GPT?"](https://www.youtube.com/watch?v=wjZofJX0v4M) and ["Attention in transformers, explained visually"](Attention in transformers, visually explained | Chapter 6, Deep Learning) -- beautiful animations + discussions, supposedly a 3rd video is on the way
- Andrej Karpathy's ["Let's build GPT"](https://www.youtube.com/watch?v=kCc8FmEb1nY) video -- live coding and excellent explanations, really helped some things "click" for me

Here's a [blog post](https://cameronrwolfe.substack.com/p/decoder-only-transformers-the-workhorse) from Cameron Wolfe walking through the decoder-only architecture in a similar style to the Illustrated/Annotated Transformer posts. There's also a nice section in d2l.ai ([11.9](https://d2l.ai/chapter_attention-mechanisms-and-transformers/large-pretraining-transformers.html)) covering the relationships between encoder-only, encoder-decoder, and decoder-only Transformers.

So far, everything we've looked has been focused on text and sequence prediction with language models, but many other "generative AI" techniques require learning distributions with less of a sequential structure (e.g. images). Here we'll examine a number of non-Transformer architectures for generative modeling, starting from simple mixture models and culminating with diffusion.


<h2>Distribution Modeling</h2>

Recalling our first glimpse of language models as simple bigram distributions, the most basic thing you can do in distributional modeling is just count co-occurrence probabilities in your dataset and repeat them as ground truth. This idea can be extended to conditional sampling or classification as  "Naive Bayes" ([blog post](https://mitesh1612.github.io/blog/2020/08/30/naive-bayes) [video](https://www.youtube.com/watch?v=O2L2Uv9pdDA)), often one of the simplest algorithms covered in introductory machine learning courses.

The next generative model students are often taught is the Gaussian Mixture Model and its Expectation-Maximization algorithm; 
Gaussian Mixture Models + Expectation-Maximization algorithm. This [blog post](https://mpatacchiola.github.io/blog/2020/07/31/gaussian-mixture-models.html) and this [video](https://www.youtube.com/watch?v=DODphRRL79c) give decent overviews; the core idea here is assuming that data distributions can be approximated as a mixture of multivariate Gaussian distributions. GMMs can also be used for clustering if individual groups can be assumed to be approximately Gaussian.

While these methods aren't very effective at representing complex structures like images or language, related ideas will appear as components of some of the more advanced methods we'll see.

<h2>Variational Auto-Encoders</h2>

Auto-encoders and variational auto-encoders are widely used for learning compressed representations of data distributions, and can also be useful for "denoising" inputs, which will come into play when we discuss diffusion. Some nice resources:
- ["Autoencoders"](https://www.deeplearningbook.org/contents/autoencoders.html) chapter in the "Deep Learning" book 
- [blog post]([https://lilianweng.github.io/posts/2018-08-12-vae/]) from Lilian Weng
- [video](https://www.youtube.com/watch?v=9zKuYvjFFS8) from Arxiv Insights 
- [blog post](https://towardsdatascience.com/deep-generative-models-25ab2821afd3) from Prakash Pandey on both VAEs and GANs

<h2>Generative Adversarial Nets</h2>

The basic idea behind Generative Adversarial Networks (GANs) is to simulate a "game" between two neural nets --- the Generator wants to create samples which are indistinguishable from real data by the Discriminator, who wants to identify the generated samples, and both nets are trained continuously until an equilibrium (or desired sample quality) is reached.  Following from von Neumann's minimax theorem for zero-sum games, you basically get a "theorem" promising that GANs succeed at learning distributions, if you assume that gradient descent finds global minimizers and allow both networks to grow arbitrarily large. Granted, neither of these are literally true in practice, but GANs do tend to be quite effective (although they've fallen out of favor somewhat in recent years, partly due to the instabilities of simultaneous training).

Resources:
- ["Complete Guide to Generative Adversarial Networks"](https://blog.paperspace.com/complete-guide-to-gans/) from Paperspace
- ["Generative Adversarial Networks (GANs): End-to-End Introduction"](https://www.analyticsvidhya.com/blog/2021/10/an-end-to-end-introduction-to-generative-adversarial-networksgans/) by
- [Deep Learning, Ch. 20 - Generative Models](https://www.deeplearningbook.org/contents/generative_models.html) (theory-focused)

<h2>Conditional GANs</h2>
Conditional GANs are where we'll start going from vanilla "distribution learning" to something which more closely resembles interactive generative tools like DALL-E and Midjourney, incorporating text-image multimodality. A key idea is to learn "representations" (in the sense of text embeddings or autoencoders) which are more abstract and can be applied to either text or image inputs. For example, you could imagine training a vanilla GAN on (image, caption) pairs by embedding the text and concatenating it with an image, which could then learn this joint distribution over images and captions. Note that this implicitly involves learning conditional distributions if part of the input (image or caption) is fixed, and this can be extended to enable automatic captioning (given an image) or image generation (given a caption). There a number of variants on this setup with differing bells and whistles. The VQGAN+CLIP architecture is worth knowing about, as it was a major popular source of early "AI art" generated from input text.

Resources:
- ["Implementing Conditional Generative Adversarial Networks"](https://blog.paperspace.com/conditional-generative-adversarial-networks/ ) blog from Paperspace
- ["Conditional Generative Adversarial Network — How to Gain Control Over GAN Outputs"](https://towardsdatascience.com/cgan-conditional-generative-adversarial-network-how-to-gain-control-over-gan-outputs-b30620bd0cc8) by Saul Dobilas
- ["The Illustrated VQGAN"](https://ljvmiranda921.github.io/notebook/2021/08/08/clip-vqgan/) by LJ  Miranda
- ["Using Deep Learning to Generate Artwork with VQGAN-CLIP"](https://www.youtube.com/watch?v=Ih4qOakCZD4) talk from Paperspace


<h2>Normalizing Flows</h2>

The aim of normalizing flows is to learn a series of invertible transformations between Gaussian noise and an output distribution, avoiding the need for "simultaneous training" in GANs, and have been popular for generative modeling in a number of domains. Here I'll just recommend ["Flow-based Deep Generative Models"](https://lilianweng.github.io/posts/2018-10-13-flow-models/) from Lilian Weng as an overview --- I haven't personally gone very deep on normalizing flows, but they come up enough that they're probably worth being aware of. 

<h2>Diffusion Models</h2> 

One of the central ideas behind diffusion models (like StableDiffusion) is iterative guided application of denoising operations, refining random noise into something that increasingly resembles an image. Diffusion originates from the worlds of stochastic differential equations and statistical physics --- relating to the "Schrodinger bridge" problem and optimal transport for probability distributions --- and a fair amount of math is basically unavoidable if you want to understand the whole picture. For a relatively soft introduction, see ["A friendly Introduction to Denoising Diffusion Probabilistic Models"](https://medium.com/@gitau_am/a-friendly-introduction-to-denoising-diffusion-probabilistic-models-cc76b8abef25) by Antony Gitau. If you're up for some more math, check out ["What are Diffusion Models?"](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) for more of a deep dive. If you're more interested in code and pictures (but still some math), see ["The Annotated Diffusion Model"](https://huggingface.co/blog/annotated-diffusion) from Hugging Face, as well as this Hugging Face [blog post](https://huggingface.co/blog/lora) on LoRA finetuning for diffusion models.


Note: This is the set of topics with which I'm the least familiar, but wanted to include for completeness. I'll be lighter on commentary and recommendations here, and will return to add more when I think I have a tighter story to tell. The post ["Multimodality and Large Multimodal Models (LMMs)"](https://huyenchip.com/2023/10/10/multimodal.html) by Chip Huyen is a nice broad overview (or ["How Multimodal LLMs Work"](https://www.determined.ai/blog/multimodal-llms) by Kevin Musgrave for a more concise one).     

<h2>Tokenization Beyond Text</h2>

The idea of tokenization isn't only relevant to text; audio, images, and video can also be "tokenized" for use in Transformer-style archictectures, and there a range of tradeoffs to consider between tokenization and other methods like convolution. The next two sections will look more into visual inputs; this [blog post](https://www.assemblyai.com/blog/recent-developments-in-generative-ai-for-audio/) from AssemblyAI touches on a number of relevant topics for audio tokenization and representation in sequence models, for applications like audio generation, text-to-speech, and speech-to-text.

<h2>VQ-VAE</h2>

The VQ-VAE architecture has become quite popular for image generation in recent years, and underlies at least the earlier versions of DALL-E. 

Resources:
- ["Understanding VQ-VAE (DALL-E Explained Pt. 1)"](https://mlberkeley.substack.com/p/vq-vae) from the Machine Learning @ Berkeley blog
- ["How is it so good ? (DALL-E Explained Pt. 2)"](https://mlberkeley.substack.com/p/dalle2)
- ["Understanding Vector Quantized Variational Autoencoders (VQ-VAE)"](https://shashank7-iitd.medium.com/understanding-vector-quantized-variational-autoencoders-vq-vae-323d710a888a) by Shashank Yadav


<h2>Vision Transformers</h2>

Vision Transformers extend the Transformer architecture to domains like image and video, and have become popular for applications like self-driving cars as well as for multimodal LLMs. There's a nice [section](https://d2l.ai/chapter_attention-mechanisms-and-transformers/vision-transformer.html) in the d2l.ai book about how they work.

["Generalized Visual Language Models"](https://lilianweng.github.io/posts/2022-06-09-vlm/) by Lilian Weng discusses a range of different approaches and for training multimodal Transformer-style models.

The post ["Guide to  Vision Language Models"](https://encord.com/blog/vision-language-models-guide/) from Encord's blog overviews several architectures for mixing text and vision.

If you're interested in practical large-scale training advice for Vision Transformers, the MM1 [paper](https://arxiv.org/abs/2403.09611) from Apple examines several architecture and data tradeoffs with experimental evidence.

["Multimodal Neurons in Artificial Neural Networks"](https://distill.pub/2021/multimodal-neurons/) from Distill.pub has some very fun visualizations of concept representations in multimodal networks.

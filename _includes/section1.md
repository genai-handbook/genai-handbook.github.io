
Our focus in this section will be on quickly overviewing classical topics in statistical prediction and reinforcement learning, which we'll make direct reference to in later sections, as well as highlighting some topics that I think are very useful as conceptual models for understanding LLMs, yet which are often omitted from deep learning crash courses -- notably time-series analysis, regret minimization, and Markov models.

<h2>Statistical Prediction and Supervised Learning</h2>

Before getting to deep learning and large language models, it’ll be useful to have a solid grasp on some foundational concepts in probability theory and machine learning. In particular, it helps to understand: 

- Random variables, expectations, and variance
- Supervised vs. unsupervised learning
- Regression vs. classification
- Linear models and regularization
- Empirical risk minimization
- Hypothesis classes and bias-variance tradeoffs

For general probability theory, having a solid understanding of how the Central Limit Theorem works is perhaps a reasonable litmus test for how much you'll need to know about random variables before tackling some of the later topics we'll cover.  This beautifully-animated 3Blue1Brown [video](https://www.youtube.com/watch?v=zeJD6dqJ5lo) is a great starting point, and there are a couple other good probability videos to check out on the channel if you'd like. This set of [course notes](https://blogs.ubc.ca/math105/discrete-random-variables/) from UBC covers the basics of random variables.
If you're into blackboard lectures, I'm a big fan of many of Ryan O'Donnell's CMU courses on YouTube, and this [video](https://www.youtube.com/watch?v=r9S2fMQiP2E&list=PLm3J0oaFux3ZYpFLwwrlv_EHH9wtH6pnX&index=13) on random variables and the Central Limit Theorem (from the excellent "CS Theory Toolkit" course) is a nice overview.

For understanding linear models and other key machine learning principles, the first two chapters of Hastie’s [Elements of Statistical Learning](https://hastie.su.domains/ElemStatLearn/printings/ESLII_print12_toc.pdf) (”Introduction” and “Overview of Supervised Learning”) should be enough to get started.

Once you're familiar with the basics, this [blog post](https://ryxcommar.com/2019/09/06/some-things-you-maybe-didnt-know-about-linear-regression/) by anonymous Twitter/X user [@ryxcommar](https://twitter.com/ryxcommar) does a nice job discussing some common pitfalls and misconceptions related to linear regression. [StatQuest](https://www.youtube.com/@statquest/playlists) on YouTube has a number of videos that might also be helpful.

Introductions to machine learning tend to emphasize linear models, and for good reason. Many phenomena in the real world are modeled quite well by linear equations — the average temperature over past 7 days is likely a solid guess for the temperature tomorrow, barring any other information about weather pattern forecasts. Linear systems and models are a lot easier to study, interpret, and optimize than their nonlinear counterparts. For more complex and high-dimensional problems with potential nonlinear dependencies between features, it’s often useful to ask: 

- What’s a linear model for the problem?
- Why does the linear model fail?
- What’s the best way to add nonlinearity, given the semantic structure of the problem?

In particular, this framing will be helpful for motivating some of the model architectures we’ll look at later (e.g. LSTMs and Transformers).

<h2>Time-Series Analysis</h2>

How much do you need to know about time-series analysis in order to understand the mechanics of more complex generative AI methods?

**Short answer:** just a tiny bit for LLMs, a good bit more for diffusion.

For modern Transformer-based LLMs, it'll be useful to know:

- The basic setup for sequential prediction problems
- The notion of an autoregressive model

There's not really a coherent way to "visualize" the full mechanics of a multi-billion-parameter model in your head, but much simpler autoregressive models (like ARIMA) can serve as a nice mental model to extrapolate from. 

When we get to neural [state-space models](#structured-state-space-models), a working knowledge of linear time-invariant systems and control theory (which have many connections to classical time-series analysis) will be helpful for intuition, but [diffusion](#diffusion-models) is really where it's most essential to dive deeper into into stochastic differential equations to get the full picture. But we can table that for now.

This blog post ([Forecasting with Stochastic Models](https://towardsdatascience.com/forecasting-with-stochastic-models-abf2e85c9679)) from Towards Data Science is concise and introduces the basic concepts along with some standard autoregressive models and code examples.

This set of [notes](https://sites.ualberta.ca/~kashlak/data/stat479.pdf) from UAlberta’s “Time Series Analysis” course is nice if you want to go a bit deeper on the math.

<h2>Online Learning and Regret Minimization</h2>

It’s debatable how important it is to have a strong grasp on regret minimization, but I think a basic familiarity is useful. The basic setting here is similar to supervised learning, but:

- Points arrive one-at-a-time in an arbitrary order
- We want low average error across this sequence

If you squint and tilt your head, most of the algorithms designed for these problems look basically like gradient descent, often with delicate choices of regularizers and learning rates require for the math to work out. But there’s a lot of satisfying math here. I have a soft spot for it, as it relates to a lot of the research I worked on during my PhD. I think it’s conceptually fascinating. Like the previous section on time-series analysis, online learning is technically “sequential prediction” but you don’t really need it to understand LLMs.

The most direct connection to it that we’ll consider is when we look at [GANs](#generative-adversarial-nets) in Section VIII. There are many deep connections between regret minimization and equilibria in games, and GANs work basically by having two neural networks play a game against each other. Practical gradient-based optimization algorithms like Adam have their roots in this field as well, following the introduction of the AdaGrad algorithm, which was first analyzed for online and adversarial settings. In terms of other insights, one takeaway I find useful is the following: If you’re doing gradient-based optimization with a sensible learning rate schedule, then the order in which you process data points doesn’t actually matter much. Gradient descent can handle it. 

I’d encourage you to at least skim Chapter 1 of [“Introduction to Online Convex Optimization”](https://arxiv.org/pdf/1909.05207.pdf) by Elad Hazan to get a feel for the goal of regret minimization. I’ve spent a lot of time with this book and I think it’s excellent.

<h2>Reinforcement Learning</h2>

Reinforcement Learning (RL) will come up most directly when we look at finetuning methods in [Section IV](#finetuning-methods-for-llms), and may also be a useful mental model for thinking about "agent" [applications](#tool-use-and-agents) and some of the "control theory" notions which come up for [state-space models](#structured-state-space-models). Like a lot of the topics discussed in this document, you can go quite deep down many different RL-related threads if you'd like; as it relates to language modeling and alignment, it'll be most important to be comfortable with the basic problem setup for Markov decision processes, notion of policies and trajectories, and high-level understanding of standard iterative + gradient-based optimization methods for RL.

This [blog post](https://lilianweng.github.io/posts/2018-02-19-rl-overview/) from Lilian Weng is a great starting point, and is quite dense with important RL ideas despite its relative conciseness. It also touches on connections to AlphaGo and gameplay, which you might find interesting as well.

The textbook ["Reinforcement Learning: An Introduction"](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf) by Sutton and Barto is generally considered the classic reference text for the area, at least for "non-deep" methods. This was my primary guide when I was first learning about RL, and it gives a more in-depth exploration of many of the topics touched on in Lilian's blog post.

If you want to jump ahead to some more neural-flavored content, Andrej Karpathy has a nice [blog post](https://karpathy.github.io/2016/05/31/rl/) on deep RL; this [manuscript](https://arxiv.org/pdf/1810.06339) by Yuxi Li and this [textbook](https://arxiv.org/pdf/2201.02135) by Aske Plaat may be useful for further deep dives. 

If you like 3Blue1Brown-style animated videos, the series ["Reinforcement Learning By the Book"](https://www.youtube.com/playlist?list=PLzvYlJMoZ02Dxtwe-MmH4nOB5jYlMGBjr) is a great alternative option, and conveys a lot of content from Sutton and Barto, along with some deep RL, using engaging visualizations.

<h2>Markov Models</h2>

Running a fixed policy in a Markov decision process yields a Markov chain; processes resembling this kind of setup are fairly abundant, and many branches of machine learning involve modeling systems under Markovian assumptions (i.e. lack of path-dependence, given the current state). This [blog post](https://thagomizer.com/blog/2017/11/07/markov-models.html) from Aja Hammerly makes a nice case for thinking about language models via Markov processes, and this [post](https://ericmjl.github.io/essays-on-data-science/machine-learning/markov-models/) from "Essays on Data Science" has examples and code building up towards auto-regressive Hidden Markov Models, which will start to vaguely resemble some of the neural network architectures we'll look at later on. 

This [blog post](https://www.tweag.io/blog/2019-10-25-mcmc-intro1/) from Simeon Carstens gives a nice coverage of Markov chain Monte Carlo methods, which are powerful and widely-used techniques for sampling from implicitly-represented distributions, and are helpful for thinking about probabilistic topics ranging from stochastic gradient descent to diffusion.

Markov models are also at the heart of many Bayesian methods. See this [tutorial](https://mlg.eng.cam.ac.uk/zoubin/papers/ijprai.pdf) from Zoubin Ghahramani for a nice overview, the textbook ["Pattern Recognition and Machine Learning"](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf) for Bayesian angles on many machine learning topics (as well as a more-involved HMM presentation), and this [chapter](https://www.deeplearningbook.org/contents/graphical_models.html) of the Goodfellow et al. "Deep Learning" textbook for some connections to deep learning.
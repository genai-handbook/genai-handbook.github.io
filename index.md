---
layout: default
title: GenAI Handbook
---

William Brown \
[@willccbb](https://x.com/willccbb) | [willcb.com](https://willcb.com) \
[v0.1](https://github.com/genai-handbook/genai-handbook.github.io) (June 5, 2024)

{% include intro.md %}

<div class="section" id="section1">
  <div class="section-header no-numbering" id="section1-header">
    <div class="section-header-container">
      <div class="section-header-content">
        <h1 id="s1">Section I: Foundations of Sequential Prediction</h1>
        <div class="section-summary"><b>Goal:</b> Recap machine learning basics + survey (non-DL) methods for tasks under the umbrella of “sequential prediction”.</div>
      </div>
    </div>
  </div>
  <div class="section-content" markdown="1">
    {% include section1.md %}
  </div>
</div>

<div class="section" id="section2">
  <div class="section-header no-numbering" id="section2-header">
    <div class="section-header-container">
      <div class="section-header-content">
        <h1 id="s2">Section II: Neural Sequential Prediction</h1>
        <div class="section-summary"><b>Goal:</b> Survey deep learning methods + applications to sequential and language modeling, up to basic Transformers.</div>
      </div>
    </div>
  </div>
  <div class="section-content" markdown="1">
    {% include section2.md %}
  </div>
</div>

<div class="section" id="section3">
  <div class="section-header no-numbering" id="section3-header">
    <div class="section-header-container">
      <div class="section-header-content">
        <h1 id="s3">Section III: Foundations for Modern Language Modeling</h1>
        <div class="section-summary"><b>Goal:</b> Survey central topics related to training LLMs, with an emphasis on conceptual primitives.</div>
      </div>
    </div>
  </div>
  <div class="section-content" markdown="1">
    {% include section3.md %}
  </div>
</div>

<div class="section" id="section4">
  <div class="section-header no-numbering" id="section4-header">
    <div class="section-header-container">
      <div class="section-header-content">
        <h1 id="s4">Section IV: Finetuning Methods for LLMs</h1>
        <div class="section-summary"><b>Goal:</b> Survey techniques used for improving and "aligning" the quality of LLM outputs after pretraining.</div>
      </div>
    </div>
  </div>
  <div class="section-content" markdown="1">
    {% include section4.md %}
  </div>
</div>

<div class="section" id="section5">
  <div class="section-header no-numbering" id="section5-header">
    <div class="section-header-container">
      <div class="section-header-content">
        <h1 id="s5">Section V: LLM Evaluations and Applications</h1>
        <div class="section-summary"><b>Goal:</b> Survey how LLMs are used and evaluated in practice, beyond just "chatbots".</div>
      </div>
    </div>
  </div>
  <div class="section-content" markdown="1">
    {% include section5.md %}
  </div>
</div>

<div class="section" id="section6">
  <div class="section-header no-numbering" id="section6-header">
    <div class="section-header-container">
      <div class="section-header-content">
        <h1 id="s6">Section VI: Performance Optimizations for Efficient Inference</h1>
        <div class="section-summary"><b>Goal:</b> Survey architecture choices and lower-level techniques for improving resource utilization (time, compute, memory).</div>
      </div>
    </div>
  </div>
  <div class="section-content" markdown="1">
    {% include section6.md %}
  </div>
</div>

<div class="section" id="section7">
  <div class="section-header no-numbering" id="section7-header">
    <div class="section-header-container">
      <div class="section-header-content">
        <h1 id="s7">Section VII: Sub-Quadratic Context Scaling</h1>
        <div class="section-summary"><b>Goal:</b> Survey approaches for avoiding the "quadratic scaling problem" faced by self-attention in Transformers.</div>
      </div>
    </div>
  </div>
  <div class="section-content" markdown="1">
    {% include section7.md %}
  </div>
</div>

<div class="section" id="section8">
  <div class="section-header no-numbering" id="section8-header">
    <div class="section-header-container">
      <div class="section-header-content">
        <h1 id="s8">Section VIII: Generative Modeling Beyond Sequences</h1>
        <div class="section-summary"><b>Goal:</b> Survey topics building towards generation of non-sequential content like images, from GANs to diffusion models.</div>
      </div>
    </div>
  </div>
  <div class="section-content" markdown="1">
    {% include section8.md %}
  </div>
</div>

<div class="section" id="section9">
  <div class="section-header no-numbering" id="section9-header">
    <div class="section-header-container">
      <div class="section-header-content">
        <h1 id="s9">Section IX: Multimodal Models</h1>
        <div class="section-summary"><b>Goal:</b> Survey how models can use multiple modalities of input and output (text, audio, images) simultaneously.</div>
      </div>
    </div>
  </div>
  <div class="section-content" markdown="1">
    {% include section9.md %}
  </div>
</div>

{% include credits.md %}
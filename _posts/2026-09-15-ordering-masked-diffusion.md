---
layout: post
title: "Ordering in masked discrete diffusion"
description: "Once a token is unmasked it is frozen. So in which order should we unmask an action chunk?"
date: 2026-09-15 10:00:00 +0200
tags: [short-post, diffusion, action-chunks]
---

Recently I started looking into masked discrete diffusion after reading this paper: [Discrete RTC](https://outsider86.github.io/DiscreteRTCSite/). 

Its simplicity is appealing, but it comes with a catch. Unlike continuous diffusion/flow, where all action dimensions are progressively denoised (from low to high frequency), here it's all or nothing: once you unmask (denoise) a token, it can't be changed afterwards, unless you use some heuristic methods, which I will cover later in this post.

In practice this is problematic, as for real-time inference you won't have as many unmasking steps as action tokens in your chunk. So you will end up in situations where you sample several marginals independently, which, in general, doesn't match the joint distribution. 

A natural question that arises is: **in what order should we unmask?**

In the small figures below, each grid is an action chunk (rows: action dimensions, columns: timesteps). Gray tokens are still masked, blue ones are already unmasked, and orange ones are unmasked at the current step.

## Domain-knowledge heuristics

{% include figure.html src="/assets/img/masked-diffusion-ordering/heuristics.png" alt="Tokens far apart in the chunk are unmasked together" width="62%" caption="Unmask together tokens that are far apart, hoping they are nearly independent." %}

For image generation, for instance, you might denoise tokens that are far apart together, assuming that they are independent, which should be relatively fine most of the time.

But for actions it's less clear and might be heavily context dependent.

## Temporal order

{% include figure.html src="/assets/img/masked-diffusion-ordering/temporal.png" alt="Action chunk unmasked one timestep at a time" width="62%" caption="Unmask one timestep at a time." %}

We naturally structure action chunks as a product of time and action dimensions. Usually, we try to make action dimensions independent. In practice, this is rarely achievable, but it can be a good enough approximation. When this approximation is good enough, it can be tempting to sample temporally.

But then, wouldn't it boil down to simple one-step prediction with past action context, without diffusion?

I think, essentially, yes. That said, if you keep masking uniformly during training, the model might learn something different from simple prediction.

But most of all, I think it would be a bad idea, as you would lose much of what makes predicting whole chunks work.
A recent study of action chunking [[1]](#ref-chunking) suggests that much of its benefit comes from predicting actions from an observation that is more than one step in the past, which forces the policy not to be short-sighted, and from implicitly ensembling all these different temporal relationships.

To be more precise, if you sample everything temporally step by step, then because past actions leak information, your network will tend to rely mainly on them for its decisions, as it's the easiest thing to do and also, most of the time, the best thing to do.
But this leads to bad policies that repeat themselves and disregard the context. This problem is called *causal confusion* [[2]](#ref-causal) and, in my experience, it tends to come up over and over.

## WFC-style

{% include figure.html src="/assets/img/masked-diffusion-ordering/wfc.png" alt="The most confident masked tokens are unmasked first" width="62%" caption="Unmask the tokens the model is most confident about (numbers: highest predicted probability)." %}

Another classical approach worth mentioning is to unmask the most certain tokens first, which is much like what Wave Function Collapse [[3]](#ref-wfc) does. But this is detrimental to diversity and biases samples toward the mode.

## Learned

{% include figure.html src="/assets/img/masked-diffusion-ordering/learned.png" alt="Predicted influence of unmasking one token on the other tokens" width="62%" caption="A network predicts how much unmasking a token would change the others; weakly coupled tokens can be unmasked together." %}

Another approach could be to have a network estimate how much unmasking each token would affect the distribution of the others. It's very similar to an idea I had when working on AlphaZero ([more on that in another post]({% post_url 2023-04-15-grouped-mcts %})), but here it doesn't come for free: since you don't run MCTS, it would be computationally expensive.

## Give up and correct

{% include figure.html src="/assets/img/masked-diffusion-ordering/correct.png" alt="An unlikely unmasked token is remasked" caption="Already unmasked tokens whose predicted probability falls below a threshold are remasked." %}

Out of despair, you might end up thinking: "fine, it will make errors, but maybe we can detect them and remask them". For instance, one approach could be: if the predicted probability of an already unmasked token is below $$x$$, then remask it. Typically, $$x$$ should increase during denoising, from very small to moderate, to give rare actions a chance.

But this is actually ill-behaved, as the network's logits for already unmasked tokens were never trained. In order to train them, you would need to add some non-masked (e.g., uniform) diffusion back.

---

So in the end, it seems that unless you can craft a good enough ordering heuristic, the best option remains random ordering, possibly with correction. And if you still aren't satisfied, discrete diffusion without masking might be the solution.

## References

<ol class="references">
  <li id="ref-chunking">F. Lazzati, K. Stachowicz, W. Chen, A. M. Metelli, A. Wagenmaker, S. Levine. <a href="https://action-chunking.github.io/">Why Does Action Chunking Improve Behavioral Cloning Performance in Robotic Control?</a> arXiv:2608.02547, 2026.</li>
  <li id="ref-causal">P. de Haan, D. Jayaraman, S. Levine. <a href="https://arxiv.org/abs/1905.11979">Causal Confusion in Imitation Learning</a>. NeurIPS 2019.</li>
  <li id="ref-wfc">M. Gumin. <a href="https://github.com/mxgmn/WaveFunctionCollapse">WaveFunctionCollapse</a>. 2016.</li>
</ol>

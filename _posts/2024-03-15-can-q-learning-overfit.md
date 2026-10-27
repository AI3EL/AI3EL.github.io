---
layout: post
title: "Can Q-learning overfit?"
description: "A classic deep-learning sanity check that silently fails for offline Q-learning, and why."
date: 2024-03-15 10:00:00 +0100
image: /assets/img/q-learning-overfit/max-noise.png
tags: [short-post, q-learning]
---

A common sanity check when implementing a deep learning algorithm is to try to overfit a small dataset.
This helps catch simple bugs in data loading, augmentations, or tensor computations: a missing minus sign, an off-by-one error, or any other nasty silent bug.
But what happens when you try the same thing with Q-learning?

You might find that, even with a perfectly correct implementation, the network refuses to overfit.
Can you see why?

---

The culprit is a fundamental challenge in offline Q-learning: **bootstrapping on noise**.
Recall that Q-learning involves a max over values that are randomly initialized and, for the most part, never touched during training:

$$
\mathcal{L}(\theta) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}} \Big[ \big( Q_\theta(s,a) - r - \gamma \max_{a'} Q_{\bar\theta}(s',a') \big)^2 \Big]
$$

The $$\max_{a'}$$ ranges over *all* actions, including the ones that never appear in the dataset. Even if you only monitor the Q-values of state-action pairs in your dataset, you will still have an issue, as they will have been contaminated by noise through the max.

{% include figure.html src="/assets/img/q-learning-overfit/max-noise.png" alt="The max in the Q-learning target selects an untrained, noisy Q-value" caption="The pair $$(s, a)$$ is in the dataset, but its regression target is built from $$Q(s', \cdot)$$ over *all* actions. Only $$a'$$ (blue) ever receives a gradient; the other values are whatever the network outputs at initialization, and the max systematically picks the largest of these untrained values (red). The noise is then copied into $$Q(s,a)$$, and from there propagates backwards along the trajectory." width="90%" %}

In online Q-learning, you assume that you get to visit enough state-action pairs and that the magic of generalization will fill in the blanks.
In an offline setting, by definition, you can't rely on that. Approaches such as IQL [[1]](#ref-iql) cleverly avoid this problem by never querying out-of-distribution Q-values.

But for a quick implementation sanity check, you can't afford to visit all states and actions... Looking at the figure above, you might think: what about initializing your Q-values to very low values (with the final bias, for instance)?

It might work in some cases but generally won't, as you get caught out by neural network generalization: **your best friend is now betraying you**! These initial values won't stay fixed for long and will start to move along with your in-distribution data.
The most effective workaround I've found for this sanity check is to use SARSA instead of Q-learning. 

Rather than maximizing over all possible next actions, SARSA bootstraps from the next action $$a'$$ actually taken in the dataset:

$$
\mathcal{L}_{\text{SARSA}}(\theta) = \mathbb{E}_{(s,a,r,s',a') \sim \mathcal{D}} \Big[ \big( Q_\theta(s,a) - r - \gamma \, Q_{\bar\theta}(s',a') \big)^2 \Big]
$$

Every quantity in this target is now in-distribution, so a correct implementation should drive the loss to (numerically) zero and recover the Monte Carlo returns of the dataset (which are lower than the Q-learning values, as expected for SARSA).

Here is what this looks like on a tiny toy problem:

{% include figure.html src="/assets/img/q-learning-overfit/overfit.png" alt="TD loss for Q-learning, pessimistically-initialized Q-learning and SARSA" caption="Trying to overfit with Q-learning and SARSA." %}

Notice the orange curve: the pessimistic initialization doesn't help at all; it even diverges slightly earlier. Generalization quickly drags the unseen actions up along with the seen ones, and from there the max does the rest.

But of course SARSA is not what you want in the end, so don't forget to turn it off for your real use case!

I think this problem is a nice way of better understanding the fundamentals of both RL and deep learning, and of building intuition for Q-learning in general.

## References

<ol class="references">
  <li id="ref-iql">I. Kostrikov, A. Nair, S. Levine. <a href="https://arxiv.org/abs/2110.06169">Offline Reinforcement Learning with Implicit Q-Learning</a>. ICLR 2022.</li>
</ol>

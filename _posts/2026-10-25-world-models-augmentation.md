---
layout: post
title: "World models as DAgger-style augmentation factories for imitation"
description: "Using a world model and a predictive inverse dynamics model to generate recovery data for low-data imitation learning."
date: 2026-10-25 10:00:00 +0200
tags: [medium-post, pidm, world-models, few-shot, imitation]
---

Augmentations are standard in deep-learning recipes: they provide inference-time robustness and well-behaved approximations. However, in domains such as RL for robotics or video games, they can be disappointing.
For instance, all the fancy color-based visual augmentations might not be relevant to a video game with fixed lighting.

But more importantly, they don't tackle the *trajectory* nature of RL.

Symmetry augmentations are closer to what you would call an RL augmentation. Basically, if you flip one world axis, you should be able to relabel the target actions accordingly so that this remains valid training data.
For instance, if you flip the x axis and your action was "go right", then you should relabel it as "go left".
But the transformations you can do like this are limited, and might be irrelevant for your use case, as you might never encounter a mirrored situation at inference time.

{% include figure.html src="/assets/img/world-model-augmentation/symmetry.png" alt="A game frame and its horizontally mirrored copy, with the action relabeled from go right to go left" caption="Symmetry augmentation: mirror the frame along the x axis and relabel the action accordingly." %}

How could we build more generic and trajectory-meaningful augmentations?

## A typical failure case

Let's focus on low-data imitation learning and try to find inspiration from typical failure cases. In low-data imitation, typically, once your agent goes a bit off-track, it gets out of distribution and won't recover. And without using domain knowledge, RL, a human in the loop or other sources of data, well, unfortunately, there's not a lot you can do about this...

{% include figure.html src="/assets/img/world-model-augmentation/compounding-errors.png" alt="Compounding errors in low-data behavioral cloning" width="80%" caption="With few demonstrations, the data only covers a thin tube around the expert trajectories. A small mistake brings the agent to a state it has never seen, where it makes bigger mistakes." %}

If you can afford human time and can have humans interact with your environment, then DAgger [[1]](#ref-dagger) is a great solution. Simply put, DAgger consists in having an expert (in this case a human) take over to recover after the agent has failed.
But human time is expensive, many DAgger iterations might be required, and you might simply not be able to have a human interact with your environment at all.

{% include figure.html src="/assets/img/world-model-augmentation/dagger.png" alt="DAgger: an expert labels recovery actions in the states visited by the policy" width="80%" caption="DAgger: an expert labels the right actions in the states actually visited by the policy." %}

What if we could combine DAgger in spirit with a way to automatically get augmentations, generalizing symmetry augmentations to small sub-trajectories?
What if we could augment our small dataset with "recovery" actions from the neighborhood of the trajectories in our dataset?

But computing the "recovery" actions might not be as trivial as in the symmetry case. And getting the neighboring states might be doable in simple state-based environments, but very hard in visual ones.

Luckily, we can address these two challenges with relatively standard tools in RL, namely: **PIDMs** and **world models**.
Let's first quickly introduce these two concepts if you are not familiar with them.

## Two tools

### Predictive inverse dynamics model (PIDM)

Instead of predicting the next action chunk directly, predict a future state (the *P* part of PIDM), and then predict the action chunk given the current state and this predicted future state (the *IDM* part of PIDM):

$$
\hat s_{t+k} = P_\phi(s_t), \qquad a_{t:t+k} \sim \mathrm{IDM}_\psi\big(\cdot \mid s_t, \hat s_{t+k}\big).
$$

Intuitively, the advantages are:

- IDMs are much easier to train than BC. BC is guessing what someone will play in the future, while an IDM is guessing which button was pressed while watching the screen of someone playing.
- IDMs can be trained on data that has nothing to do with your task, as long as it's the same environment.
- Predicting the next state is better behaved than predicting the next actions, and corresponds better to learning the intent of the player.
- Predicting the next state doesn't need action labels.

### World model

A world model is basically a learned, approximate environment. Typically, it can help with accelerating slow environments, safety concerns, or extracting gradients from the environment dynamics. But a less known advantage is that it gives you a **save/restore state** mechanism. Many environments have this built in, but some don't.
Think, for instance, of a commercial video game. Of course you can save and load the game, but most of the time you won't end up in the exact same place as when you saved.

## Putting them together

Now let's see how we can use these two tools to help with our problem.

First, remember we wanted to add neighboring states. The world model enables exactly that: just sample actions, unroll them in the world model, and there you are. But what distribution should we use for the sampling? Uniform would be wasteful, as it would produce neighboring states that will never occur during inference. So we should use the policy itself, in an exploratory mode.

Next, what action labels do we need for these neighboring states?
Looking at the diagram, we would like the actions that bring us back to the trajectory. We can't just use the ground-truth actions, as they wouldn't correct the trajectory. That's where the PIDM comes into play: whatever neighboring state we ended up in, we would still like to go where we initially wanted to go. And since the IDM takes care of turning "where to go" into actions, we only need to label the *target state*, not the actions. Translated into a training objective, this gives:

$$
\hat s_{t+h} = \mathrm{WM}\big(s_t, \hat a_{t:t+h}\big), \quad \hat a_{t:t+h} \sim \pi_{\text{explore}},
\qquad
\mathcal{L}_{\text{aug}}(\phi) = \mathbb{E}_{\tau \sim \mathcal{D},\, t,\, \hat a_{t:t+h}} \Big[ \ell\big(P_\phi(\hat s_{t+h}),\, s_{t+h+k}\big) \Big].
$$

{% include figure.html src="/assets/img/world-model-augmentation/augmentation.png" alt="Augmentation scheme with a world model and a PIDM" caption="From a dataset state $$s_t$$, the world model imagines an exploratory rollout that drifts to a neighboring state $$\hat s_{t+h}$$. Instead of the dataset action, the new training pair asks the predictor $$P_\phi$$ to still aim for the original future state $$s_{t+h+k}$$; the IDM then produces the recovery actions." %}

## Limitations

Now, of course, as is often the case with example-inspired solutions, this approach comes with a number of flaws in the general case.

Once you're in state $$\hat s_{t+h}$$, nothing guarantees that $$s_{t+h+k}$$ is reachable from $$\hat s_{t+h}$$. Worse, reaching $$s_{t+h+k}$$ might not be the best thing to do from $$\hat s_{t+h}$$ when your environment is non-Markovian.

Imagine you're in a video game and $$s_{t+h}$$ happens to be a state where you just opened a chest and got a key, and this key is not included in your state. Then $$s_{t+h+k}$$ is a state where the player opens a door with this key. If $$\hat s_{t+h}$$ doesn't have the key, then it should target $$s_{t+h}$$, not $$s_{t+h+k}$$.

## Comparison with DITTO

A few weeks after thinking about this, I discovered a paper called DITTO [[2]](#ref-ditto), which is similar in several respects.

Simply put, DITTO applies online RL to rollouts in a world model starting from states of the demonstrations, with a reward defined as the similarity between the embeddings of the policy rollout and the dataset rollout.

{% include figure.html src="/assets/img/world-model-augmentation/ditto.png" alt="DITTO: a policy rollout in the world model is rewarded for staying close to the dataset trajectory" caption="DITTO: starting from a dataset state $$s_t$$, the policy is rolled out in the world model. At each step, it is rewarded by the similarity between the embeddings $$z$$ of the imagined state $$\hat s_i$$ and the dataset state $$s_i$$, and trained with RL." %}

Let's quickly compare both approaches:

- Both approaches heavily leverage the save/restore state capability of the world model. Without it, going back to a state to perform the rollouts requires a seedable environment and would be slow.
- DITTO's objective helps the policy not to go wrong, while this approach rather learns to fix the mess after errors were made (same as DAgger).
- This approach seems simpler than DITTO, as it only acts as a dataset augmentation for the PIDM, while DITTO requires RL objectives.

## Conclusion

Although far from perfect, I think this solution is a good example of combining existing ideas, guided by a concrete failure case.

Unfortunately, I haven't had the time to test this approach yet.
If you give it a try, I'd love to hear your feedback!

## References

<ol class="references">
  <li id="ref-dagger">S. Ross, G. Gordon, D. Bagnell. <a href="https://arxiv.org/abs/1011.0686">A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning</a>. AISTATS 2011.</li>
  <li id="ref-ditto">B. DeMoss, P. Duckworth, J. Foerster, N. Hawes, I. Posner. <a href="https://arxiv.org/abs/2302.03086">DITTO: Offline Imitation Learning with World Models</a>. 2023.</li>
</ol>

---
layout: post
title: "Action frequency and real-time environments"
date: 2025-09-15 10:00:00 +0200
description: "How the action frequency affects real-time agents: action independence, argmax vs sampling, latency, partial observability and data."
image: /assets/img/realtime-envs/latency.png
tags: [medium-post, real-time-env, action-space, pomdp]
---

Research environments are usually turn-based: the environment waits for the agent to act. Real-time environments, such as video games or robots, don't wait. This makes the action frequency, i.e. how many times per second your agent acts, an important design choice, and one with a lot of surprising side effects. Here are a few I've run into.

## Fast independence is better than slow independence

In research environments, action spaces are often simple: either multi-categorical or continuous, and usually small (say, less than 10k dimensions).
In real-life environments, action spaces can be much larger, and simplifications based on meaningful approximations become necessary.

A typical simplifying approximation is independence between action dimensions. It turns one problem of size $$A^n$$ into $$n$$ problems of size $$A$$. For instance, with Q-learning, you get $$n$$ independent Q-heads of size $$A$$ instead of one Q-head of size $$A^n$$. Tokenizers such as FAST [[1]](#ref-fast) are another way to tame large action spaces, by compressing action chunks into a few tokens.

Something worth noting, I think, is that for real-time environments, *if your agent acts at a high enough frequency, you can more or less ignore the dependency between action dimensions, as long as you condition on the recent past*.

Imagine a video game with a button for running. The running button and the stick won't be independent: knowing the running button is pressed, a non-idle stick is much more likely.
If the time granularity is fine, the model might make an error on the first frame, predicting the running button with an idle stick, but very soon after, it will correct it.

{% include figure.html src="/assets/img/realtime-envs/dependency.png" alt="Run button and stick sampled independently over five frames" width="60%" caption="Sampling the run button and the stick independently: at $$t=1$$ the agent runs in place, but conditioning on that frame fixes the stick at $$t=2$$." %}

In practice, *doubling the action frequency is a bit like sampling in two passes*: once unconditionally, and a second time conditionally on the first sample.

## Argmax can be worse than sampling

In RL, you typically use your policy in explore mode (higher temperature, additional noise, …) during training and in exploit mode (lower temperature, little or no noise) during evaluation.

In low-data imitation learning, using a predictive (non-diffusion) policy with no or limited temporal context in exploit mode (argmax, in this case) might give you worse results than sampling, especially when the action frequency is high.

Take the example of a task where the demonstrator goes forward and then turns left around a target point. In practice, an imperfect, stochastic demonstrator won't turn at exactly the same place every time. We can model this shift around the target point as a small Gaussian.

{% include figure.html src="/assets/img/realtime-envs/straight-line.png" alt="Demonstrators turn left around a target point; argmax never turns" width="50%" caption="Demonstrators turn left at slightly different places. Sampling from the policy turns somewhere in that range; argmax never turns." %}

In low-data imitation learning, the model will overfit to this distribution. At each frame, the probability of turning left is roughly the Gaussian's mass over that frame's duration, so it shrinks as the frequency grows. Going forward always has the highest probability, and argmax never turns: *the higher the fps, the less the model turns in exploit mode*.

## Latency

When running in real-world, real-time environments, you get latency. Whether it comes from your code or from things you don't control makes no difference in the end, as long as it's consistent and your demonstration data is recorded with the same lag.

If there is a 100 ms lag and the demonstrations are good, it means the demonstrator, who played with the same lag, learned to act 100 ms in advance.

If we wanted to be really precise, we would also take into account human reaction time (around 200 ms): the demonstrator's actions are already late by that much, so we could shift the action labels by the difference between the human reaction time and the model's inference time.

{% include figure.html src="/assets/img/realtime-envs/latency.png" alt="Timeline of frames, human and model actions with reaction time, inference time and lag" width="85%" caption="Both the human and the model see frame f0. The human reacts in about 200 ms, the model in its inference time; both actions then take effect after the same 100 ms lag." %}

## Delays turn MDPs into POMDPs

Another effect to take into account with delayed real-time environments is that they turn an MDP into a POMDP, *unless you include in the observation the actions you played during the last lag seconds*.

Imagine you have a model without temporal context, and your agent needs to open a menu in a game. It presses the "open menu" button, but because of the lag, the menu doesn't appear in the next frames. Without knowing it has already pressed the button, the agent presses it again, which closes the menu.

{% include figure.html src="/assets/img/realtime-envs/delay-pomdp.png" alt="Opening a menu with lag at high and low fps" width="70%" caption="With a lag of several frames, the screen doesn't change after the first press, so a memoryless agent presses again. At low fps, the result is visible in the next frame." %}

Here again, *there is an interaction with the frequency you choose*. If the frequency is low enough, you will "directly" see the result of your action in the next frame.

## More data for free at low fps

Another trick that favors low-fps inference: if you train your model for 10 fps inference while your data is recorded at 30 fps, you can actually triple your data. Each 30 fps replay can be turned into three 10 fps replays, using frame indices modulo 3.

{% include figure.html src="/assets/img/realtime-envs/frame-subsampling.png" alt="A 30 fps recording split into three interleaved 10 fps recordings" width="80%" caption="One 30 fps recording gives three 10 fps recordings, offset by one frame." %}

Of course, these replays look quite alike, so it's not really like having three different demonstrations. But I think it can be thought of as having two small augmentations of the same trajectory. Pushing this further, we could even get an unlimited number of augmentations by interpolating between frames, not by averaging them, which just gives blur, but with optical flow or similar techniques.

## Temporal context and causal confusion

Finally, I want to mention an important point: adding temporal context can (and will) lead to causal confusion [[2]](#ref-causal), where the policy learns to copy its own past actions instead of reacting to what it sees. The higher the frequency, the more consecutive actions look alike, and the stronger the temptation to copy.

{% include figure.html src="/assets/img/realtime-envs/copy-tokens.png" alt="A smooth action binned per timestep at 10 Hz and 50 Hz" caption="The same smooth action, discretized per timestep. At higher frequency, many more tokens are just a copy of the previous one." %}

The FAST paper [[1]](#ref-fast) studies exactly this effect of the sampling rate: with naive per-timestep tokens, an autoregressive policy mostly learns to repeat the previous token, and performance drops as the frequency increases. Compressing the action chunk (with a DCT followed by byte pair encoding) makes each token carry real information again and gets rid of the problem. However, combining this kind of compressed encoding with time conditioning is not trivial.

## Conclusion

If I had to summarize this post in a few rules of thumb for real-time imitation learning:
- pick the lowest frequency that still lets you solve the task;
- record demonstrations with the same latency as at inference time;
- give the policy the actions it played during the last lag period;
- prefer sampling over argmax when your policy has little temporal context;
- be careful with action history, and compress it if you use it.

## References

<ol class="references">
  <li id="ref-fast">K. Pertsch, K. Stachowicz, B. Ichter, D. Driess, S. Nair, Q. Vuong, O. Mees, C. Finn, S. Levine. <a href="https://arxiv.org/abs/2501.09747">FAST: Efficient Action Tokenization for Vision-Language-Action Models</a>. RSS 2025.</li>
  <li id="ref-causal">P. de Haan, D. Jayaraman, S. Levine. <a href="https://arxiv.org/abs/1905.11979">Causal Confusion in Imitation Learning</a>. NeurIPS 2019.</li>
</ol>

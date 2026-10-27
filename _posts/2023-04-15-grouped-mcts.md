---
layout: post
title: "Grouped MCTS: searching tactical games one battle at a time"
date: 2023-04-15 10:00:00 +0200
tags: [long-post, mcts, model-based, online, action-space]
---

MCTS-based agents such as AlphaZero [[1]](#ref-az) have shown impressive scaling, from chess to Go. But that scaling has limits. In tactical games like Fire Emblem, XCOM or Civilization, each turn consists of *many* actions, one or more per unit, and these actions depend on each other. Vanilla MCTS won't work out of the box there without an absurd amount of compute.

I spent quite some time trying to make an AlphaZero-style agent work on a turn-based tactics game of this kind. This post collects the main problems I ran into and the ideas that came out of them. The central observation is simple: **big maps are not one big battle, but several small ones**.

## The action space

The first question is how to represent a turn. A whole turn as a single action is out of the question, as the number of combinations explodes. A natural decomposition is to act unit by unit: first choose *which unit* to play, then choose *what to do* with it (where to move, whom to attack, which property to capture, …). Let's call the first policy $$\pi_\text{unit}$$ and the second one $$\pi_\text{action}$$.

Both are spatial: $$\pi_\text{unit}$$ is a distribution over the board positions of your playable units, and $$\pi_\text{action}$$ over the reachable destinations and targets of the selected unit. This fits a convolutional network nicely and keeps every policy head small.

But there is one action that doesn't fit: **ending the turn**. Sometimes the best thing to do with many units is to wait, so you need a way to stop before every unit has played.

## The end-turn trap

The most natural way to add the end-turn action is to make it one more candidate of $$\pi_\text{unit}$$, next to the units. It turns out this is a bad idea, for two reasons that amplify each other.

First, putting it at the same level as a unit choice gives it the same "scale" in the search: the same prior mass, the same exploration bonus. But end turn is a very special action: it prevents you from visiting all the states where your remaining units act. Giving it the same exploration bonus as moving one unit is rarely what you want.

Second, ending the turn changes the observation much more than any other action: the opponent now plays a whole turn. The value of the state after ending the turn is therefore much harder to estimate, and **its estimate is much noisier** than the value after a unit move. Since the search keeps the best-looking action, a noisy estimate that happens to be too high regularly wins the comparison.

{% include figure.html src="/assets/img/grouped-mcts/end-turn-noise.png" alt="Noisy value estimate of end turn vs precise estimates of unit moves" width="80%" caption="Values of the states reached by each action, as estimated by the network. Unit moves are estimated precisely; ending the turn leads to a much less predictable state. When the spread is larger than the actual loss in value, end turn often looks best." %}

In practice, the result is an agent that ends its turn far too often, leaving units idle.

Here is roughly how my fixes evolved:

1. **Penalize early end turns.** A small negative reward for ending the turn, proportional to the number of units that haven't acted yet. It helps, but it's a hack and it biases the objective.
2. **Remove the global end turn.** Instead, each unit gets its own "wait" action, and the turn ends implicitly once every unit has acted. The decision is now local and has the same scale as the unit's other actions, which removes most of the imbalance.
3. **Discount per turn, not per action.** With many actions per turn, a per-action discount factor makes a turn with ten units much "longer" than a turn with two. Discounting only when the player changes makes values comparable across turns.

But ending the turn early is actually *very* useful inside the search: it lets the tree quickly see how the opponent can respond, an artificial shortening of the horizon. So instead of removing it entirely, I kept it as an **early end turn** that exists only inside the tree, as we'll see below.

The general lesson, I think, is that you cannot blindly design an action space without thinking about the balance and scale of each action, especially in search, where priors, exploration bonuses and value noise all interact.

## Big maps are several small battles

On a large map, the game naturally splits into *sub-battles*: a fight around a city in the north, an infantry rush in the south, some units just walking towards the front. What happens in one of them barely affects the others, at least in the short term.

{% include figure.html src="/assets/img/grouped-mcts/sub-battles.png" alt="A map with two independent local battles" width="70%" caption="Two local battles on the same map. Within a turn, the units of one battle barely influence the other." %}

This structure causes two related problems.

**Credit assignment.** With a single scalar reward, a good trade in the north and a disaster in the south get mixed into the same number. The network has to figure out on its own which units are responsible for what, from relatively few games.

The first fix is to **spatialize rewards**. Instead of one reward per step, each reward is attributed to the entity that caused it: the damage dealt goes to the attacking unit (and the damage taken to the unit that was hit), a capture goes to the capturing unit, and global rewards such as winning or losing go to a "commander" entity. The value head then predicts a value *per entity*, spatially, on the board. A unit fighting in the north is now credited for what happens in the north.

**Search.** The second problem is combinatorial. If you have ten units, the tree has to consider their actions in sequence, and the orders interleave: unit 1 then 2, or 2 then 1, and so on. Most of this is wasted, as the order of independent units doesn't matter. Worse, the depth needed to see the opponent's answer grows with the number of units.

Both problems would be much easier if we knew the sub-battles. Each one is close to an **independent sub-MDP**: we could search within it, credit rewards within it, and ignore the rest.

## Finding the sub-battles

The simplest approach is a heuristic. In my case: start from a seed unit, then add the units that can reach the same enemies, and the enemies they can reach, and so on. This works decently but is brittle: what counts as "interacting" depends on the units, the terrain and the situation.

A more principled approach is to **measure** independence. Two units are independent if what one does doesn't change what the other wants to do. And that can be measured directly with the policy: compare unit B's policy in a state where it plays right away, $$\pi_B$$, with its policy after unit A has played, $$\pi_{B \mid A}$$:

$$
d(A, B) = \tfrac{1}{2}\Big(\mathrm{KL}\big(\pi_B \,\|\, \pi_{B \mid A}\big) + \mathrm{KL}\big(\pi_{B \mid A} \,\|\, \pi_B\big)\Big).
$$

If the divergence is close to zero, A and B are in different sub-battles. If it's large, they are in the same one.

{% include figure.html src="/assets/img/grouped-mcts/divergence.png" alt="Policy of unit B before and after unit A acts" caption="Measuring independence. When A is far away, B's policy doesn't change after A acts. When A is next to B, it does, so A and B belong to the same group." %}

The nice thing with MCTS is that the tree gives you these pairs of states for free: you often visit both "B plays first" and "A plays, then B" from the same node. I accumulated these divergences during search (weighted by visit counts, as rarely visited nodes have unreliable policies), and used them as training targets for a **grouper network**, a graph/attention network that predicts the pairwise divergences directly from the state. At play time, the groups come from this network instead of the hand-crafted heuristic. Comparing values instead of policies is another option, but policies were the more informative signal in my experiments.

## Grouped MCTS

Once you have groups, you can use them to cut the tree:

1. At the root, pick a seed unit and build its group.
2. Restrict the search to the units of this group. Units outside the group don't appear in the tree.
3. Once the group has acted, apply an **early end turn**: the tree skips the remaining units and lets the opponent play.
4. On the opponent's side, restrict the search to the **counter group**: the enemy units that interact with your group.

{% include figure.html src="/assets/img/grouped-mcts/grouped-tree.png" alt="Search tree restricted to a group, with an early end turn and the opponent's counter group" width="85%" caption="Grouped search: only the units of the current group are expanded, then an early end turn lets the opponent's counter group respond." %}

This is where the end turn finds its proper place: it is not a real action of the policy anymore, but a tool of the search that artificially shortens the horizon, so that the tree can evaluate a local plan against the opponent's local answer within a few plies. In the real game, the group's units are played, then a new group is formed from the remaining units, until everyone has acted.

(Under the hood, the search itself followed the view of MCTS as regularized policy optimization [[2]](#ref-grill), which made it easy to plug in these restrictions.) The search becomes much shallower and narrower, and the policy and value targets produced by the tree are cleaner, as they don't mix unrelated decisions.

## What worked, what didn't

The action-space fixes were the clearest wins: removing the global end turn and discounting per turn made the agent stop wasting its units, and made learning much more stable.

Spatialized rewards and grouping helped on large maps, but they come with a lot of engineering complexity: groups must be recomputed when they're exhausted, the tree has to deal with induced groups and counter groups, and a bad grouping can hide an important interaction (an artillery unit far away can still matter). The learned grouper was promising, but it adds yet another network whose errors interact with the search.

## Conclusion

Tactical games are a great stress test for MCTS: many actions per turn, strong dependencies between some of them and none between others, and a huge branching factor. The main lesson for me was that the structure of the problem matters more than raw search power. The action space has to be balanced, and the near-independence of sub-battles is something you can, and probably should, exploit: for credit assignment, for the search, and for deciding when the opponent should get to answer.

I think the idea of *measuring* independence from policy divergences goes beyond games: any multi-agent or factored action problem where you'd like to know which decisions can be made separately could use it.

## References

<ol class="references">
  <li id="ref-az">D. Silver et al. <a href="https://arxiv.org/abs/1712.01815">Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm</a>. 2017.</li>
  <li id="ref-grill">J.-B. Grill et al. <a href="https://arxiv.org/abs/2007.12509">Monte-Carlo Tree Search as Regularized Policy Optimization</a>. ICML 2020.</li>
</ol>

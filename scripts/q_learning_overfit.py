# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "matplotlib"]
# ///
"""Toy experiment for the post "Can Q-learning overfit?".

Tries to overfit a tiny offline dataset with a small MLP Q-network, comparing:
  - Q-learning (bootstraps on max_a' Q(s', a'))
  - Q-learning with a pessimistic initialization of the final bias
  - SARSA (bootstraps on the next action actually taken in the dataset)

Pure numpy so it runs anywhere:  uv run scripts/q_learning_overfit.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "..", "assets", "img", "q-learning-overfit")

N_STATES, N_ACTIONS, OBS_DIM = 60, 16, 16
N_TRAJ, TRAJ_LEN = 4, 12
GAMMA = 0.97
HIDDEN = 128
STEPS = 30_000
LR = 3e-4
TAU = 0.005
SEEDS = [0, 1, 2, 3, 4]


def make_dataset(rng):
    feats = rng.normal(size=(N_STATES, OBS_DIM)).astype(np.float32)
    s, a, r, s2, a2, done, ret = [], [], [], [], [], [], []
    for _ in range(N_TRAJ):
        states = rng.choice(N_STATES, size=TRAJ_LEN + 1, replace=False)
        actions = rng.integers(N_ACTIONS, size=TRAJ_LEN + 1)
        for t in range(TRAJ_LEN):
            terminal = t == TRAJ_LEN - 1
            s.append(states[t]); a.append(actions[t]); s2.append(states[t + 1]); a2.append(actions[t + 1])
            r.append(1.0 if terminal else 0.0); done.append(float(terminal))
            ret.append(GAMMA ** (TRAJ_LEN - 1 - t))  # true return of the logged behavior
    d = dict(s=np.array(s), a=np.array(a), r=np.array(r, np.float32), s2=np.array(s2),
             a2=np.array(a2), done=np.array(done, np.float32), ret=np.array(ret, np.float32))
    d["x"], d["x2"] = feats[d["s"]], feats[d["s2"]]
    return d


def init_params(rng, final_bias=0.0):
    def lin(i, o):
        return rng.normal(scale=np.sqrt(2.0 / i), size=(i, o)).astype(np.float32), np.zeros(o, np.float32)
    w1, b1 = lin(OBS_DIM, HIDDEN)
    w2, b2 = lin(HIDDEN, HIDDEN)
    w3, b3 = lin(HIDDEN, N_ACTIONS)
    w3 *= 0.1
    b3 += final_bias
    return [w1, b1, w2, b2, w3, b3]


def forward(p, x):
    w1, b1, w2, b2, w3, b3 = p
    h1 = np.maximum(x @ w1 + b1, 0)
    h2 = np.maximum(h1 @ w2 + b2, 0)
    return h2 @ w3 + b3, (x, h1, h2)


def backward(p, cache, dq):
    w1, b1, w2, b2, w3, b3 = p
    x, h1, h2 = cache
    gw3, gb3 = h2.T @ dq, dq.sum(0)
    dh2 = (dq @ w3.T) * (h2 > 0)
    gw2, gb2 = h1.T @ dh2, dh2.sum(0)
    dh1 = (dh2 @ w2.T) * (h1 > 0)
    gw1, gb1 = x.T @ dh1, dh1.sum(0)
    return [gw1, gb1, gw2, gb2, gw3, gb3]


def train(d, mode, seed, final_bias=0.0):
    rng = np.random.default_rng(1000 + seed)
    p = init_params(rng, final_bias)
    tgt = [w.copy() for w in p]
    m = [np.zeros_like(w) for w in p]
    v = [np.zeros_like(w) for w in p]
    n = len(d["a"])
    idx = np.arange(n)
    log = {"td": [], "q_err": [], "q_mean": []}
    for step in range(1, STEPS + 1):
        q_next, _ = forward(tgt, d["x2"])
        boot = q_next.max(1) if mode == "q" else q_next[idx, d["a2"]]
        y = d["r"] + GAMMA * (1 - d["done"]) * boot
        q, cache = forward(p, d["x"])
        q_sa = q[idx, d["a"]]
        err = q_sa - y
        dq = np.zeros_like(q)
        dq[idx, d["a"]] = 2 * err / n
        grads = backward(p, cache, dq)
        for i, g in enumerate(grads):  # Adam
            m[i] = 0.9 * m[i] + 0.1 * g
            v[i] = 0.999 * v[i] + 0.001 * g * g
            p[i] -= LR * (m[i] / (1 - 0.9 ** step)) / (np.sqrt(v[i] / (1 - 0.999 ** step)) + 1e-8)
        for i in range(len(p)):
            tgt[i] = (1 - TAU) * tgt[i] + TAU * p[i]
        if step % 100 == 0:
            log["td"].append(float(np.mean(err ** 2)))
            log["q_err"].append(float(np.mean(np.abs(q_sa - d["ret"]))))
            log["q_mean"].append(float(np.mean(q_sa)))
    return {k: np.array(val) for k, val in log.items()}


def smooth(x, alpha=0.1):
    out, acc = np.empty_like(x), x[0]
    for i, v in enumerate(x):
        acc = (1 - alpha) * acc + alpha * v
        out[i] = acc
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    runs = {
        "Q-learning": dict(mode="q"),
        "Q-learning, pessimistic init": dict(mode="q", final_bias=-5.0),
        "SARSA": dict(mode="sarsa"),
    }
    colors = {"Q-learning": "#d62728", "Q-learning, pessimistic init": "#ff7f0e", "SARSA": "#1f77b4"}
    results = {name: [] for name in runs}
    for seed in SEEDS:
        d = make_dataset(np.random.default_rng(seed))
        for name, kw in runs.items():
            results[name].append(train(d, seed=seed, **kw))
            last = results[name][-1]
            print(f"seed={seed} {name:30s} td={last['td'][-1]:.2e}  |Q-G|={last['q_err'][-1]:.3f}  meanQ={last['q_mean'][-1]:.3f}")

    # The loss spans ~25 orders of magnitude, so seeds are smoothed and aggregated in log space.
    steps = np.arange(1, STEPS // 100 + 1) * 100
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(7, 4))
    for name in runs:
        logs = np.stack([smooth(np.log10(r["td"])) for r in results[name]])
        mean, std = logs.mean(0), logs.std(0)
        c = colors[name]
        ax.plot(steps, 10 ** mean, color=c, lw=2, label=name)
        ax.fill_between(steps, 10 ** (mean - std), 10 ** (mean + std), color=c, alpha=0.2, lw=0)
    ax.set_yscale("log")
    ax.set_xlabel("gradient steps")
    ax.set_ylabel("TD loss on the dataset")
    ax.grid(alpha=0.3, which="major")
    ax.legend(frameon=False, fontsize=10, loc="center right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "overfit.png"), dpi=200)
    print("saved", os.path.join(OUT, "overfit.png"))


if __name__ == "__main__":
    main()

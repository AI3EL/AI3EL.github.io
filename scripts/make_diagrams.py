# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "matplotlib"]
# ///
"""Schematic figures for the Q-learning and world-model posts.

    uv run scripts/make_diagrams.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

ROOT = os.path.join(os.path.dirname(__file__), "..", "assets", "img")
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

BLUE, ORANGE, RED, GREEN, GRAY = "#1f77b4", "#ff7f0e", "#d62728", "#2ca02c", "#888888"


def save(fig, sub, name):
    os.makedirs(os.path.join(ROOT, sub), exist_ok=True)
    path = os.path.join(ROOT, sub, name)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


# ---------------------------------------------------------------- world model
def expert_path(x):
    return 0.6 * np.sin(x / 1.6)


def arrow(ax, p, q, color, ls="-", lw=1.6, rad=0.0, head=12):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=head, color=color, lw=lw,
                                 ls=ls, connectionstyle=f"arc3,rad={rad}", zorder=4))


def draw_support(ax, x):
    y = expert_path(x)
    ax.fill_between(x, y - 0.35, y + 0.35, color=BLUE, alpha=0.10, lw=0)
    ax.plot(x, y, color=BLUE, lw=2, zorder=3)
    xs = np.linspace(x[0], x[-1], 9)
    ax.scatter(xs, expert_path(xs), color=BLUE, s=22, zorder=4)


def compounding_errors():
    x = np.linspace(0, 10, 300)

    def panel():
        fig, ax = plt.subplots(figsize=(6.2, 3.1))
        draw_support(ax, x)
        ax.set_xlim(-0.3, 10.6)
        ax.set_ylim(-1.6, 2.6)
        ax.axis("off")
        ax.text(0.0, -1.35, "demonstrations (+ region covered by data)", color=BLUE, fontsize=10)
        return fig, ax

    # Behavioral cloning drifts, leaves the data support and never comes back.
    fig, ax = panel()
    xa = np.linspace(0, 10, 300)
    drift = expert_path(xa) + 0.012 * np.maximum(xa - 2.0, 0) ** 2.4
    ax.plot(xa, drift, color=RED, lw=2, ls="--", zorder=3)
    ax.text(9.0, 2.2, "OOD: the policy has\nnever seen these states", color=RED, ha="center", fontsize=10)
    save(fig, "world-model-augmentation", "compounding-errors.png")

    # DAgger: an expert labels recovery actions from the states the policy visits.
    fig, ax = panel()
    xb = np.linspace(0, 6.5, 200)
    drift_b = expert_path(xb) + 0.012 * np.maximum(xb - 2.0, 0) ** 2.4
    ax.plot(xb, drift_b, color=RED, lw=2, ls="--", zorder=3)
    for xi in (4.0, 5.0, 6.0):
        p = (xi, expert_path(xi) + 0.012 * (xi - 2.0) ** 2.4)
        q = (xi + 1.1, expert_path(xi + 1.1) + 0.05)
        arrow(ax, p, q, GREEN, rad=0.25)
    ax.text(7.6, 1.5, "expert labels\nrecovery actions", color=GREEN, ha="center", fontsize=10)
    save(fig, "world-model-augmentation", "dagger.png")


def augmentation_scheme():
    x = np.linspace(0, 10, 300)
    fig, ax = plt.subplots(figsize=(9, 3.8))
    draw_support(ax, x)
    ax.axis("off")
    ax.set_xlim(-0.5, 10.8)
    ax.set_ylim(-1.3, 2.5)

    t0, th, thk = 1.25, 5.0, 8.75  # on the dataset dots drawn by draw_support
    s_t, s_th, s_thk = [(v, expert_path(v)) for v in (t0, th, thk)]
    for (px, py), lab, dy in [(s_t, r"$s_t$", -0.35), (s_th, r"$s_{t+h}$", -0.42), (s_thk, r"$s_{t+h+k}$", -0.42)]:
        ax.scatter([px], [py], s=70, color=BLUE, zorder=5, edgecolor="white")
        ax.text(px, py + dy, lab, ha="center", va="top", fontsize=12, color=BLUE)

    # Exploratory rollout imagined by the world model.
    xr = np.linspace(t0, th, 60)
    yr = expert_path(xr) + 0.9 * np.sin(np.pi * (xr - t0) / (th - t0) / 2) ** 1.5
    ax.plot(xr, yr, color=ORANGE, lw=2, ls="--", zorder=3)
    s_hat = (xr[-1], yr[-1])
    ax.scatter(*s_hat, s=70, color=ORANGE, zorder=5, edgecolor="white")
    ax.text(s_hat[0] - 0.1, s_hat[1] + 0.2, r"$\hat s_{t+h}$", ha="center", fontsize=12, color=ORANGE)
    ax.text(2.4, 1.45, "imagined rollout in the world model\n" r"$\hat a_{t:t+h} \sim \pi_{\mathrm{explore}}$",
            ha="center", fontsize=10, color=ORANGE)

    # New training pair: from the perturbed state, still aim for the original future state.
    arrow(ax, s_hat, s_thk, GREEN, lw=2, rad=-0.25, head=14)
    ax.text(8.3, 1.35, "new PIDM target:\n" r"$\hat s_{t+h} \rightarrow s_{t+h+k}$", ha="center", fontsize=10, color=GREEN)
    ax.text(-0.3, -1.15, "dataset trajectory", color=BLUE, fontsize=10)
    save(fig, "world-model-augmentation", "augmentation.png")


def ditto_scheme():
    x = np.linspace(0, 10, 300)
    fig, ax = plt.subplots(figsize=(9, 3.8))
    draw_support(ax, x)
    ax.axis("off")
    ax.set_xlim(-0.5, 10.8)
    ax.set_ylim(-1.3, 2.5)

    # Policy rollout imagined by the world model, starting from a dataset state (same shape as augmentation.png).
    t0, t1 = 1.25, 5.0
    drift = lambda v: expert_path(v) + 0.9 * np.sin(np.pi * (v - t0) / (t1 - t0) / 2) ** 1.5
    xr = np.linspace(t0, t1, 60)
    ax.plot(xr, drift(xr), color=ORANGE, lw=2, ls="--", zorder=3)
    s_t = (t0, expert_path(t0))
    ax.scatter(*s_t, s=70, color=BLUE, zorder=5, edgecolor="white")
    ax.text(s_t[0], s_t[1] - 0.35, r"$s_t$", ha="center", va="top", fontsize=12, color=BLUE)

    # Step-by-step reward: similarity between embeddings of matching imagined and dataset states.
    for xi in (2.5, 3.75, 5.0):  # on the dataset dots drawn by draw_support
        yi_hat, yi = drift(xi), expert_path(xi)
        ax.plot([xi, xi], [yi + 0.08, yi_hat - 0.08], color=GREEN, lw=1.8, ls=":", zorder=4)
        ax.scatter([xi], [yi_hat], s=45, color=ORANGE, zorder=5, edgecolor="white")
        ax.scatter([xi], [yi], s=45, color=BLUE, zorder=5, edgecolor="white")

    ax.text(2.4, 1.45, "policy rollout in the world model\n" r"$\hat a_i \sim \pi_\theta(\hat s_i)$",
            ha="center", fontsize=10, color=ORANGE)
    ax.text(8.0, 1.35, r"reward $r_i = \mathrm{sim}(z(\hat s_i),\, z(s_i))$" "\n→ optimize $\\pi_\\theta$ with RL",
            ha="center", fontsize=10, color=GREEN)
    ax.text(-0.3, -1.15, "dataset trajectory", color=BLUE, fontsize=10)
    save(fig, "world-model-augmentation", "ditto.png")


# ---------------------------------------------------------------- q-learning
def max_contamination():
    from matplotlib.patches import Circle, Rectangle

    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.set_xlim(0, 12.4)
    ax.set_ylim(-1.5, 4.6)
    ax.set_aspect("equal")
    ax.axis("off")

    # Dataset transition (s, a, r, s').
    for (x, lab) in [(1.0, "$s$"), (4.2, "$s'$")]:
        ax.add_patch(Circle((x, 1.0), 0.45, fc="white", ec=BLUE, lw=2, zorder=3))
        ax.text(x, 1.0, lab, ha="center", va="center", fontsize=15, color=BLUE, zorder=4)
    arrow(ax, (1.5, 1.0), (3.7, 1.0), BLUE, lw=2, head=14)
    ax.text(2.6, 1.22, "$a,\\ r$", ha="center", fontsize=12, color=BLUE)
    ax.text(2.6, 0.45, "transition in\nthe dataset", ha="center", va="top", fontsize=10, color=BLUE)

    # Q(s', .) over all actions: one seen in the dataset, the others never trained.
    x0, w, gap, base = 6.0, 0.5, 0.72, 0.0
    q = [0.9, 0.4, 0.75, 1.5, 0.25, 2.7, 0.55, 1.15]
    seen, imax = 2, int(np.argmax(q))
    for i, v in enumerate(q):
        x = x0 + i * gap
        color = BLUE if i == seen else (RED if i == imax else "#c8c8c8")
        ax.add_patch(Rectangle((x, base), w, v, fc=color, ec="none", zorder=3))
    ax.plot([x0 - 0.2, x0 + 7 * gap + w + 0.2], [base, base], color="black", lw=1)
    ax.text(x0 + seen * gap + w / 2, -0.12, "$a'$\nin data", ha="center", va="top", fontsize=10, color=BLUE)
    ax.text(x0 + imax * gap + w / 2, -0.12, "never\nseen", ha="center", va="top", fontsize=10, color=RED)
    ax.text(x0 + 3.5 * gap + w / 2, -1.15, "$Q(s', a')$ for every action $a'$ at $s'$", ha="center", fontsize=10.5, color=GRAY)
    ax.text(x0 + 7 * gap + w + 0.35, 1.9, "gray: random\ninitialization,\nnever trained", fontsize=9.5,
            color=GRAY, va="center")
    ax.plot([4.7, x0 - 0.3], [1.0, 0.6], color=GRAY, lw=1, ls=":")

    # The max picks the noisy value, which becomes the regression target of an in-dataset pair.
    top = (x0 + imax * gap + w / 2, q[imax] + 0.08)
    arrow(ax, top, (1.15, 1.5), RED, lw=2, rad=0.32, head=15)
    ax.text(5.1, 4.15, r"target for $Q(s,a)$:  $r + \gamma\, \mathrm{max}_{a'}\, Q(s', a')$",
            ha="center", fontsize=12, color=RED)
    ax.text(1.0, -0.2, "in the dataset,\nbut its target is not", ha="center", va="top", fontsize=10, color=RED)
    save(fig, "q-learning-overfit", "max-noise.png")


def symmetry_augmentation():
    from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle

    W, H, gap = 10.0, 5.2, 1.6

    def scene(ax, x0, flip, action, title):
        fx = (lambda x: x0 + W - x) if flip else (lambda x: x0 + x)

        def rect(x, y, w, h, **kw):
            ax.add_patch(Rectangle((min(fx(x), fx(x + w)), y), w, h, **kw))

        ax.add_patch(FancyBboxPatch((x0, 0), W, H, boxstyle="round,pad=0,rounding_size=0.25",
                                    fc="#eef5fb", ec="#c9d6e3", lw=1.2, zorder=0))
        rect(0, 0, W, 0.8, fc="#8fbf7a", ec="none", zorder=1)                 # ground
        rect(5.6, 2.2, 2.6, 0.45, fc="#b07c52", ec="none", zorder=1)          # platform
        rect(3.4, 0.8, 0.9, 0.9, fc="#9e9e9e", ec="none", zorder=1)           # rock
        ax.add_patch(Polygon([(fx(8.55), 0.8), (fx(8.55), 2.0), (fx(9.35), 1.7), (fx(8.55), 1.4)],
                             closed=False, fc=RED, ec="#555", lw=1.2, zorder=2))  # flag
        ax.add_patch(Circle((fx(7.0), 3.15), 0.32, fc="#ffc845", ec="#d9a520", lw=1.2, zorder=2))  # coin
        rect(1.35, 0.8, 0.7, 1.0, fc=BLUE, ec="none", zorder=3)              # agent body
        ax.add_patch(Circle((fx(1.7), 2.15), 0.34, fc=BLUE, ec="none", zorder=3))
        ax.add_patch(Circle((fx(1.82), 2.22), 0.08, fc="white", ec="none", zorder=4))  # eye: shows facing
        arrow(ax, (fx(2.3), 1.45), (fx(3.15), 1.45), ORANGE, lw=3, head=16)
        ax.text(x0 + W / 2, H + 0.35, title, ha="center", va="bottom", fontsize=11)
        ax.text(x0 + W / 2, -0.35, f"action: {action}", ha="center", va="top", fontsize=12,
                color=ORANGE, fontweight="bold")

    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.set_aspect("equal")
    ax.axis("off")
    scene(ax, 0, False, "go right →", "original frame")
    scene(ax, W + gap, True, "← go left", "mirrored frame, relabeled action")
    mx = W + gap / 2
    ax.plot([mx, mx], [-0.1, H + 0.1], color=GRAY, ls="--", lw=1.4)
    ax.text(mx, H / 2, "flip\nx", ha="center", va="center", fontsize=10, color=GRAY,
            bbox=dict(fc="white", ec="none", pad=1.5))
    ax.set_xlim(-0.2, 2 * W + gap + 0.2)
    ax.set_ylim(-1.0, H + 1.0)
    save(fig, "world-model-augmentation", "symmetry.png")


if __name__ == "__main__":
    symmetry_augmentation()
    max_contamination()
    compounding_errors()
    augmentation_scheme()
    ditto_scheme()

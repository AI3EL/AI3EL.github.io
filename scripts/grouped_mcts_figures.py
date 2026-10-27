# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "matplotlib"]
# ///
"""Figures for the post on grouped MCTS for tactical games.

    uv run scripts/grouped_mcts_figures.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

OUT = os.path.join(os.path.dirname(__file__), "..", "assets", "img", "grouped-mcts")
BLUE, RED, ORANGE, GREEN, GRAY, PURPLE = "#3a78c2", "#d1495b", "#f09228", "#2ca02c", "#888888", "#6a3d9a"
plt.rcParams.update({"font.size": 10})


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print("saved", path)


def arrow(ax, p, q, color, lw=1.6, rad=0.0, head=10, ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=head, color=color, lw=lw, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", zorder=4, shrinkA=6, shrinkB=6))


def end_turn_noise():
    """Q estimates: unit moves are precise, end turn is noisy; its best sample often wins the max."""
    rng = np.random.default_rng(0)
    xs = np.linspace(-1.6, 1.6, 400)
    gauss = lambda m, s: np.exp(-0.5 * ((xs - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))
    fig, ax = plt.subplots(figsize=(6.6, 2.8))
    moves = [(0.25, 0.08), (0.1, 0.08), (0.18, 0.08)]
    for i, (m, s) in enumerate(moves):
        ax.fill_between(xs, gauss(m, s) * 0.18, color=BLUE, alpha=0.35, lw=0, label="unit moves" if i == 0 else None)
    ax.fill_between(xs, gauss(-0.05, 0.45) * 0.18, color=RED, alpha=0.35, lw=0, label="end turn")
    ax.axvline(0.25, color=BLUE, lw=1.2, ls="--")
    ax.axvline(-0.05, color=RED, lw=1.2, ls="--")
    sample = 0.62
    ax.scatter([sample], [0.02], color=RED, s=40, zorder=5)
    ax.annotate("a lucky estimate\nwins the max", xy=(sample, 0.03), xytext=(0.85, 0.45), fontsize=9, color=RED,
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.2))
    ax.text(0.25, 1.02, "best move", ha="left", va="bottom", fontsize=9, color=BLUE, transform=ax.get_xaxis_transform())
    ax.text(-0.05, 1.02, "end turn (true value)", ha="right", va="bottom", fontsize=9, color=RED,
            transform=ax.get_xaxis_transform())
    ax.set_yticks([])
    ax.set_xlabel("estimated value of the next state")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    ax.set_xlim(-1.6, 1.6)
    save(fig, "end-turn-noise.png")


def map_grid(ax, w, h):
    for x in range(w + 1):
        ax.plot([x, x], [0, h], color="#e2e2e2", lw=0.8, zorder=0)
    for y in range(h + 1):
        ax.plot([0, w], [y, y], color="#e2e2e2", lw=0.8, zorder=0)
    ax.set_aspect("equal")
    ax.axis("off")


def unit(ax, x, y, color, label=None):
    ax.add_patch(Circle((x + 0.5, y + 0.5), 0.33, fc=color, ec="white", lw=1.2, zorder=3))
    if label:
        ax.text(x + 0.5, y + 0.5, label, ha="center", va="center", fontsize=7.5, color="white", zorder=4)


def sub_battles():
    """A big map with two local battles; groups and counter groups."""
    w, h = 16, 9
    fig, ax = plt.subplots(figsize=(7.2, 4.1))
    map_grid(ax, w, h)
    north_blue, north_red = [(2, 6), (3, 7), (4, 6)], [(6, 7), (7, 6), (6, 5)]
    south_blue, south_red = [(9, 1), (10, 2), (11, 1), (9, 3)], [(13, 2), (14, 1), (13, 3)]
    for x, y in north_blue + south_blue:
        unit(ax, x, y, BLUE)
    for x, y in north_red + south_red:
        unit(ax, x, y, RED)
    for (x0, y0, ww, hh, lab, col) in [(1.6, 4.6, 6.8, 3.8, "battle 1", PURPLE), (8.6, 0.6, 6.8, 3.8, "battle 2", GREEN)]:
        ax.add_patch(FancyBboxPatch((x0, y0), ww, hh, boxstyle="round,pad=0,rounding_size=0.5", fc="none",
                                    ec=col, lw=2, ls="--", zorder=2))
        ax.text(x0 + 0.15, y0 + hh + 0.12, lab, color=col, fontsize=10, va="bottom")
    ax.text(w / 2, -0.35, "blue: my units, red: opponent's units", ha="center", va="top", fontsize=9, color="#555")
    ax.set_xlim(-0.2, w + 0.2)
    ax.set_ylim(-0.9, h + 0.6)
    save(fig, "sub-battles.png")


def divergence():
    """Unit B's policy before and after unit A acts: unchanged if A is far away, changed if A is close."""
    rng = np.random.default_rng(3)
    base = rng.dirichlet(np.ones(6) * 1.3)
    same = base * np.exp(rng.normal(0, 0.1, 6))
    same /= same.sum()
    diff = np.roll(base, 2) * np.exp(rng.normal(0, 0.3, 6))
    diff /= diff.sum()

    fig = plt.figure(figsize=(8.0, 3.1))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1, 1], hspace=0.75, wspace=0.3)
    cases = [("A far from B", (0.5, 0.5), (5.5, 2.5), same, GREEN, "KL ≈ 0: independent"),
             ("A next to B", (2.6, 1.5), (3.6, 1.5), diff, RED, "large KL: same sub-battle")]
    for r, (title, posA, posB, after, col, verdict) in enumerate(cases):
        ax = fig.add_subplot(gs[r, 0])
        map_grid(ax, 7, 3)
        for (x, y), lab, c in ((posA, "A", ORANGE), (posB, "B", BLUE)):
            ax.add_patch(Circle((x + 0.5 - 0.5, y + 0.5 - 0.5), 0.36, fc=c, ec="white", lw=1.2, zorder=3))
            ax.text(x, y, lab, ha="center", va="center", fontsize=8, color="white", zorder=4)
        enemy = (4.6, 0.5) if r == 0 else (5.0, 1.5)
        ax.add_patch(Circle(enemy, 0.36, fc=RED, ec="white", lw=1.2, zorder=3))
        ax.set_xlim(-0.1, 7.1)
        ax.set_ylim(-0.1, 3.1)
        ax.set_title(title, fontsize=9.5, color="#333", pad=3)
        for c_i, (p, lab) in enumerate(((base, r"$\pi_B$ before A acts"), (after, r"$\pi_B$ after A acts"))):
            a = fig.add_subplot(gs[r, 1 + c_i])
            a.bar(range(6), p, color=BLUE, width=0.75)
            a.set_xticks([])
            a.set_yticks([])
            a.set_ylim(0, max(base.max(), diff.max()) * 1.15)
            for s in a.spines.values():
                s.set_visible(False)
            a.set_title(lab, fontsize=9.5, color=BLUE, pad=3)
        fig.text(0.92, 0.72 - 0.47 * r, verdict, ha="left", va="center", fontsize=9.5, color=col)
    save(fig, "divergence.png")


def grouped_tree():
    """Search restricted to a group, early end turn, then the opponent's counter group."""
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    ax.axis("off")
    dy = 1.05
    y_root, y1, y2, y3, y4 = 0, -dy, -2 * dy, -3 * dy, -4 * dy

    def node(x, y, col, hollow=False):
        ax.add_patch(Circle((x, y), 0.11, fc="white" if hollow else col, ec=col, lw=1.4, zorder=3))

    def edge(p, q, col, ls="-"):
        ax.plot([p[0], q[0]], [p[1], q[1]], color=col, lw=1.6, ls=ls, zorder=1)

    root = (2.4, y_root)
    node(*root, "#333")
    lvl1 = [(0.9, y1), (2.4, y1), (3.9, y1)]
    for p in lvl1:
        edge(root, p, BLUE)
        node(*p, BLUE)
    lvl2 = [(0.4, y2), (1.4, y2), (2.4, y2), (3.9, y2)]
    for p, parent in zip(lvl2, (lvl1[0], lvl1[0], lvl1[1], lvl1[2])):
        edge(parent, p, BLUE)
        node(*p, BLUE)
    lvl3 = [(x, y3) for x, _ in lvl2]
    for p, parent in zip(lvl3, lvl2):
        edge(parent, p, GRAY, "--")
        node(*p, GRAY, hollow=True)
    for parent in lvl3[:2]:
        for o in (-0.22, 0.22):
            q = (parent[0] + o, y4)
            edge(parent, q, RED)
            node(*q, RED)
    for x in (2.4, 3.9):
        ax.text(x, y4, "…", ha="center", va="center", fontsize=12, color=RED)

    lx = 4.8
    for y, label, col in [(y_root, "root", "#333"), (y1, "my group: unit 1", BLUE), (y2, "my group: unit 2", BLUE),
                          (y3, "early end turn", GRAY), (y4, "counter group: enemy units", RED)]:
        ax.text(lx, y, label, va="center", fontsize=9.5, color=col)
    ax.set_xlim(-0.2, 7.6)
    ax.set_ylim(y4 - 0.4, 0.4)
    save(fig, "grouped-tree.png")


if __name__ == "__main__":
    end_turn_noise()
    sub_battles()
    divergence()
    grouped_tree()

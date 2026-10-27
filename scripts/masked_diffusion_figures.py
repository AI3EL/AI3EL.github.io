# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "matplotlib"]
# ///
"""Small per-section figures for the post "Ordering in masked discrete diffusion".

All figures show the same action chunk (D action dimensions x T timesteps) at one unmasking step:
gray "?" = masked, blue = already unmasked, orange = unmasked at this step, red = flagged / remasked.

    uv run scripts/masked_diffusion_figures.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = os.path.join(os.path.dirname(__file__), "..", "assets", "img", "masked-diffusion-ordering")
T, D = 8, 3
MASKED, DONE, NOW, BAD = "#e6e6e6", "#4c84c4", "#f09228", "#d62728"
plt.rcParams.update({"font.size": 10})


def new_fig(width=4.6, height=1.9):
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def cell(ax, t, d, state, x0=0.0, text=None, lw=0, ec="none", text_color=None):
    color = {"m": MASKED, "d": DONE, "n": NOW, "b": BAD}[state]
    y = D - 1 - d
    ax.add_patch(FancyBboxPatch((x0 + t + 0.06, y + 0.06), 0.88, 0.88, boxstyle="round,pad=0,rounding_size=0.12",
                                fc=color, ec=ec, lw=lw, zorder=2))
    if text is None and state == "m":
        text = "?"
    if text:
        ax.text(x0 + t + 0.5, y + 0.5, text, ha="center", va="center", zorder=3,
                fontsize=8.5 if len(text) > 1 else 10,
                color=text_color or ("#999" if state == "m" else "white"))


def grid(ax, states, x0=0.0, texts=None, labels=True):
    for d in range(D):
        for t in range(T):
            cell(ax, t, d, states[d][t], x0, None if texts is None else texts[d][t])
    if labels:
        for d in range(D):
            ax.text(x0 - 0.15, D - 1 - d + 0.5, f"$a^{d}$", ha="right", va="center", fontsize=9, color="#555")
        ax.text(x0 + T / 2, -0.35, "time →", ha="center", va="top", fontsize=9, color="#555")


def arrow(ax, p, q, color, lw=1.6, rad=0.0, head=10, alpha=1.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=head, color=color, lw=lw,
                                 connectionstyle=f"arc3,rad={rad}", zorder=4, alpha=alpha))


def save(fig, name, xlim, ylim):
    ax = fig.axes[0]
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print("saved", path)


def heuristics():
    states = [["n" if (t + 3 * d) % 4 == 0 else "m" for t in range(T)] for d in range(D)]
    fig, ax = new_fig()
    grid(ax, states)
    save(fig, "heuristics.png", (-0.8, T + 0.1), (-0.9, D + 0.1))


def temporal():
    states = [["d" if t < 3 else ("n" if t == 3 else "m") for t in range(T)] for d in range(D)]
    fig, ax = new_fig()
    grid(ax, states)
    save(fig, "temporal.png", (-0.8, T + 0.1), (-0.9, D + 0.1))


def wfc():
    rng = np.random.default_rng(4)
    conf = np.clip(rng.beta(2.2, 1.6, size=(D, T)), 0.15, 0.98)
    done = np.zeros((D, T), bool)
    done[0, :2] = done[1, 0] = done[2, :1] = True
    masked_conf = np.where(done, -1, conf)
    top = np.argsort(masked_conf.ravel())[::-1][:4]
    chosen = np.zeros(D * T, bool)
    chosen[top] = True
    chosen = chosen.reshape(D, T)
    fig, ax = new_fig()
    for d in range(D):
        for t in range(T):
            if done[d, t]:
                cell(ax, t, d, "d")
            elif chosen[d, t]:
                cell(ax, t, d, "n", text=f"{conf[d, t]:.2f}".lstrip("0"))
            else:
                cell(ax, t, d, "m", text=f"{conf[d, t]:.2f}".lstrip("0"), text_color="#888")
    for d in range(D):
        ax.text(-0.15, D - 1 - d + 0.5, f"$a^{d}$", ha="right", va="center", fontsize=9, color="#555")
    ax.text(T / 2, -0.35, "time →", ha="center", va="top", fontsize=9, color="#555")
    save(fig, "wfc.png", (-0.8, T + 0.1), (-0.9, D + 0.1))


def learned():
    from matplotlib.colors import LinearSegmentedColormap, to_hex

    purple = LinearSegmentedColormap.from_list("infl", [MASKED, "#6a3d9a"])
    # Predicted effect of unmasking the orange token on each other masked token (closer = stronger).
    src_t, src_d = 4, 1
    fig, ax = new_fig(height=2.4)
    for d in range(D):
        for t in range(T):
            if t < 2:
                cell(ax, t, d, "d")
            elif (t, d) == (src_t, src_d):
                cell(ax, t, d, "n")
            else:
                infl = np.exp(-0.55 * (abs(t - src_t) + 1.6 * abs(d - src_d)))
                y = D - 1 - d
                ax.add_patch(FancyBboxPatch((t + 0.06, y + 0.06), 0.88, 0.88, boxstyle="round,pad=0,rounding_size=0.12",
                                            fc=to_hex(purple(infl)), ec="none", zorder=2))
                ax.text(t + 0.5, y + 0.5, "?", ha="center", va="center", zorder=3, fontsize=10,
                        color="white" if infl > 0.45 else "#999")
    for d in range(D):
        ax.text(-0.15, D - 1 - d + 0.5, f"$a^{d}$", ha="right", va="center", fontsize=9, color="#555")
    ax.text(T / 2, -0.35, "time →", ha="center", va="top", fontsize=9, color="#555")
    ax.text(T / 2, D + 0.2, "predicted influence of unmasking the orange token", ha="center", va="bottom",
            fontsize=9, color="#6a3d9a")
    save(fig, "learned.png", (-0.8, T + 0.1), (-0.9, D + 0.7))


def correct():
    before = [["d"] * T for _ in range(D)]
    for d, t in [(0, 6), (0, 7), (1, 7), (2, 5), (2, 6), (2, 7)]:
        before[d][t] = "m"
    after = [row[:] for row in before]
    after[1][3] = "m"
    gap = T + 2.2
    fig, ax = new_fig(width=8.4, height=2.2)
    grid(ax, before)
    cell(ax, 3, 1, "b", text=".03", lw=0)
    grid(ax, after, x0=gap, labels=False)
    ax.text(gap + T / 2, -0.35, "time →", ha="center", va="top", fontsize=9, color="#555")
    cell(ax, 3, 1, "m", x0=gap, lw=1.8, ec=BAD)
    arrow(ax, (T + 0.35, D / 2), (gap - 0.35, D / 2), BAD, lw=2, head=14)
    ax.text((T + gap) / 2, D / 2 + 0.3, "p < x:\nremask", ha="center", va="bottom", fontsize=9, color=BAD)
    save(fig, "correct.png", (-0.8, gap + T + 0.1), (-0.9, D + 0.1))


if __name__ == "__main__":
    heuristics()
    temporal()
    wfc()
    learned()
    correct()

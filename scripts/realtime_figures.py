# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "matplotlib"]
# ///
"""Figures for the post "Action frequency and real-time environments".

    uv run scripts/realtime_figures.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

OUT = os.path.join(os.path.dirname(__file__), "..", "assets", "img", "realtime-envs")
BLUE, ORANGE, RED, GREEN, GRAY = "#4c84c4", "#f09228", "#d62728", "#2ca02c", "#888888"
plt.rcParams.update({"font.size": 10})


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print("saved", path)


def box(ax, x, y, text, fc, tc="white", ec="none", lw=0, w=0.9, h=0.8, fs=9.5):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.1",
                                fc=fc, ec=ec, lw=lw, zorder=2))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=tc, zorder=3)


def dependency_fix():
    """Running button and stick sampled independently at each frame: one incoherent frame, then recovery."""
    run = ["off", "on", "on", "on", "on"]
    stick = ["idle", "idle", "fwd", "fwd", "fwd"]
    fig, ax = plt.subplots(figsize=(5.6, 1.9))
    ax.set_aspect("equal")
    ax.axis("off")
    for t in range(5):
        box(ax, t, 1, run[t], BLUE)
        bad = t == 1
        box(ax, t, 0, stick[t], RED if bad else BLUE, ec=RED if bad else "none", lw=0)
        ax.text(t, -0.65, f"t={t}", ha="center", va="top", fontsize=9, color="#555")
    ax.text(-0.65, 1, "run", ha="right", va="center", fontsize=10, color="#555")
    ax.text(-0.65, 0, "stick", ha="right", va="center", fontsize=10, color="#555")
    ax.text(1, 1.65, "incoherent:\nrunning in place", ha="center", va="bottom", fontsize=9, color=RED)
    ax.set_xlim(-1.6, 4.6)
    ax.set_ylim(-1.1, 2.5)
    save(fig, "dependency.png")


def straight_line():
    """Demonstrators turn left around a target point; an argmax policy never turns."""
    rng = np.random.default_rng(2)
    turns = np.sort(rng.normal(6.0, 0.5, 7))
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(plt.Rectangle((-0.6, turns.min() - 0.15), 1.2, np.ptp(turns) + 0.3, fc="#f3e3cf", ec="none", zorder=0))
    for i, yt in enumerate(turns):
        ax.plot([0, 0, -3.5], [0, yt, yt], color=GRAY, lw=1.2, alpha=0.6, zorder=1,
                label="demonstrations" if i == 0 else None)
    ax.plot([0.06, 0.06, -3.5], [0, 6.0, 6.0], color=GREEN, lw=2.6, zorder=3, label="sampling: turns")
    ax.plot([-0.06, -0.06], [0, 9.6], color=RED, lw=2.6, ls="--", zorder=2, label="argmax: never turns")
    ax.annotate("", xy=(-0.06, 10.2), xytext=(-0.06, 9.5), arrowprops=dict(arrowstyle="-|>", color=RED, lw=2))
    ax.scatter([0], [0], s=60, color="#333", zorder=4)
    ax.text(0.35, 0, "start", va="center", fontsize=9, color="#333")
    ax.text(0.8, 6.0, "turning\npoints", va="center", fontsize=9, color="#b07c52")
    ax.legend(frameon=False, loc="upper center", fontsize=9, bbox_to_anchor=(0.5, -0.01), ncol=3,
              handlelength=1.6, columnspacing=1.0)
    ax.set_xlim(-3.8, 2.4)
    ax.set_ylim(-0.4, 10.4)
    save(fig, "straight-line.png")


def frame_subsampling():
    """One 30 fps recording turned into three interleaved 10 fps recordings."""
    n = 12
    colors = [BLUE, ORANGE, GREEN]
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    ax.set_aspect("equal")
    ax.axis("off")
    for i in range(n):
        box(ax, i, 3, str(i), "#9aa5b1")
    ax.text(-0.75, 3, "30 fps", ha="right", va="center", fontsize=10, color="#555")
    for k in range(3):
        y = 1.6 - 1.05 * k
        for i in range(k, n, 3):
            box(ax, i, y, str(i), colors[k])
        ax.text(-0.75, y, f"10 fps #{k + 1}", ha="right", va="center", fontsize=10, color=colors[k])
    ax.set_xlim(-2.6, n - 0.4)
    ax.set_ylim(-0.95, 3.5)
    save(fig, "frame-subsampling.png")


def latency_timeline():
    """When frames are shown, when actions are pressed and when they take effect, for a human and a model."""
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(figsize=(8.4, 3.2))
    ax.axis("off")
    lag, react, infer = 100, 200, 30
    y_frames, y_human, y_model = 3.0, 1.9, 0.8
    for name, y in (("frames", y_frames), ("human", y_human), ("model", y_model)):
        ax.text(-18, y, name, ha="right", va="center", fontsize=10, color="#555")
    for t in range(0, 401, 100):
        ax.add_patch(Rectangle((t - 14, y_frames - 0.25), 28, 0.5, fc="#9aa5b1", ec="none"))
        ax.text(t, y_frames, f"f{t // 100}", ha="center", va="center", fontsize=8.5, color="white")
        ax.plot([t, t], [0.3, y_frames - 0.3], color="#e6e6e6", lw=1, zorder=0)

    def bar(y, t0, t1, color, label, above=True):
        ax.add_patch(Rectangle((t0, y - 0.15), t1 - t0, 0.3, fc=color, ec="none", alpha=0.85, zorder=2))
        ax.text((t0 + t1) / 2, y + (0.24 if above else -0.24), label, ha="center", va="bottom" if above else "top",
                fontsize=8.5, color=color)

    def event(t, y, label, color):
        ax.scatter([t], [y], s=50, color=color, zorder=4, edgecolor="white")
        ax.text(t, y - 0.24, label, ha="center", va="top", fontsize=8.5, color=color)

    bar(y_human, 0, react, BLUE, f"reaction ≈ {react} ms")
    bar(y_human, react, react + lag, RED, f"lag {lag} ms")
    event(react, y_human, "press", BLUE)
    event(react + lag, y_human, "effect", RED)
    bar(y_model, 0, infer, GREEN, f"inference\n{infer} ms")
    bar(y_model, infer, infer + lag, RED, f"lag {lag} ms")
    ax.scatter([infer], [y_model], s=50, color=GREEN, zorder=4, edgecolor="white")
    ax.text(infer, y_model - 0.24, "press", ha="center", va="top", fontsize=8.5, color=GREEN)
    event(infer + lag, y_model, "effect", RED)

    ax.annotate("", xy=(infer, 0.3), xytext=(react, 0.3), arrowprops=dict(arrowstyle="<->", color=ORANGE, lw=1.6))
    ax.text((infer + react) / 2, 0.18, f"model presses {react - infer} ms earlier than the human:\nshift the action labels",
            ha="center", va="top", fontsize=8.5, color=ORANGE)
    ax.text(330, 0.18, "time →", ha="left", va="top", fontsize=9, color="#555")
    ax.set_xlim(-90, 420)
    ax.set_ylim(-0.6, 3.35)
    save(fig, "latency.png")


def delay_pomdp():
    """Pressing 'open menu' with a 3-frame lag: the screen doesn't change, so a memoryless agent presses again."""
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    ax.set_aspect("equal")
    ax.axis("off")
    # High fps: 8 frames, lag = 3 frames. Presses at t=1 and t=2 open the menu at t=4 and close it at t=5.
    screen = ["game", "game", "game", "game", "menu", "game", "game", "game"]
    presses = {1: BLUE, 2: RED}
    yh = 4.2
    ax.text(-0.5, yh + 1.25, "high fps (lag = 3 frames)", ha="left", va="bottom", fontsize=10)
    for t in range(8):
        box(ax, t + 0.5, yh + 0.55, screen[t], ORANGE if screen[t] == "menu" else "#9aa5b1", w=0.92, h=0.8, fs=8)
        p = presses.get(t)
        box(ax, t + 0.5, yh - 0.45, "press" if p else "–", p or "#eeeeee", tc="white" if p else "#999",
            w=0.92, h=0.8, fs=8)
    ax.text(4.0, yh - 0.95, "no menu yet: presses again → the menu opens, then closes", ha="center", va="top",
            fontsize=8.5, color=RED)
    # Low fps: 2 frames of 4 high-fps frames each; the lag is shorter than one frame.
    yl = 0.6
    ax.text(-0.5, yl + 1.25, "low fps (lag < 1 frame)", ha="left", va="bottom", fontsize=10)
    for k, (scr, pressed) in enumerate((("game", True), ("menu", False))):
        box(ax, 4 * k + 2, yl + 0.55, scr, ORANGE if scr == "menu" else "#9aa5b1", w=3.92, h=0.8, fs=8.5)
        box(ax, 4 * k + 2, yl - 0.45, "press" if pressed else "–", BLUE if pressed else "#eeeeee",
            tc="white" if pressed else "#999", w=3.92, h=0.8, fs=8.5)
    for y in (yh + 0.55, yl + 0.55):
        ax.text(-0.15, y, "screen", ha="right", va="center", fontsize=9, color="#555")
        ax.text(-0.15, y - 1.0, "action", ha="right", va="center", fontsize=9, color="#555")
    ax.text(4, -0.5, "time →", ha="center", va="top", fontsize=9, color="#555")
    ax.set_xlim(-1.6, 8.1)
    ax.set_ylim(-1.0, yh + 1.8)
    save(fig, "delay-pomdp.png")


def copy_tokens():
    """A smooth action binned per timestep: at high fps almost every token repeats the previous one."""
    t_fine = np.linspace(0, 1, 400)
    signal = lambda t: 0.6 * np.sin(2 * np.pi * 1.3 * t) + 0.25 * np.sin(2 * np.pi * 3.1 * t + 1)
    bins = np.linspace(-1, 1, 17)
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 2.6), sharey=True)
    for ax, fps in zip(axes, (10, 50)):
        ts = np.arange(0, 1, 1 / fps)
        tokens = np.digitize(signal(ts), bins)
        repeat = tokens[1:] == tokens[:-1]
        ax.plot(t_fine, signal(t_fine), color="#ccc", lw=1.5, zorder=1)
        q = (bins[np.clip(tokens - 1, 0, len(bins) - 2)] + bins[np.clip(tokens, 0, len(bins) - 1)]) / 2
        ax.step(ts, q, where="post", color=BLUE, lw=1.4, zorder=2)
        ax.scatter(ts[1:][repeat], q[1:][repeat], s=14, color=RED, zorder=3, label="same token as previous")
        ax.scatter(ts[1:][~repeat], q[1:][~repeat], s=14, color=GREEN, zorder=3, label="new token")
        ax.set_title(f"{fps} Hz: {repeat.mean():.0%} of tokens copy the previous one", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xlabel("time in chunk (1 s)", fontsize=9, color="#555")
    axes[0].legend(frameon=False, fontsize=8.5, loc="lower left")
    fig.tight_layout(w_pad=2)
    save(fig, "copy-tokens.png")


if __name__ == "__main__":
    dependency_fix()
    straight_line()
    frame_subsampling()
    latency_timeline()
    delay_pomdp()
    copy_tokens()

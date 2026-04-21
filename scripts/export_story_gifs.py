"""Render short animated GIFs from the sim traces for slide embedding.

Two GIFs:
- `story_fixed_fails.gif`    — Fixed PID on the swapped chassis. Drifts off.
- `story_fixed_vs_bandit.gif` — Fixed vs the best-performing bandit (EpsilonGreedy)
                                on the same swapped chassis.

EpsilonGreedy is the hero because on the dashboard's canonical swapped chassis
it picks (kp=5.0, kd=1.5) and holds the line cleanly, while LinUCB happens to
pick near-Fixed gains and drifts too. The aggregate holdout (slide 5) still
shows all three bandits cut Fixed's 79% violation rate to 21-43%.

Uses matplotlib + PillowWriter so no extra deps beyond what run_experiments already needs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, PillowWriter

ROOT = Path(__file__).resolve().parent.parent
DASH = ROOT / "public" / "dashboard.json"
OUT = ROOT / "public"

NAVY = "#012169"
COPPER = "#C84E00"
ROYAL = "#00539B"
HATTERAS = "#E2E6ED"
WHISPER = "#F3F2F1"


def load():
    d = json.loads(DASH.read_text())
    line = d["line"]
    line_x = [p["x"] for p in line]
    line_y = [p["y"] for p in line]
    return d, line_x, line_y


def auto_xlim(*trace_xs_lists, pad: float = 0.6) -> tuple:
    """Crop the x-axis to where the robot actually goes (plus a small buffer)."""
    x_max = max(max(xs) for xs in trace_xs_lists)
    return (0.0, round(x_max + pad, 2))


def render_single(policy_name: str, chassis: str, out_path: Path,
                  label: str, color: str = COPPER, step_stride: int = 3):
    """One robot on the line."""
    d, lx, ly = load()
    trace = d["policies"][policy_name][chassis]
    xs, ys, errs = trace["xs"], trace["ys"], trace["errs"]
    n = len(xs)

    fig, ax = plt.subplots(figsize=(8.5, 2.4), dpi=110)
    fig.patch.set_facecolor(WHISPER)
    ax.set_facecolor("white")

    xlim = auto_xlim(xs)
    visible_line = [(x, y) for x, y in zip(lx, ly) if xlim[0] <= x <= xlim[1]]
    vlx, vly = zip(*visible_line)

    ax.plot(vlx, vly, color=NAVY, linewidth=2.4, label="target line", zorder=3)
    ax.fill_between(vlx, [y - 0.5 for y in vly], [y + 0.5 for y in vly],
                    color=COPPER, alpha=0.08, zorder=1)

    # Violation markers — faint copper dots where |err| > 0.5
    viol_xs = [xs[i] for i in range(n) if abs(errs[i]) > 0.5 and xs[i] <= xlim[1]]
    viol_ys = [ys[i] for i in range(n) if abs(errs[i]) > 0.5 and xs[i] <= xlim[1]]

    trail, = ax.plot([], [], color=color, linewidth=2.0, zorder=4)
    robot, = ax.plot([], [], marker="o", markersize=8, color=color, zorder=5)
    violations, = ax.plot([], [], marker="o", markersize=4, color=COPPER,
                          alpha=0.6, linestyle="none", zorder=3.5)

    ax.set_xlim(*xlim)
    ax.set_ylim(0, 4)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ("top", "right", "bottom", "left"):
        ax.spines[s].set_visible(False)
    ax.text(0.05 * xlim[1], 3.7, label, fontsize=11, color=NAVY, weight="bold",
            family="DejaVu Sans")

    frames = list(range(0, n, step_stride)) + [n - 1] * 10

    def update(i):
        end = min(i, n - 1)
        trail.set_data(xs[: end + 1], ys[: end + 1])
        robot.set_data([xs[end]], [ys[end]])
        vx = [x for j, x in enumerate(viol_xs) if j < len([k for k in range(end + 1) if abs(errs[k]) > 0.5 and xs[k] <= xlim[1]])]
        vy = viol_ys[: len(vx)]
        violations.set_data(vx, vy)
        return trail, robot, violations

    anim = FuncAnimation(fig, update, frames=frames, interval=60, blit=True)
    anim.save(out_path, writer=PillowWriter(fps=20))
    plt.close(fig)
    print(f"wrote {out_path} ({out_path.stat().st_size/1e3:.0f} KB, xlim={xlim})")


def render_vs(p1: str, p2: str, chassis: str, out_path: Path,
              label1: str, label2: str, step_stride: int = 3):
    """Two robots on the same line, side by side in time."""
    d, lx, ly = load()
    t1 = d["policies"][p1][chassis]
    t2 = d["policies"][p2][chassis]
    n = min(len(t1["xs"]), len(t2["xs"]))

    fig, ax = plt.subplots(figsize=(8.5, 2.6), dpi=110)
    fig.patch.set_facecolor(WHISPER)
    ax.set_facecolor("white")

    xlim = auto_xlim(t1["xs"], t2["xs"])
    visible_line = [(x, y) for x, y in zip(lx, ly) if xlim[0] <= x <= xlim[1]]
    vlx, vly = zip(*visible_line)

    ax.plot(vlx, vly, color=NAVY, linewidth=2.4, zorder=3)
    ax.fill_between(vlx, [y - 0.5 for y in vly], [y + 0.5 for y in vly],
                    color=COPPER, alpha=0.08, zorder=1)

    trail1, = ax.plot([], [], color=COPPER, linewidth=2.0, zorder=4, alpha=0.95)
    robot1, = ax.plot([], [], marker="o", markersize=8, color=COPPER, zorder=5)
    trail2, = ax.plot([], [], color=ROYAL, linewidth=2.0, zorder=4, alpha=0.95)
    robot2, = ax.plot([], [], marker="o", markersize=8, color=ROYAL, zorder=5)

    ax.set_xlim(*xlim)
    ax.set_ylim(0, 4)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ("top", "right", "bottom", "left"):
        ax.spines[s].set_visible(False)

    ax.text(0.05 * xlim[1], 3.75, label1, fontsize=10.5, color=COPPER, weight="bold",
            family="DejaVu Sans")
    ax.text(0.05 * xlim[1], 3.40, label2, fontsize=10.5, color=ROYAL, weight="bold",
            family="DejaVu Sans")

    frames = list(range(0, n, step_stride)) + [n - 1] * 10

    def update(i):
        end = min(i, n - 1)
        trail1.set_data(t1["xs"][: end + 1], t1["ys"][: end + 1])
        robot1.set_data([t1["xs"][end]], [t1["ys"][end]])
        trail2.set_data(t2["xs"][: end + 1], t2["ys"][: end + 1])
        robot2.set_data([t2["xs"][end]], [t2["ys"][end]])
        return trail1, robot1, trail2, robot2

    anim = FuncAnimation(fig, update, frames=frames, interval=60, blit=True)
    anim.save(out_path, writer=PillowWriter(fps=20))
    plt.close(fig)
    print(f"wrote {out_path} ({out_path.stat().st_size/1e3:.0f} KB)")


def main():
    OUT.mkdir(exist_ok=True)
    render_single("Fixed", "swapped", OUT / "story_fixed_fails.gif",
                  label="Fixed PID on swapped chassis")
    render_vs("Fixed", "EpsilonGreedy", "swapped",
              OUT / "story_fixed_vs_bandit.gif",
              label1="Fixed PID — drifts off",
              label2="EpsilonGreedy — holds the line")


if __name__ == "__main__":
    main()

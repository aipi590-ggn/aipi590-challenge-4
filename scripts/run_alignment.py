"""Reward-hacking demo.

The controller has three knobs: kp, kd, and forward speed. When the reward is
simply `-MAE(line error)`, the bandit discovers it can stop the robot (low
speed) to keep the error sensor near zero. The robot never tracks the curve.
It just idles near its start.

This is the line-follow analogue of OpenAI's CoastRunners boat spinning in
circles to farm power-ups rather than finish the race.

The fix: add a motion penalty so the reward is
    reward = -MAE - lambda * mean(|omega|)  +  speed_bonus  =  -MAE - lambda * mean(|omega|) - mu * (1 - speed).
The equivalent, cleaner framing used here: reward penalizes mean travel deficit
(mu * (target_distance - x_final), clipped) so stopping is no longer free.

Artifacts
---------
results/alignment.json
results/alignment.png
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.bandit import LinUCB
from src.control import PID
from src.eval import build_robot, run_episode, sample_chassis

RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)


def arms_with_speed() -> List[Tuple[float, float, float]]:
    """Arm space now includes a speed scalar. 3 kp * 2 kd * 4 speed = 24 arms."""
    arms = []
    for kp in [1.0, 2.0, 3.5]:
        for kd in [0.3, 0.8]:
            for speed in [0.1, 0.35, 0.7, 1.0]:
                arms.append((kp, kd, speed))
    return arms


def reward_hackable(trace, target_x: float = 10.0):
    errs = trace["errs"]
    return -float(np.mean(np.abs(errs)))


def reward_aligned(trace, target_x: float = 12.0, travel_weight: float = 0.40,
                   motion_weight: float = 0.01):
    errs = trace["errs"]
    omegas = trace["omegas"]
    xs = trace["xs"]
    mae = float(np.mean(np.abs(errs)))
    motion = float(np.mean(np.abs(omegas)))
    travel_deficit = max(0.0, target_x - float(xs[-1])) / target_x
    return -mae - travel_weight * travel_deficit - motion_weight * motion


def train_linucb(reward_fn, arms, episodes: int = 200, alpha: float = 0.6,
                 seed: int = 0):
    bandit = LinUCB(len(arms), 4, alpha=alpha)
    rng = random.Random(seed)
    curve = []
    speeds, kps = [], []
    for ep in range(episodes):
        f, inr, nz = sample_chassis(rng)
        robot = build_robot(f, inr, nz, rng)
        ctx = np.array(robot.context(), dtype=float)
        a = bandit.select(ctx)
        kp, kd, speed = arms[a]
        pid = PID(kp=kp, kd=kd)
        trace = run_episode(robot, pid, steps=300, speed_scale=speed)
        r = reward_fn(trace)
        bandit.update(a, ctx, r)
        curve.append(r)
        speeds.append(speed)
        kps.append(kp)
    return bandit, curve, speeds, kps


def eval_linucb(bandit, arms, n=30, seed=777):
    rng = random.Random(seed)
    out = {
        "rewards_aligned": [], "rewards_hackable": [],
        "speeds": [], "kps": [], "kds": [],
        "maes": [], "x_finals": [], "motions": [],
        "violations": [],
    }
    for _ in range(n):
        f, inr, nz = sample_chassis(rng)
        robot = build_robot(f, inr, nz, rng)
        ctx = np.array(robot.context(), dtype=float)
        a = bandit.select(ctx)
        kp, kd, speed = arms[a]
        pid = PID(kp=kp, kd=kd)
        trace = run_episode(robot, pid, steps=300, speed_scale=speed)
        errs = trace["errs"]
        out["rewards_aligned"].append(reward_aligned(trace))
        out["rewards_hackable"].append(reward_hackable(trace))
        out["speeds"].append(speed)
        out["kps"].append(kp)
        out["kds"].append(kd)
        out["maes"].append(float(np.mean(np.abs(errs))))
        out["x_finals"].append(float(trace["xs"][-1]))
        out["motions"].append(float(np.mean(np.abs(trace["omegas"]))))
        out["violations"].append(int(np.any(np.abs(errs) > 0.5)))
    agg = {k: float(np.mean(v)) if k not in ("violations",) else int(sum(v))
           for k, v in out.items()}
    agg["n"] = n
    agg["raw"] = out
    return agg


def main():
    DUKE_NAVY = "#012169"
    COPPER = "#C84E00"
    PERSIMMON = "#E89923"
    GRAPHITE = "#666666"

    arms = arms_with_speed()

    hack_bandit, hack_curve, hack_speeds, hack_kps = train_linucb(
        reward_hackable, arms, episodes=250, seed=42)
    ok_bandit, ok_curve, ok_speeds, ok_kps = train_linucb(
        reward_aligned, arms, episodes=250, seed=42)

    hack_eval = eval_linucb(hack_bandit, arms, n=30, seed=777)
    ok_eval   = eval_linucb(ok_bandit,   arms, n=30, seed=777)

    out = {
        "hackable_reward_training": {
            "speeds":   hack_eval["raw"]["speeds"],
            "kps":      hack_eval["raw"]["kps"],
            "kds":      hack_eval["raw"]["kds"],
            "maes":     hack_eval["raw"]["maes"],
            "x_finals": hack_eval["raw"]["x_finals"],
            "motions":  hack_eval["raw"]["motions"],
            "violations": hack_eval["raw"]["violations"],
            "rewards_aligned_metric":  hack_eval["raw"]["rewards_aligned"],
            "rewards_hackable_metric": hack_eval["raw"]["rewards_hackable"],
            "summary": {
                "mean_speed":   hack_eval["speeds"],
                "mean_x_final": hack_eval["x_finals"],
                "mean_mae":     hack_eval["maes"],
                "mean_motion":  hack_eval["motions"],
                "total_violations": hack_eval["violations"],
            },
        },
        "aligned_reward_training": {
            "speeds":   ok_eval["raw"]["speeds"],
            "kps":      ok_eval["raw"]["kps"],
            "kds":      ok_eval["raw"]["kds"],
            "maes":     ok_eval["raw"]["maes"],
            "x_finals": ok_eval["raw"]["x_finals"],
            "motions":  ok_eval["raw"]["motions"],
            "violations": ok_eval["raw"]["violations"],
            "rewards_aligned_metric":  ok_eval["raw"]["rewards_aligned"],
            "rewards_hackable_metric": ok_eval["raw"]["rewards_hackable"],
            "summary": {
                "mean_speed":   ok_eval["speeds"],
                "mean_x_final": ok_eval["x_finals"],
                "mean_mae":     ok_eval["maes"],
                "mean_motion":  ok_eval["motions"],
                "total_violations": ok_eval["violations"],
            },
        },
        "diagnosis": {
            "note": (
                "With a speed knob in the arm space, LinUCB trained on -MAE "
                "converges on slow-speed arms. The robot idles near x=0 and "
                "collects near-zero error. Adding a travel-deficit term to "
                "the reward makes this strategy unprofitable, and the bandit "
                "learns full-speed arms with well-tuned PID gains."
            ),
        },
    }
    (RESULTS / "alignment.json").write_text(json.dumps(out, indent=2))

    # ─── Plot ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3))

    # Training reward curves
    axes[0].plot(hack_curve, color=COPPER, alpha=0.35, linewidth=0.8)
    axes[0].plot(ok_curve,   color=DUKE_NAVY, alpha=0.35, linewidth=0.8)
    w = 15
    sm_h = np.convolve(hack_curve, np.ones(w) / w, mode="valid")
    sm_o = np.convolve(ok_curve,   np.ones(w) / w, mode="valid")
    axes[0].plot(range(w - 1, len(hack_curve)), sm_h, color=COPPER,
                 linewidth=2.0, label="hackable reward")
    axes[0].plot(range(w - 1, len(ok_curve)),   sm_o, color=DUKE_NAVY,
                 linewidth=2.0, label="aligned reward")
    axes[0].set_xlabel("training episode")
    axes[0].set_ylabel("reward (each under its own reward shape)")
    axes[0].set_title("online training curves")
    axes[0].legend(loc="lower right", frameon=False)
    axes[0].grid(alpha=0.25)

    # Selected speed over time
    axes[1].plot(hack_speeds, color=COPPER, alpha=0.25, linewidth=0.8)
    axes[1].plot(ok_speeds,   color=DUKE_NAVY, alpha=0.25, linewidth=0.8)
    sm_hs = np.convolve(hack_speeds, np.ones(w) / w, mode="valid")
    sm_os = np.convolve(ok_speeds,   np.ones(w) / w, mode="valid")
    axes[1].plot(range(w - 1, len(hack_speeds)), sm_hs, color=COPPER,
                 linewidth=2.0, label="hackable")
    axes[1].plot(range(w - 1, len(ok_speeds)),   sm_os, color=DUKE_NAVY,
                 linewidth=2.0, label="aligned")
    axes[1].set_xlabel("training episode")
    axes[1].set_ylabel("selected forward speed")
    axes[1].set_title("what the bandit picks")
    axes[1].legend(loc="best", frameon=False)
    axes[1].grid(alpha=0.25)

    # Final x across regimes
    labels = ["hackable training", "aligned training"]
    xfinals = [hack_eval["x_finals"], ok_eval["x_finals"]]
    speeds = [hack_eval["speeds"], ok_eval["speeds"]]
    bars = axes[2].bar(labels, xfinals, color=[COPPER, DUKE_NAVY])
    axes[2].set_ylabel("mean x at end of episode (target 15)")
    axes[2].set_title("did the robot actually move?")
    axes[2].grid(axis="y", alpha=0.25)
    axes[2].axhline(15, color=GRAPHITE, linewidth=0.6, linestyle="--")
    for b, xf, sp in zip(bars, xfinals, speeds):
        axes[2].text(b.get_x() + b.get_width() / 2, b.get_height() + 0.2,
                     f"x_final={xf:.2f}\nspeed={sp:.2f}",
                     ha="center", fontsize=9, color=GRAPHITE)

    plt.tight_layout()
    plt.savefig(RESULTS / "alignment.png", dpi=130)
    plt.close(fig)

    print("=== alignment demo ===")
    print(f"  hackable-trained: mean x_final = {hack_eval['x_finals']:.2f}  "
          f"mean speed = {hack_eval['speeds']:.2f}  mean MAE = {hack_eval['maes']:.3f}  "
          f"violations = {hack_eval['violations']}/30")
    print(f"  aligned-trained : mean x_final = {ok_eval['x_finals']:.2f}  "
          f"mean speed = {ok_eval['speeds']:.2f}  mean MAE = {ok_eval['maes']:.3f}  "
          f"violations = {ok_eval['violations']}/30")
    print(f"Artifacts: {RESULTS}/alignment.json  {RESULTS}/alignment.png")


if __name__ == "__main__":
    main()

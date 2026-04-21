"""Export two episode traces for the dashboard's alignment visualization.

Runs one episode under the hackable policy (slow speed, -MAE reward) and one
under the aligned policy (fast speed, reward with travel-deficit term), on the
same chassis and seed so the comparison is clean. Dumps per-step positions to
`public/alignment_replays.json` for the canvas to animate.

Usage:
    python3 scripts/export_alignment_replays.py
"""
from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.control import PID
from src.eval import build_robot, run_episode
from src.world import line_y

OUT = ROOT / "public" / "alignment_replays.json"


def trace_xytheta(robot_x0, robot_y0, trace):
    """Combine xs, ys from run_episode with thetas reconstructed from consecutive points."""
    xs = trace["xs"].tolist()
    ys = trace["ys"].tolist()
    thetas = []
    prev_x, prev_y = robot_x0, robot_y0
    for x, y in zip(xs, ys):
        dx, dy = x - prev_x, y - prev_y
        if abs(dx) + abs(dy) > 1e-6:
            thetas.append(math.atan2(dy, dx))
        else:
            thetas.append(thetas[-1] if thetas else 0.0)
        prev_x, prev_y = x, y
    return [[round(x, 4), round(y, 4), round(t, 4)] for x, y, t in zip(xs, ys, thetas)]


def main():
    # Deliberately pick a hard-ish chassis so the idling vs tracking is visible.
    friction = 0.7
    inertia = 1.0
    noise = 0.06
    steps = 250

    # Winning arms from the alignment experiment (see run_alignment.py output).
    # Hackable reward drives the bandit toward low speed; aligned drives it to high.
    hackable_kp, hackable_kd, hackable_speed = 2.0, 0.3, 0.26
    aligned_kp, aligned_kd, aligned_speed = 2.0, 0.3, 0.87

    # Matched seeds so the randomness is identical across both.
    rng_h = random.Random(42)
    rng_a = random.Random(42)
    robot_h = build_robot(friction, inertia, noise, rng_h)
    robot_a = build_robot(friction, inertia, noise, rng_a)

    trace_h = run_episode(robot_h, PID(kp=hackable_kp, kd=hackable_kd),
                          steps=steps, speed_scale=hackable_speed)
    trace_a = run_episode(robot_a, PID(kp=aligned_kp, kd=aligned_kd),
                          steps=steps, speed_scale=aligned_speed)

    # Pre-compute the target line for rendering (denser than episode steps).
    line_pts = [[round(x, 3), round(line_y(x), 3)] for x in [i * 0.1 for i in range(0, 201)]]

    bounds = {
        "x_min": 0.0,
        "x_max": 20.0,
        "y_min": 0.5,
        "y_max": 3.5,
    }

    data = {
        "chassis": {"friction": friction, "inertia": inertia, "noise": noise},
        "bounds": bounds,
        "line": line_pts,
        "hackable": {
            "label": "hackable reward (-MAE only)",
            "speed": hackable_speed,
            "kp": hackable_kp, "kd": hackable_kd,
            "trajectory": trace_xytheta(0.0, line_y(0.0), trace_h),
            "x_final": round(float(trace_h["xs"][-1]), 3),
            "mean_abs_err": round(float(abs(trace_h["errs"]).mean()), 4),
        },
        "aligned": {
            "label": "aligned reward (+ travel penalty)",
            "speed": aligned_speed,
            "kp": aligned_kp, "kd": aligned_kd,
            "trajectory": trace_xytheta(0.0, line_y(0.0), trace_a),
            "x_final": round(float(trace_a["xs"][-1]), 3),
            "mean_abs_err": round(float(abs(trace_a["errs"]).mean()), 4),
        },
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data))
    print(f"wrote {OUT} ({OUT.stat().st_size/1e3:.1f} KB)")
    print(f"  hackable: x_final={data['hackable']['x_final']}, mae={data['hackable']['mean_abs_err']}")
    print(f"  aligned:  x_final={data['aligned']['x_final']}, mae={data['aligned']['mean_abs_err']}")


if __name__ == "__main__":
    main()

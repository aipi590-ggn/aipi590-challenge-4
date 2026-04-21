"""Episode runner, aligned reward, and chassis helpers."""
import random
from typing import Dict, Tuple

import numpy as np

from .control import PID
from .world import Robot


def run_episode(
    robot: Robot,
    pid: PID,
    steps: int = 300,
    dt: float = 0.05,
    speed_scale: float = 1.0,
    omega_clip: float = 3.0,
) -> Dict[str, np.ndarray]:
    """Run a single episode. Returns dict of per-step arrays."""
    pid.reset()
    errs = np.zeros(steps)
    ys = np.zeros(steps)
    xs = np.zeros(steps)
    omegas = np.zeros(steps)
    for t in range(steps):
        err = robot.sense_line_error()
        omega = pid.step(err, dt=dt)
        omega = max(-omega_clip, min(omega_clip, omega))
        robot.step(omega, dt=dt, speed_scale=speed_scale)
        errs[t] = err
        ys[t] = robot.y
        xs[t] = robot.x
        omegas[t] = omega
    return {"errs": errs, "ys": ys, "xs": xs, "omegas": omegas}


def reward_aligned(
    trace: Dict[str, np.ndarray],
    motion_penalty: float = 0.05,
    off_line_cap: float = 0.5,
) -> Tuple[float, int]:
    """Primary reward: -MAE minus a small motion penalty. Violation flag on cap."""
    errs = trace["errs"]
    omegas = trace["omegas"]
    mae = float(np.mean(np.abs(errs)))
    motion = float(np.mean(np.abs(omegas)))
    r = -mae - motion_penalty * motion
    viol = int(np.any(np.abs(errs) > off_line_cap))
    return r, viol


def sample_chassis(rng: random.Random) -> Tuple[float, float, float]:
    """Sample a (friction, inertia, noise) triple from the training distribution."""
    friction = rng.choice([0.4, 0.7, 1.0])
    inertia = rng.choice([0.8, 1.0, 2.0])
    noise = 0.05 + rng.random() * 0.04
    return friction, inertia, noise


def build_robot(friction: float, inertia: float, noise: float,
                rng: random.Random) -> Robot:
    return Robot(friction=friction, inertia=inertia, noise=noise, rng=rng)

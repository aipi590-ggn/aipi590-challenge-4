"""Simulator world: target line and differential-drive robot with chassis parameters.

The robot follows a signed-error sensor on a curvy target line. Chassis parameters
(friction, inertia, noise) are what change when a student swaps a motor or tire.
"""
import math
import random
from typing import Optional


def line_y(x: float) -> float:
    """Target line as a function of x. Smooth, piecewise sinusoidal."""
    return 2.0 + 0.8 * math.sin(0.6 * x) + 0.3 * math.sin(1.8 * x + 1.0)


class Robot:
    """Simple differential-drive style robot. Chassis parameters affect dynamics.

    friction in (0, 1]: forward speed scalar (low = worn tires / carpet).
    inertia  in [1, inf): yaw command is divided by this (high = heavy chassis).
    noise    sigma on yaw-rate process noise.
    """

    def __init__(
        self,
        friction: float = 1.0,
        inertia: float = 1.0,
        noise: float = 0.05,
        rng: Optional[random.Random] = None,
        base_speed: float = 0.8,
    ):
        self.x = 0.0
        self.y = line_y(0.0)
        self.theta = 0.0
        self.v = base_speed
        self.friction = friction
        self.inertia = inertia
        self.noise = noise
        self.rng = rng if rng is not None else random.Random()

    def step(self, omega_cmd: float, dt: float = 0.05, speed_scale: float = 1.0) -> None:
        """Advance dynamics by dt given commanded yaw rate and optional speed scale."""
        omega_effective = omega_cmd / self.inertia
        self.theta += omega_effective * dt + self.rng.gauss(0.0, self.noise) * dt
        speed = self.v * speed_scale
        vx = speed * math.cos(self.theta) * self.friction
        vy = speed * math.sin(self.theta) * self.friction
        self.x += vx * dt
        self.y += vy * dt

    def sense_line_error(self, sensor_noise: float = 0.02) -> float:
        """Sensor: signed distance from line at current x. Adds measurement noise."""
        target = line_y(self.x)
        return (target - self.y) + self.rng.gauss(0.0, sensor_noise)

    def context(self):
        """Return a 4-dim context vector: [1, friction, inertia, noise]."""
        return [1.0, self.friction, self.inertia, self.noise]

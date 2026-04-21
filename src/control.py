"""PID controller. Stateful. Reset between episodes."""


class PID:
    """Textbook PID. D term is on the error, with a fixed dt."""

    def __init__(self, kp: float = 2.0, ki: float = 0.0, kd: float = 0.5):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_err = 0.0
        self.integ = 0.0

    def reset(self) -> None:
        self.prev_err = 0.0
        self.integ = 0.0

    def step(self, err: float, dt: float = 0.05) -> float:
        self.integ += err * dt
        d = (err - self.prev_err) / dt
        self.prev_err = err
        return self.kp * err + self.ki * self.integ + self.kd * d

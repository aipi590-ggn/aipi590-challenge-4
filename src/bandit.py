"""Contextual bandits over a discrete set of PID gain arms.

Implements:
- LinUCB (Li et al. 2010, WWW)
- EpsilonGreedy over the same arm space
- NeuralBandit: a tiny MLP per arm with dropout-based Thompson sampling

The arm space is a hand-set Cartesian product of (kp, kd) values. Students can
expand it by editing `discretize_arms`.
"""
import math
import random
from typing import List, Optional, Tuple

import numpy as np


def discretize_arms() -> List[Tuple[float, float]]:
    """Arms = discretized (kp, kd) pairs. 20 arms total."""
    arms = []
    for kp in [0.5, 1.0, 2.0, 3.5, 5.0]:
        for kd in [0.0, 0.3, 0.8, 1.5]:
            arms.append((kp, kd))
    return arms


class LinUCB:
    """Disjoint LinUCB (Li et al. 2010).

    One ridge-regression model per arm on the same context vector.
    Upper confidence bound: theta_a^T x + alpha * sqrt(x^T A_a^{-1} x).
    """

    name = "LinUCB"

    def __init__(self, n_arms: int, d_context: int, alpha: float = 0.6):
        self.n_arms = n_arms
        self.d = d_context
        self.alpha = alpha
        self.A = [np.eye(d_context) for _ in range(n_arms)]
        self.b = [np.zeros(d_context) for _ in range(n_arms)]

    def select(self, x: np.ndarray) -> int:
        best, best_p = 0, -1e9
        for a in range(self.n_arms):
            Ainv = np.linalg.inv(self.A[a])
            theta = Ainv @ self.b[a]
            p = float(theta @ x + self.alpha * math.sqrt(max(0.0, x @ Ainv @ x)))
            if p > best_p:
                best_p, best = p, a
        return best

    def update(self, a: int, x: np.ndarray, r: float) -> None:
        self.A[a] += np.outer(x, x)
        self.b[a] += r * x


class EpsilonGreedy:
    """Epsilon-greedy over arm-conditional running means. Context-blind baseline."""

    name = "EpsilonGreedy"

    def __init__(self, n_arms: int, d_context: int, epsilon: float = 0.1,
                 rng: Optional[random.Random] = None):
        self.n_arms = n_arms
        self.d = d_context
        self.epsilon = epsilon
        self.counts = np.zeros(n_arms)
        self.sums = np.zeros(n_arms)
        self.rng = rng if rng is not None else random.Random()

    def select(self, x: np.ndarray) -> int:
        if self.rng.random() < self.epsilon or np.all(self.counts == 0):
            return self.rng.randrange(self.n_arms)
        means = np.where(self.counts > 0, self.sums / np.maximum(self.counts, 1), -1e9)
        return int(np.argmax(means))

    def update(self, a: int, x: np.ndarray, r: float) -> None:
        self.counts[a] += 1
        self.sums[a] += r


# ─── Neural bandit ────────────────────────────────────────────────────────────

def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


class _TinyMLP:
    """Tiny 1-hidden-layer MLP trained by plain SGD. Dropout at inference
    serves as a cheap stand-in for a Bayesian posterior (Gal and Ghahramani, 2016)."""

    def __init__(self, d_in: int, hidden: int = 16, lr: float = 0.05,
                 dropout: float = 0.2, rng: Optional[np.random.RandomState] = None):
        self.d_in = d_in
        self.hidden = hidden
        self.lr = lr
        self.dropout = dropout
        rs = rng if rng is not None else np.random.RandomState()
        self.W1 = rs.normal(0.0, 0.3, size=(d_in, hidden))
        self.b1 = np.zeros(hidden)
        self.W2 = rs.normal(0.0, 0.3, size=(hidden,))
        self.b2 = 0.0
        self.rs = rs

    def _forward(self, x, apply_dropout=False):
        z1 = x @ self.W1 + self.b1
        h = np.tanh(z1)
        if apply_dropout and self.dropout > 0.0:
            mask = (self.rs.rand(self.hidden) > self.dropout).astype(float)
            h = h * mask / (1.0 - self.dropout)
        else:
            mask = np.ones(self.hidden)
        y = h @ self.W2 + self.b2
        return y, h, mask, z1

    def predict(self, x, apply_dropout=False):
        y, _, _, _ = self._forward(x, apply_dropout=apply_dropout)
        return float(y)

    def train_step(self, x, target):
        y, h, mask, z1 = self._forward(x, apply_dropout=False)
        # MSE gradient
        dy = 2.0 * (y - target)
        dW2 = dy * h
        db2 = dy
        dh = dy * self.W2
        dz1 = dh * (1.0 - np.tanh(z1) ** 2)
        dW1 = np.outer(x, dz1)
        db1 = dz1
        # SGD
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1


class NeuralBandit:
    """One tiny MLP per arm. Thompson sampling via dropout noise at selection time.

    Keeps a small replay buffer per arm and does a few SGD steps per update.
    Not a true Bayesian posterior, but the dropout trick gives a cheap,
    honest uncertainty signal for exploration.
    """

    name = "NeuralBandit"

    def __init__(self, n_arms: int, d_context: int, hidden: int = 16,
                 lr: float = 0.05, dropout: float = 0.2,
                 replay_size: int = 64, train_steps: int = 4,
                 seed: int = 0):
        self.n_arms = n_arms
        self.d = d_context
        self.replay_size = replay_size
        self.train_steps = train_steps
        rs = np.random.RandomState(seed)
        self.nets = [
            _TinyMLP(d_context, hidden=hidden, lr=lr, dropout=dropout,
                     rng=np.random.RandomState(seed + a + 1))
            for a in range(n_arms)
        ]
        self.replay: List[List[Tuple[np.ndarray, float]]] = [[] for _ in range(n_arms)]
        self.rs = rs
        self.pulls = np.zeros(n_arms, dtype=int)

    def select(self, x: np.ndarray) -> int:
        scores = np.zeros(self.n_arms)
        for a in range(self.n_arms):
            if self.pulls[a] == 0:
                # Force-explore untouched arms first
                return a
            scores[a] = self.nets[a].predict(x, apply_dropout=True)
        return int(np.argmax(scores))

    def update(self, a: int, x: np.ndarray, r: float) -> None:
        buf = self.replay[a]
        buf.append((np.array(x, dtype=float), float(r)))
        if len(buf) > self.replay_size:
            del buf[0]
        self.pulls[a] += 1
        # A few SGD steps over the replay buffer
        for _ in range(self.train_steps):
            idx = self.rs.randint(0, len(buf))
            xi, ri = buf[idx]
            self.nets[a].train_step(xi, ri)

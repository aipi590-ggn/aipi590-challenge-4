"""Main experiments.

Trains three adaptive policies (epsilon-greedy, LinUCB, neural bandit) online over
a distribution of chassis parameters, then evaluates all policies (including a
fixed-gain baseline and a per-chassis oracle) on a common holdout of 30 chassis.

Outputs
-------
results/summary.json          aggregate stats per policy
results/runs.csv              per-holdout-episode rows
public/learning_curve.png     online learning curves (consumed by dashboard)
public/holdout.png            holdout bar chart (consumed by dashboard)
public/dashboard.json         pre-computed traces for the dashboard

Reproducible: fixed seed per run, 5 seeds aggregated.
"""
from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.bandit import EpsilonGreedy, LinUCB, NeuralBandit, discretize_arms
from src.control import PID
from src.eval import (
    build_robot,
    reward_aligned,
    run_episode,
    sample_chassis,
)
from src.world import Robot, line_y

RESULTS = ROOT / "results"
PUBLIC = ROOT / "public"
RESULTS.mkdir(exist_ok=True)
PUBLIC.mkdir(exist_ok=True)


# ─── Oracle: best arm per chassis under the aligned reward ────────────────────

def oracle_best_arm(friction, inertia, noise, arms, rng_seed: int = 0):
    """Grid-search the arm space once per chassis. Used for the regret baseline."""
    best_r, best_a = -1e9, 0
    for a_idx, (kp, kd) in enumerate(arms):
        # 3 micro-rollouts for a stable estimate
        rs = []
        for k in range(3):
            rng = random.Random(rng_seed * 997 + k * 7 + a_idx)
            robot = build_robot(friction, inertia, noise, rng)
            pid = PID(kp=kp, kd=kd)
            trace = run_episode(robot, pid, steps=300)
            r, _ = reward_aligned(trace)
            rs.append(r)
        mean_r = float(np.mean(rs))
        if mean_r > best_r:
            best_r, best_a = mean_r, a_idx
    return best_a, best_r


def fixed_policy(kp: float = 2.0, kd: float = 0.5):
    def _p(_robot):
        return (kp, kd)
    return _p


# ─── Online training for a single bandit ──────────────────────────────────────

def train_bandit(bandit, arms, episodes: int, seed: int):
    rng = random.Random(seed)
    history = []
    for ep in range(episodes):
        f, inr, nz = sample_chassis(rng)
        robot = build_robot(f, inr, nz, rng)
        ctx = np.array(robot.context(), dtype=float)
        a = bandit.select(ctx)
        kp, kd = arms[a]
        pid = PID(kp=kp, kd=kd)
        trace = run_episode(robot, pid, steps=300)
        r, v = reward_aligned(trace)
        bandit.update(a, ctx, r)
        history.append({
            "ep": ep, "friction": f, "inertia": inr, "noise": nz,
            "arm": a, "kp": kp, "kd": kd, "r": r, "violation": v,
        })
    return history


# ─── Full experiment ──────────────────────────────────────────────────────────

def run(seeds=(0, 1, 2, 3, 4), train_episodes: int = 150, holdout_n: int = 30):
    arms = discretize_arms()
    d_ctx = 4

    # Build holdout chassis once per seed so all policies see identical robots
    per_seed_results = []
    all_learning_curves = {"EpsilonGreedy": [], "LinUCB": [], "NeuralBandit": []}

    for seed in seeds:
        train_rng_seed = 10_000 + seed
        holdout_rng_seed = 20_000 + seed

        # Train each adaptive policy on its own RNG stream
        eps = EpsilonGreedy(len(arms), d_ctx, epsilon=0.1,
                            rng=random.Random(train_rng_seed + 1))
        linucb = LinUCB(len(arms), d_ctx, alpha=0.6)
        neural = NeuralBandit(len(arms), d_ctx, seed=train_rng_seed + 3)

        h_eps = train_bandit(eps, arms, train_episodes, seed=train_rng_seed + 10)
        h_lin = train_bandit(linucb, arms, train_episodes, seed=train_rng_seed + 20)
        h_neu = train_bandit(neural, arms, train_episodes, seed=train_rng_seed + 30)

        all_learning_curves["EpsilonGreedy"].append([h["r"] for h in h_eps])
        all_learning_curves["LinUCB"].append([h["r"] for h in h_lin])
        all_learning_curves["NeuralBandit"].append([h["r"] for h in h_neu])

        # Build a fixed holdout set of (friction, inertia, noise)
        hrng = random.Random(holdout_rng_seed)
        chassis = [sample_chassis(hrng) for _ in range(holdout_n)]

        def _greedy_bandit_policy(bandit):
            def _p(robot):
                ctx = np.array(robot.context(), dtype=float)
                a = bandit.select(ctx)
                return arms[a]
            return _p

        def _oracle_policy(oracle_arms_, chassis_):
            lookup = {ch: a for ch, a in zip(chassis_, oracle_arms_)}
            def _p(robot):
                key = (robot.friction, robot.inertia, robot.noise)
                return lookup.get(key, (2.0, 0.5))
            return _p

        # Oracle: per-chassis best arm (grid search over all arms)
        oracle_arms = []
        for (f, inr, nz) in chassis:
            a_best, _ = oracle_best_arm(f, inr, nz, arms, rng_seed=seed * 131 + hash((f, inr, nz)) % 1000)
            oracle_arms.append(arms[a_best])

        policies = {
            "Fixed":         fixed_policy(kp=2.0, kd=0.5),
            "EpsilonGreedy": _greedy_bandit_policy(eps),
            "LinUCB":        _greedy_bandit_policy(linucb),
            "NeuralBandit":  _greedy_bandit_policy(neural),
            "Oracle":        _oracle_policy(oracle_arms, chassis),
        }

        seed_result = {"seed": seed, "policies": {}, "chassis": chassis}

        for pname, policy in policies.items():
            # Evaluate on the fixed holdout
            rewards, viols = [], []
            for (f, inr, nz) in chassis:
                eval_rng = random.Random(hash((seed, pname, f, inr, nz)) & 0xFFFFFFFF)
                robot = build_robot(f, inr, nz, eval_rng)
                kp, kd = policy(robot)
                pid = PID(kp=kp, kd=kd)
                trace = run_episode(robot, pid, steps=300)
                r, v = reward_aligned(trace)
                rewards.append(r)
                viols.append(v)
            seed_result["policies"][pname] = {
                "rewards": rewards, "violations": viols,
                "mean_reward": float(np.mean(rewards)),
                "violation_rate": float(np.mean(viols)),
                "violation_count": int(sum(viols)),
            }

        per_seed_results.append(seed_result)

    # Aggregate across seeds
    policy_names = ["Fixed", "EpsilonGreedy", "LinUCB", "NeuralBandit", "Oracle"]
    summary = {"seeds": list(seeds), "train_episodes": train_episodes,
               "holdout_n": holdout_n, "n_arms": len(arms), "policies": {}}
    for p in policy_names:
        mean_rs = [s["policies"][p]["mean_reward"] for s in per_seed_results]
        viol_rates = [s["policies"][p]["violation_rate"] for s in per_seed_results]
        viol_counts = [s["policies"][p]["violation_count"] for s in per_seed_results]
        summary["policies"][p] = {
            "mean_reward_per_seed": mean_rs,
            "mean_reward_across_seeds": float(np.mean(mean_rs)),
            "std_reward_across_seeds": float(np.std(mean_rs)),
            "violation_rate_per_seed": viol_rates,
            "violation_rate_across_seeds": float(np.mean(viol_rates)),
            "violation_count_per_seed": viol_counts,
            "total_violations": int(sum(viol_counts)),
            "total_holdout_episodes": int(len(mean_rs) * holdout_n),
        }

    # Write summary
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2))

    # Write per-episode CSV
    with open(RESULTS / "runs.csv", "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["seed", "policy", "ep_idx", "friction", "inertia", "noise",
                    "reward", "violation"])
        for s in per_seed_results:
            chassis = s["chassis"]
            for p in policy_names:
                pr = s["policies"][p]
                for i, (f, inr, nz) in enumerate(chassis):
                    w.writerow([s["seed"], p, i, f, inr, nz,
                                pr["rewards"][i], pr["violations"][i]])

    # ─── Plots ────────────────────────────────────────────────────────────────
    DUKE_NAVY = "#012169"
    ROYAL = "#00539B"
    SHALE = "#0577B1"
    COPPER = "#C84E00"
    PERSIMMON = "#E89923"
    GRAPHITE = "#666666"

    # Learning curves: mean across seeds with min/max band
    fig, ax = plt.subplots(figsize=(9, 4.5))
    window = 10
    colors = {"EpsilonGreedy": COPPER, "LinUCB": DUKE_NAVY, "NeuralBandit": PERSIMMON}
    for name, curves in all_learning_curves.items():
        arr = np.array(curves)  # (seeds, episodes)
        mean = arr.mean(axis=0)
        lo = arr.min(axis=0)
        hi = arr.max(axis=0)
        # Smooth mean with moving average
        if len(mean) >= window:
            sm = np.convolve(mean, np.ones(window) / window, mode="valid")
            xs = np.arange(window - 1, len(mean))
        else:
            sm, xs = mean, np.arange(len(mean))
        ax.fill_between(range(len(mean)), lo, hi, color=colors[name], alpha=0.12)
        ax.plot(xs, sm, color=colors[name], linewidth=1.8, label=name)
    ax.set_xlabel("training episode")
    ax.set_ylabel("reward per episode")
    ax.set_title("online learning curves (mean across 5 seeds, min/max band)")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", frameon=False)
    plt.tight_layout()
    plt.savefig(PUBLIC / "learning_curve.png", dpi=130)
    plt.close(fig)

    # Holdout bar chart: mean reward + violation counts
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    pol_order = ["Fixed", "EpsilonGreedy", "LinUCB", "NeuralBandit", "Oracle"]
    mean_rs = [summary["policies"][p]["mean_reward_across_seeds"] for p in pol_order]
    std_rs = [summary["policies"][p]["std_reward_across_seeds"] for p in pol_order]
    total_viols = [summary["policies"][p]["total_violations"] for p in pol_order]
    total_eps = [summary["policies"][p]["total_holdout_episodes"] for p in pol_order]
    bar_colors = [COPPER, PERSIMMON, DUKE_NAVY, ROYAL, GRAPHITE]

    axes[0].bar(pol_order, mean_rs, yerr=std_rs, color=bar_colors, capsize=4)
    axes[0].set_title("mean holdout reward (5 seeds × 30 chassis)")
    axes[0].set_ylabel("reward")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].axhline(0, color=GRAPHITE, linewidth=0.6)

    pct = [100.0 * v / max(1, t) for v, t in zip(total_viols, total_eps)]
    axes[1].bar(pol_order, pct, color=bar_colors)
    axes[1].set_title("off-line violation rate (%)")
    axes[1].set_ylabel("% episodes with any |err| > 0.5")
    axes[1].grid(axis="y", alpha=0.25)
    for i, p in enumerate(pct):
        axes[1].text(i, p + 1, f"{p:.0f}%", ha="center", fontsize=9, color=GRAPHITE)

    for ax in axes:
        for label in ax.get_xticklabels():
            label.set_rotation(15)
    plt.tight_layout()
    plt.savefig(PUBLIC / "holdout.png", dpi=130)
    plt.close(fig)

    # ─── Dashboard pre-computed traces ────────────────────────────────────────
    # Build a small set of animated trajectories for the dashboard. For each
    # policy on two canonical chassis (normal + swapped), store xs, ys, target,
    # per-step reward snapshots, cumulative reward.
    dash = {"policies": {}, "line": [], "x_range": [0.0, 15.0]}
    xs_line = np.linspace(0.0, 15.0, 200)
    dash["line"] = [{"x": float(x), "y": float(line_y(x))} for x in xs_line]

    canonical = {
        "normal":  (1.0, 1.0, 0.06),
        "swapped": (0.4, 2.0, 0.08),
    }

    # Pick the first-seed trained bandits for the dashboard
    dash_arms = arms
    seed0 = 10_000 + seeds[0]
    dash_eps = EpsilonGreedy(len(arms), d_ctx, epsilon=0.1, rng=random.Random(seed0 + 1))
    dash_lin = LinUCB(len(arms), d_ctx, alpha=0.6)
    dash_neu = NeuralBandit(len(arms), d_ctx, seed=seed0 + 3)
    train_bandit(dash_eps, dash_arms, train_episodes, seed=seed0 + 10)
    train_bandit(dash_lin, dash_arms, train_episodes, seed=seed0 + 20)
    train_bandit(dash_neu, dash_arms, train_episodes, seed=seed0 + 30)

    dash_policies = {
        "Fixed":         ("fixed", None),
        "EpsilonGreedy": ("bandit", dash_eps),
        "LinUCB":        ("bandit", dash_lin),
        "NeuralBandit":  ("bandit", dash_neu),
    }
    for pname, (kind, bandit) in dash_policies.items():
        per_chassis = {}
        for cname, (f, inr, nz) in canonical.items():
            rng = random.Random(hash((pname, cname)) & 0xFFFFFFFF)
            robot = build_robot(f, inr, nz, rng)
            if kind == "fixed":
                kp, kd = 2.0, 0.5
            else:
                ctx = np.array(robot.context(), dtype=float)
                a = bandit.select(ctx)
                kp, kd = dash_arms[a]
            pid = PID(kp=kp, kd=kd)
            trace = run_episode(robot, pid, steps=300)
            r, v = reward_aligned(trace)
            per_chassis[cname] = {
                "friction": f, "inertia": inr, "noise": nz,
                "kp": kp, "kd": kd,
                "xs": [float(x) for x in trace["xs"]],
                "ys": [float(y) for y in trace["ys"]],
                "errs": [float(e) for e in trace["errs"]],
                "omegas": [float(o) for o in trace["omegas"]],
                "reward": r, "violation": v,
            }
        dash["policies"][pname] = per_chassis

    # Cumulative-regret learning curve (LinUCB vs oracle approximate)
    dash["learning_curves"] = {
        name: {"mean": np.array(curves).mean(axis=0).tolist(),
               "min":  np.array(curves).min(axis=0).tolist(),
               "max":  np.array(curves).max(axis=0).tolist()}
        for name, curves in all_learning_curves.items()
    }
    # Holdout summary copied in for convenience
    dash["holdout"] = {p: {
        "mean_reward": summary["policies"][p]["mean_reward_across_seeds"],
        "violation_rate": summary["policies"][p]["violation_rate_across_seeds"],
    } for p in pol_order}

    (PUBLIC / "dashboard.json").write_text(json.dumps(dash))

    # Console summary
    print("\n=== holdout summary (5 seeds x 30 chassis) ===")
    for p in pol_order:
        s = summary["policies"][p]
        print(f"  {p:16s}  reward={s['mean_reward_across_seeds']:+.3f} "
              f"(std {s['std_reward_across_seeds']:.3f})  "
              f"viol_rate={100 * s['violation_rate_across_seeds']:5.1f}%  "
              f"total_viols={s['total_violations']}/{s['total_holdout_episodes']}")
    print(f"\nArtifacts: {RESULTS}/summary.json  runs.csv  |  {PUBLIC}/learning_curve.png  holdout.png  dashboard.json")


if __name__ == "__main__":
    run()

# AIPI 590 · Challenge 4: Line-Follow PID Tuner via Contextual Bandit

[![Live Dashboard](https://img.shields.io/badge/dashboard-live-2ca02c?style=flat)](https://aipi590-ggn.github.io/aipi590-challenge-4/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg?style=flat)](https://www.python.org/)
[![LinUCB](https://img.shields.io/badge/algorithm-LinUCB-00539B?style=flat)](https://rob.schapire.net/papers/www10.pdf)

A contextual bandit that re-tunes PID gains for a line-following robot when its
chassis changes. Framed as the live-tuning layer of a larger robotics-education
stack: students swap motors, replace tires, change the chassis. The gains that
worked yesterday no longer fit. Rather than re-tune by hand, a bandit picks
`(kp, kd)` from a discrete menu based on a short feature vector describing the
chassis.

**[Live dashboard](https://aipi590-ggn.github.io/aipi590-challenge-4/)** ·
**[Slides](presentation.md)** ·
**[Scope](SCOPE.md)** ·
**[Rubric checklist](REQUIREMENTS_CHECKLIST.md)**

---

## Headline numbers

Holdout of 5 seeds x 30 randomly-sampled chassis, each evaluated under the
aligned reward `r = -MAE - 0.05 * mean(|omega|)`. "Violation" is any episode
where the robot strays more than 0.5 units off the target line.

| policy          | mean reward | violation rate | notes |
|-----------------|-------------|----------------|-------|
| Fixed PID       | -0.272      | 76.0%          | static gains, no adaptation |
| Epsilon-greedy  | -0.181      | 20.7%          | context-blind baseline |
| LinUCB          | -0.199      | 37.3%          | linear bandit, 20 arms |
| Neural bandit   | -0.226      | 42.7%          | tiny MLP per arm + dropout Thompson sampling |
| Oracle          | -0.151      | 20.7%          | per-chassis grid search (upper bound) |

The win is decisive against fixed PID: violation rate drops from 76% to 21-43%,
and mean reward improves by 0.04-0.09 across all three bandits. Epsilon-greedy
edges out LinUCB on this task, which is honest reporting: with only 4 context
features and a modest arm space, the gap between the two is narrow and the
simpler algorithm wins on sample efficiency. LinUCB still gives you calibrated
uncertainty, which is what matters if you later want to bolt a safety wrapper
on top. Even the oracle violates 21% of the time, because the aligned reward
trades off MAE and motion while "violation" is a hard threshold on any-step
max error. That gap between "best mean reward" and "no worst-case excursions"
is a useful thing to show students.

## Why a bandit, not PPO

Line following is a *one-step* decision per chassis. Pick `(kp, kd)`, run the
episode, observe the reward. There is no credit-assignment problem across time
steps that a full RL method needs to solve. The student is not trying to teach
the robot to plan. They are trying to answer: "given what this chassis feels
like, which PID gains should I use?" That question is a contextual bandit by
definition (Li et al. 2010). PPO would work, but it is the wrong shape: more
hyperparameters, more samples, and a harder story to tell a beginner.

Bandits are also the right fit for classroom deployment. The state space is
tiny (4 features), the arm space is discrete (20 arms), and the reward signal
is available after a single short episode. LinUCB converges in ~80 episodes
and gives you confidence bounds that an instructor can read directly off the
arm statistics.

## Two experiments

| # | Question | Result | Artifact |
|---|---|---|---|
| main  | Can a contextual bandit beat fixed PID across a distribution of chassis? | Yes. Violation rate drops from 75% to 21-42% across three bandit algorithms. | [run_experiments.py](scripts/run_experiments.py) · [plot](results/holdout.png) |
| align | Does naive `-MAE` reward invite reward hacking? | Yes. Adding a forward-speed knob to the arm space causes LinUCB to pick slow speeds and idle near the start line. Mean final x drops from 6.13 to 1.69. | [run_alignment.py](scripts/run_alignment.py) · [plot](results/alignment.png) |

## Project structure

```
aipi590-challenge-4/
├── README.md                    # this file
├── SCOPE.md                     # plan + locked decisions
├── REQUIREMENTS_CHECKLIST.md    # rubric tracker
├── presentation.md              # 6-slide deck
├── src/
│   ├── world.py                 # target line, Robot class
│   ├── control.py               # PID controller
│   ├── bandit.py                # LinUCB, EpsilonGreedy, NeuralBandit, arm space
│   └── eval.py                  # run_episode, reward functions, holdout harness
├── scripts/
│   ├── run_experiments.py       # main: 4 policies + oracle, 5 seeds, holdout n=30
│   └── run_alignment.py         # reward-hacking demo under -MAE reward
├── results/                     # summary.json, runs.csv, plots, dashboard.json
├── public/                      # static dashboard (index.html + json)
└── docs -> public               # symlink; Pages source path is /docs
```

## Quickstart

```bash
# 1. Setup (stdlib + numpy + matplotlib)
python3 -m venv .venv && source .venv/bin/activate
pip install numpy matplotlib

# 2. Run the experiments
python3 scripts/run_experiments.py    # ~90 seconds on M1
python3 scripts/run_alignment.py      # ~15 seconds

# 3. View the dashboard locally
python3 -m http.server -d public 8080
# open http://localhost:8080
```

Artifacts are written to `results/` and copied into `public/` for Pages.

## Key literature

- **Li et al. 2010.** [A Contextual-Bandit Approach to Personalized News
  Article Recommendation](https://rob.schapire.net/papers/www10.pdf) (WWW).
  Canonical LinUCB paper.
- **Dogru, Lopez-Ulloa, and Sanchez-Lopez 2021.** [Reinforcement Learning
  Approach to Autonomous PID Tuning](https://ieeexplore.ieee.org/document/9438840)
  (ICUAS). Adaptive PID tuning via RL in the control literature. Uses full RL
  rather than bandits, making it the natural comparison for "why not PPO here."
- **Gal and Ghahramani 2016.** [Dropout as a Bayesian Approximation: Representing
  Model Uncertainty in Deep Learning](https://proceedings.mlr.press/v48/gal16.html)
  (ICML). Justification for the dropout-Thompson-sampling trick in the neural
  bandit.
- **Amodei et al. 2016.** [Concrete Problems in AI Safety](https://arxiv.org/abs/1606.06565)
  (arXiv). The taxonomy the reward-hacking demo sits in: reward hacking,
  distributional shift, negative side effects.

## Cross-project context

This repository is one layer of a broader robotics-education stack. The
contextual-bandit tuner ships as a library that can be dropped into any
controller with a discrete hyperparameter menu and a per-episode reward. A
sibling project, [aipi540-tabletop-perception](https://github.com/jonasneves/aipi540-tabletop-perception),
provides the perception layer (detecting where the target line actually is
from a top-down camera). The same `LinUCB` object used here is intended to
tune visual-servoing parameters in that stack.

## Team

Jonas Neves

Duke University · AIPI 590 · Spring 2026

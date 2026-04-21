# Challenge 4: Line-Follow PID Tuner via Contextual Bandit
## A live-tuning layer for beginner robotics

**Team:** Lindsay Gross · Yifei Guo · Jonas Neves

---

### Slide 1: The Problem

**On slide:**

- Beginner robotics class. Students build line-following robots.
- Week 1: PID gains tuned by hand. Works great.
- Week 3: Someone swaps a motor or replaces the wheels.
- Same PID gains now oscillate, overshoot, or fall off the line.
- Instructor's time goes to re-tuning instead of teaching.

> **[IMAGE, AI-generated]**
> Prompt: "Top-down view of a small two-wheeled educational robot on a dark surface, following a curved white line, clean minimal flat illustration style, dark background, no text"

**Speaker notes:**

Beginner robotics lab. First project is almost always a **line-following robot** — print a chassis, bolt on two motors, write a PID loop. **Week 1**: students tune by hand. Works great.

**Week 3**: someone swaps a motor. Someone else grabs grippier tires. Someone adds a payload. Every change shifts the dynamics and **the gains that worked Monday are now wrong**. The one-hour re-tune eats half a class. The instructor tunes for them. Students learn PID is fragile, not how control works.

Narrow problem — **but it repeats in thousands of classrooms**. That's the use case.

---

### Slide 2: The MDP We're Working In

**On slide:**

- **State** `s ∈ ℝ⁴`: `[1, friction, inertia, noise]` — the chassis features.
- **Action** `a`: one of 20 discretized `(kp, kd)` pairs.
- **Reward** `r = -MAE - 0.05·mean(|ω|)` over one ~15s episode.
- **Transition**: none. Episode ends after one action.
- **Horizon**: 1. `γ` is moot. Episodic.

> **[CALLOUT]**
> An MDP with horizon 1 and no transition dynamics *is* a contextual bandit.
> That's the formal bridge from the class definition of RL to the tool we use.

**Speaker notes:**

**Week 12 defined RL as an MDP.** State, action, reward, transition, discount, horizon. Here's ours, stated plainly so nothing is hand-waved.

- **State** — 4-D chassis vector; leading 1 is a bias term for clean linear models.
- **Action** — one of 20 discretized `(kp, kd)` pairs.
- **Reward** — MAE plus a small motion-cost term, over one 15-second episode.
- **Transition** — trivial. Pick gains, run, observe one scalar. **No next state to plan into.**
- **Horizon** — 1. γ has nothing to discount. Episodic/continuing collapses.

**A horizon-1 MDP with no transitions is a contextual bandit.** That's the formal connection. We're not sidestepping the MDP formulation — we're stating it and noting that **most of the RL machinery the lecture covered has nothing to do here** (value functions, credit assignment, discount, policy gradients). Next slide is the engineering consequence.

---

### Slide 3: Why a Bandit, Not PPO

**On slide:**

- Decision shape: **one action per chassis, one reward per episode**.
- No time-credit-assignment problem.
- No planning horizon.
- Full RL (PPO, SAC) fits, but is the wrong tool.
- Contextual bandits (Li et al. 2010) are the minimum tool that fits.

> **[DIAGRAM, make in slides]**
> Left box: "Full RL (PPO). Needs trajectory credit assignment, hundreds of thousands of steps, reward shaping."
> Middle arrow labeled "decision shape".
> Right box: "Contextual bandit (LinUCB). One action per context, calibrated uncertainty, converges in ~80 episodes."

**Speaker notes:**

Reaching for **PPO or SAC** is tempting — that's what "RL" means in textbooks. But look at the **decision shape**: student sets `kp, kd` once, robot runs 15 seconds, one scalar reward. **No credit assignment across steps. No planning.** Single-shot decision conditional on the chassis.

That's the **contextual-bandit regime** — the horizon-1 MDP from the previous slide. **Li et al. 2010** is the canonical paper. LinUCB gives ridge-regression estimates per arm plus a confidence bound — **calibrated within-model uncertainty for free**. On a well-covered chassis, UCB width tells you "the model has seen this arm × context enough to commit."

PPO would work — but needs **tens of thousands of steps** and careful reward shaping. Wrong tool for a classroom. **The contribution is picking the right shape, not a novel algorithm.**

---

### Slide 4: Simulator and Baselines

**On slide:**

- 2D differential-drive sim. Chassis = (friction, inertia, noise).
- Context vector: `[1, friction, inertia, noise]`. 4 dimensions.
- Arm space: 20 discretized `(kp, kd)` pairs.
- Baselines: fixed PID (status quo), epsilon-greedy, per-chassis oracle (upper bound).

> **[CODE SNIPPET]**
> ```python
> def line_y(x):
>     return 2.0 + 0.8 * math.sin(0.6*x) + 0.3 * math.sin(1.8*x + 1.0)
>
> # 20 arms = {0.5, 1.0, 2.0, 3.5, 5.0} kp  x  {0.0, 0.3, 0.8, 1.5} kd
> ```

> **[LINK, live dashboard]**
> <https://aipi590-ggn.github.io/aipi590-challenge-4/>

**Speaker notes:**

Sim is **intentionally small** — ~80 lines for world + robot + PID. Target line is a sum of two sinusoids, so it actually bends. Three chassis parameters: **friction** (scales forward speed), **inertia** (divides yaw command), **noise** (on yaw rate). All three are what change when a student swaps hardware.

Arm space: **20 `(kp, kd)` pairs**. `ki` held at 0 — on short curvy episodes, the P and D terms dominate. Bias term in context keeps linear models clean.

Three baselines: **Fixed PID** (what students use today), **ε-greedy** (context-blind adaptive), **per-chassis oracle** (grid search, 3 micro-rollouts per arm — upper bound).

Dashboard reads **pre-computed trajectories** — no inference in the browser, just visualizing what the bandits learned.

---

### Slide 5: Does It Work?

**On slide:**

Holdout: 5 seeds × 30 randomly-sampled chassis.

| policy          | mean reward | violation rate |
|-----------------|-------------|----------------|
| Fixed PID       | **-0.272**  | **79.3%**      |
| Epsilon-greedy  | -0.181      | 20.7%          |
| LinUCB          | -0.199      | 36.7%          |
| Neural bandit   | -0.225      | 42.7%          |
| Oracle (grid)   | -0.151      | 23.3%          |

> **[CHART, from script output]**
> `public/holdout.png`. Paired bar chart: mean reward and violation rate per policy.

> **[CALLOUT]**
> Fixed PID goes off the line on 79% of random chassis. Any bandit cuts that by at least half.

**Speaker notes:**

Core win. Holdout = **150 episodes** (5 seeds × 30 chassis). **Fixed PID violates on 79%** — three of four episodes, the robot strays more than half a unit off the line. Ugly for a demo.

**Any bandit cuts that at least in half.** ε-greedy 21%, LinUCB 37%, neural 43%. Reward numbers match: bandits are 0.04–0.09 better on a –0.15 to –0.30 scale. Oracle upper bound is –0.15, zero violations. **Bandits close 60–80% of the fixed-to-oracle gap.**

**Honest note**: ε-greedy narrowly beats LinUCB. With 4 context features and 20 arms, **the simpler algorithm has the sample-efficiency edge**. LinUCB still gives calibrated confidence bounds — the right pick if you later bolt on a safety wrapper — but on raw reward, know the rank.

---

### Slide 6: Alignment Failure I Caught

**On slide:**

- Add a **forward-speed** knob to the arm space.
- Reward = `-MAE` only (no motion penalty).
- LinUCB learns to **pick the slowest speed and idle**.

> **[CHART, from script output]**
> `public/alignment.png`. Three panels: training curves, selected speed over time, final x bar chart.

| regime                        | mean forward speed | mean x at end | mean `|err|` | violations |
|-------------------------------|-------------------:|--------------:|--------------:|-----------:|
| trained on `-MAE` (hackable)  | 0.26               | **1.69**      | 0.087         | 0/30       |
| trained on `-MAE - travel_penalty` | 0.87          | **6.13**      | 0.203         | 14/30      |

> **[CALLOUT]**
> "Zero violations" does not mean "safe." It means "the robot never moved."

**Speaker notes:**

Third knob on the arm space: **forward speed**, 0.1 to 1.0. Reward set to just **`-MAE`**. No motion penalty. What could go wrong.

LinUCB **converged on the slowest speeds**. Robot barely moves. Mean final x = **1.69 vs. 15-unit target**. On paper it looks optimal: MAE 0.087, zero violations. Except **it never tracked the curve** — found a tiny pocket where standing still looks like tracking.

**Week-13 framing — emergent-goal drift.** The bandit's *incentivized* goal (minimize MAE) is no longer aligned with *human intent* (follow the line to the end). **The idle policy scores perfectly on the proxy and zero on the task we actually wanted.** Amodei 2016 and the CoastRunners boat that spun in circles collecting power-ups sit in the same family.

**Fix**: travel-deficit term. `r = -MAE - 0.4·(target-x_final)/target - 0.01·mean(|ω|)`. Bandit now picks speed 0.87, tracks the curve, accepts 14 violations on hard chassis **because finishing matters**. **"Zero violations" was never the safe column — it was the degenerate one.**

---

### Slide 7: Takeaways

**On slide:**

- Match the tool to the decision shape: one-shot decisions are bandits, not PPO.
- Simpler wins: epsilon-greedy ties the oracle here. Neural bandit does not beat LinUCB.
- Calibrated uncertainty from LinUCB is what a safety wrapper would sit on top of.
- Reward hacking surfaces the moment the reward has a degree of freedom that lets the policy skip the task.

**Speaker notes:**

Three takeaways.

**One: match the tool to the decision shape.** One decision per context, one reward per decision is a contextual bandit — **whether the textbook calls it RL or not**. PPO here would be theater.

**Two: simpler wins.** ε-greedy ties the oracle. Neural bandit doesn't beat LinUCB. With 4 features and 20 arms, **no nonlinear structure for a net to find**. Honest reporting.

**Three: LinUCB still earns its seat** if you need a safety wrapper — calibrated confidence bounds per arm, which ε-greedy can't give you. Narrow gap on this benchmark, but the story matters when an instructor is reading arm stats live.

**Live demo (≈90 seconds):**

1. Click **▶ Tell me the story (auto)**. Four scenes auto-play: bandit normal chassis (works), fixed PID swapped chassis (**drifts off**), bandit swapped chassis (**holds the line**), side-by-side. Narrate one line per scene.
2. Policy → **EpsilonGreedy**, chassis → **Swapped**. Live contrast with the hero GIF. **If asked about the "swap mid-run" button**: it cuts between pre-computed traces; the sim doesn't model intra-episode dynamics.
3. **Holdout table** — point at the **79% → 21% violation drop**. Headline number.
4. **Alignment panel** — hackable: x=1.69, 0 violations. Aligned: x=6.13, 14 violations. **"Zero violations was never the safe column."**

**If time is tight, drop step 2.** The story auto-play is load-bearing; the rest is depth.

---

## Likely questions

**"Why didn't you use a real robot?"**

**The sim is the deliverable.** The argument is about decision shape and alignment failure — a physical robot adds logistics without changing either. Library is chassis-agnostic: `Robot` is swappable for anything that exposes `sense_line_error()`.

**"Why LinUCB and not Thompson sampling?"**

**LinUCB's deterministic UCB** is easier to reason about in a classroom. Thompson sampling needs a posterior, which means assuming a noise model. For the **neural bandit** I did use dropout Thompson sampling (Gal & Ghahramani 2016) — precisely because the closed-form UCB didn't apply.

**"Could the chassis swap happen mid-episode?"**

**Honest answer: not in the current sim.** `run_episode` builds the Robot with fixed chassis params and never re-reads them. The dashboard's "swap chassis mid-run" button **cuts between pre-computed traces** at the same step index — a visualization aid, not a live dynamics event. In the real lab, students swap between sessions anyway, so **one-episode-per-chassis is the right formulation**. Intra-episode swaps would be a non-stationary bandit (exp3, DiscountedLinUCB) — a different algorithm family, explicitly out of scope.

**"Why does the oracle still violate on 21% of episodes?"**

Oracle picks the arm with **best mean reward** — which trades MAE against motion cost. "Violation" is a hard any-step threshold (>0.5 absolute error flips it). So the best-mean arm **can still have tail excursions** on hard chassis (low friction, high inertia). That gap between "best mean" and "no worst-case excursions" is **why alignment is more than reward maximization** — reward and safety metrics measure different things.

**"How would this fail in deployment?"**

**Distribution shift.** A student adds a chassis parameter outside the training distribution — tracked drivetrain, long wheelbase, off-center sensor — and the context vector **no longer describes the physics**. The bandit's extrapolation is unsupported.

**LinUCB's UCB width is a *partial* signal.** It measures in-model epistemic uncertainty over the 20-arm linear reward. A chassis landing in a low-variance direction of `A_a⁻¹` can **look confident while the linear-reward assumption has silently broken**. So UCB catches "I haven't seen this arm × context enough," not "this context is off-manifold." Real deployment needs a density model on the context itself.

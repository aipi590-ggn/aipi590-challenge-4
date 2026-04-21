# Challenge 4: Line-Follow PID Tuner via Contextual Bandit
## A live-tuning layer for beginner robotics

**Speaker:** Jonas Neves

---

### Slide 1: The Problem

**On slide:**

- Beginner robotics class. Students build line-following robots.
- Week 1: PID gains tuned by hand. Works great.
- Week 3: Someone swaps a motor or replaces the wheels.
- Same PID gains now oscillate, overshoot, or fall off the line.
- Instructor's time goes to re-tuning instead of teaching.

> **[IMAGE — AI-generated]**
> Prompt: "Top-down view of a small two-wheeled educational robot on a dark surface, following a curved white line, clean minimal flat illustration style, dark background, no text"

**Speaker notes:**

Picture an AIPI or undergraduate robotics lab. The first hands-on project is
almost always a line-following robot. You print a chassis, bolt on two motors,
stick a line sensor underneath, and write a PID loop. The students tune the
gains by hand in week one. It works great.

Then week three happens. Someone's motor burns out and they swap in a
different brand. Someone else replaces the wheels with a grippier tire for a
different project. Someone adds a payload for a demo day. Every one of those
changes shifts the chassis dynamics, and the gains that worked on Monday are
now wrong. What was supposed to be a one-hour re-tune takes half a class
period. The instructor ends up doing the tuning. The students don't learn
control theory, they just learn that PID is fragile.

This is a narrow problem but it repeats in thousands of classrooms. That's
the use case.

---

### Slide 2: Why a Bandit, Not PPO

**On slide:**

- Decision shape: **one action per chassis, one reward per episode**.
- No time-credit-assignment problem.
- No planning horizon.
- Full RL (PPO, SAC) fits, but is the wrong tool.
- Contextual bandits (Li et al. 2010) are the minimum tool that fits.

> **[DIAGRAM — make in slides]**
> Left box: "Full RL (PPO) — needs trajectory credit assignment, hundreds of thousands of steps, reward shaping"
> Middle arrow labeled "decision shape"
> Right box: "Contextual bandit (LinUCB) — one action per context, calibrated uncertainty, converges in ~80 episodes"

**Speaker notes:**

It's tempting to reach for PPO or SAC because that's what "RL" means in most
textbooks. But look at the decision shape. The student sets kp and kd once.
The robot runs for maybe fifteen seconds. You observe a single reward:
tracking error plus a little motion penalty. That's it. There is no
credit-assignment problem across time steps. There's no planning. It's a
single-shot decision conditional on the chassis you have in front of you.

That's the definition of a contextual bandit. Li et al. 2010 is the canonical
paper. LinUCB gives you a ridge-regression estimate per arm plus a confidence
bound, which means you get calibrated uncertainty for free. That matters for
safety. It means you can read off "the model is not confident about this
chassis yet" and fall back to a safe default.

PPO would work, but it would need tens of thousands of steps, a learned value
function, and careful reward shaping. For a classroom, it's the wrong tool.
Algorithm choice is intentionally classical. The contribution is picking the
right shape for the problem.

---

### Slide 3: Simulator and Baselines

**On slide:**

- 2D differential-drive sim. Chassis = (friction, inertia, noise).
- Context vector: `[1, friction, inertia, noise]` — 4 dimensions.
- Arm space: 20 discretized `(kp, kd)` pairs.
- Baselines: fixed PID (status quo), epsilon-greedy, per-chassis oracle (upper bound).

> **[CODE SNIPPET]**
> ```python
> def line_y(x):
>     return 2.0 + 0.8 * math.sin(0.6*x) + 0.3 * math.sin(1.8*x + 1.0)
>
> # 20 arms = {0.5, 1.0, 2.0, 3.5, 5.0} kp  x  {0.0, 0.3, 0.8, 1.5} kd
> ```

> **[LINK — live dashboard]**
> <https://aipi590-ggn.github.io/aipi590-challenge-4/>

**Speaker notes:**

The sim is intentionally small. About 80 lines for world plus robot plus PID.
The target line is a sum of two sinusoids so it actually bends. The robot is
differential-drive with three chassis parameters: friction, inertia, and
process noise. Friction scales forward speed. Inertia divides the yaw
command. Noise is on the yaw rate. All three are what change when a student
swaps hardware.

The arm space is 20 discretized `(kp, kd)` pairs. I left ki at zero on
purpose. On short curvy episodes it's the derivative and proportional terms
that matter. The bias term in the context vector is just to keep the linear
models clean.

Baselines: fixed PID, what students actually use today. Epsilon-greedy, a
context-blind adaptive baseline. And a per-chassis oracle that grid-searches
the arm space for each chassis and averages three micro-rollouts. That's the
upper bound I'm allowed to compare against.

The dashboard on the slide reads pre-computed trajectories. No inference
happens in the browser. It just visualizes what the bandits learned.

---

### Slide 4: Does It Work?

**On slide:**

Holdout: 5 seeds × 30 randomly-sampled chassis.

| policy          | mean reward | violation rate |
|-----------------|-------------|----------------|
| Fixed PID       | **-0.272**  | **76.0%**      |
| Epsilon-greedy  | -0.181      | 20.7%          |
| LinUCB          | -0.199      | 37.3%          |
| Neural bandit   | -0.226      | 42.7%          |
| Oracle (grid)   | -0.151      | 20.7%          |

> **[CHART — from notebook output]**
> `results/holdout.png` — paired bar chart: mean reward + violation rate per policy.

> **[CALLOUT]**
> Fixed PID goes off the line on 76% of random chassis. Any bandit cuts that by at least half.

**Speaker notes:**

Here's the core win. On a holdout of 150 episodes — five seeds, thirty
chassis each — fixed PID violates the off-line tolerance on 76 percent of
episodes. Three out of four times, the robot strays more than half a unit
from the target line. That's ugly for a demo.

Any bandit cuts that at least in half. Epsilon-greedy is at 21 percent.
LinUCB is at 37. Neural bandit at 43. The reward numbers tell the same story:
bandits are 0.04 to 0.09 better on a roughly -0.15 to -0.30 scale. The oracle
upper bound is -0.15 with zero violations. Bandits are closing 60 to 80
percent of the gap from fixed to oracle.

I want to be honest about one thing on this slide. Epsilon-greedy narrowly
beats LinUCB here. With only four context features and 20 arms, the
simpler algorithm has the sample-efficiency edge. LinUCB still gives
calibrated confidence bounds, which is what you want if you ever need to
bolt a safety wrapper on top. But if you're grading on raw reward, you
should know the rank.

---

### Slide 5: Alignment Failure I Caught

**On slide:**

- Add a **forward-speed** knob to the arm space.
- Reward = `-MAE` only (no motion penalty).
- LinUCB learns to **pick the slowest speed and idle**.

> **[CHART — from notebook output]**
> `results/alignment.png` — three panels: training curves, selected speed over time, final x bar chart.

| regime                        | mean forward speed | mean x at end | mean `|err|` | violations |
|-------------------------------|-------------------:|--------------:|--------------:|-----------:|
| trained on `-MAE` (hackable)  | 0.26               | **1.69**      | 0.087         | 0/30       |
| trained on `-MAE - travel_penalty` | 0.87          | **6.13**      | 0.203         | 14/30      |

> **[CALLOUT]**
> "Zero violations" does not mean "safe." It means "the robot never moved."

**Speaker notes:**

This is the alignment slide. I added a third knob to the arm space: forward
speed, from 0.1 to 1.0. Then I set the reward to just minus mean absolute
error. No motion penalty, no travel bonus. What could go wrong.

LinUCB converged on the slowest speeds. The robot barely moves. It idles
near the start line. Mean final x is 1.69 units versus the 15-unit target. On
paper it looks great: mean absolute error of 0.087 and zero off-line
violations. Optimal!

Except it never tracked the curve. It never did the task. The bandit found a
tiny pocket of reward space where standing still looks like tracking.

This is the line-follow analogue of the OpenAI CoastRunners boat that
learned to spin in circles collecting power-ups instead of finishing the
race (Amodei et al. 2016, Concrete Problems in AI Safety). Same failure
mode, same lesson: if the reward has a degree of freedom that lets you
skip the task, the agent will find it.

The fix is a travel-deficit term. Reward = `-MAE - 0.4 * (target_x -
x_final)/target_x - 0.01 * mean(|omega|)`. With that, the bandit picks
speed 0.87 and actually tracks the curve, accepting 14 violations on the
hard chassis because now finishing the task matters. "Zero violations" was
never the safe column. It was the degenerate one.

---

### Slide 6: Take and Cross-Project Tie

**On slide:**

- Use the right tool: **one-shot decisions are bandits, not PPO**.
- Neural bandit did not beat LinUCB here. Honest reporting.
- The library generalizes: any controller with a discrete hyperparameter menu can drop it in.
- **Sibling project:** [aipi540-tabletop-perception](https://github.com/jonasneves/aipi540-tabletop-perception) is the perception layer. Same `LinUCB` object tunes visual-servoing parameters there.

> **[IMAGE — optional]**
> Two-panel diagram: left panel is a robot on a line labeled "control", right panel is a camera over a workspace labeled "perception". A shared box in the middle labeled "LinUCB tuner" has arrows to both.

**Speaker notes:**

Three takeaways.

One: match the tool to the decision shape. If you're making one decision per
context and observing one reward per decision, that's a contextual bandit
whether your textbook calls it RL or not. Using PPO here would have been
theater.

Two: the neural bandit didn't beat LinUCB, and I'm reporting that honestly.
With a four-dimensional context and 20 arms, there's no nonlinear structure
for a neural net to find. Simpler model wins. That is a fine answer.

Three: this is one layer of a larger robotics-education stack I'm building.
The perception layer is a sibling project, aipi540-tabletop-perception, which
detects where the target line actually is from a top-down camera. The same
LinUCB object can tune the visual-servoing gains on that stack. The bandit
doesn't care that the arms are PID coefficients here. It cares that they're
a discrete menu with a per-episode reward. Drop the library in, define the
arms, define the reward, and the live-tuning layer ships.

Full dashboard with animated robot and the full set of traces is at the URL
on the slide.

---

## Likely questions

**"Why didn't you use a real robot?"**

The sim is the deliverable. The point is the decision-shape argument and the
alignment failure. A physical robot adds logistics without changing the
story. The library is written to be chassis-agnostic — the `Robot` class is
swappable for any object that exposes `sense_line_error()`.

**"Why LinUCB and not Thompson sampling?"**

LinUCB gives a deterministic upper confidence bound, which is easier to
reason about in a classroom. Thompson sampling needs a posterior, which
means assuming a noise model. For the neural bandit I used dropout
Thompson sampling (Gal and Ghahramani 2016) precisely because the closed-form
UCB didn't apply.

**"Could the chassis swap happen mid-episode?"**

The dashboard has a "swap chassis mid-run" button that demonstrates exactly
that. In the real lab, students usually swap between sessions, so the
formulation models one episode per chassis. Sequential intra-episode swaps
would be non-stationary bandits, which is a different algorithm family
(exp3, DiscountedLinUCB) and out of scope.

**"Why does the oracle still violate on 21% of episodes?"**

The oracle picks the arm with the best mean *reward*, which trades off MAE
and motion cost. "Violation" is a hard threshold: any single step with more
than 0.5 absolute error flips it. So the arm with the best average can
still have tail excursions on the hardest chassis (low friction, high
inertia). That gap between "best mean" and "no worst-case excursions" is
a useful illustration of why alignment is more than reward maximization:
the reward and the safety metric measure different things.

**"How would this fail in deployment?"**

Distribution shift is the obvious one. If a student adds a chassis
parameter that wasn't in the training distribution — a tracked drivetrain,
a very long wheelbase, a sensor mounted off-center — the context vector no
longer describes the physics, and the bandit's extrapolation is
unsupported. A production version would need either an out-of-distribution
detector on the context or an escalation path back to fixed gains when
confidence is low. LinUCB's UCB width gives you that signal for free.

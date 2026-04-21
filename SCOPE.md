# Scope · AIPI 590 Challenge 4

## The assignment

Pick an applied use case in a structured domain. Justify RL. Implement a
prototype. Reflect on alignment and safety. Report metrics. Ground in
literature. Present in class, no paper required.

## The use case

Beginner robotics education. Students build line-following robots as a first
project. They swap motors between weeks, change tires, add payloads, change
batteries. Each swap shifts the chassis dynamics. The PID gains tuned by hand
on Monday no longer fit on Wednesday. The instructor's time goes to re-tuning
instead of teaching.

The problem is narrow but real and repeats across thousands of classrooms. It
is a structured domain with a well-defined state representation (chassis
features) and a well-defined action space (PID gains). Impact is measured in
two numbers that any instructor cares about: mean tracking error and the
fraction of episodes where the robot goes off the line hard enough that
demonstration stops working.

## Why RL and specifically a contextual bandit

Supervised learning would need labeled pairs of (chassis features, optimal
gains). Generating that label set requires exactly the hand-tuning we are
trying to avoid. The data is not lying around.

Rules or classical optimization would need a closed-form model of the chassis
dynamics. Students do not have one. The friction and inertia of a 3D-printed
chassis with worn tires is not in a datasheet.

Full RL (PPO, SAC) would work, but it is a poor fit for the problem shape. The
decision is a single action per episode: pick gains, run, observe reward. No
credit assignment across time. No planning. A contextual bandit is the minimum
tool that fits: per-context decision, explore-exploit, delayed noisy reward.

LinUCB (Li et al. 2010) is the canonical algorithm for this shape. It gives
calibrated uncertainty estimates per arm per context, which an instructor can
read off directly. Epsilon-greedy is a common simpler baseline. A small neural
bandit with dropout Thompson sampling is included to test whether nonlinear
structure in the context helps on this task (it does not, and saying so is
honest reporting).

## Decisions locked in

1. **Arm space: 20 (kp, kd) pairs.** kp in {0.5, 1.0, 2.0, 3.5, 5.0}, kd in
   {0.0, 0.3, 0.8, 1.5}. ki held at 0 because the steady-state error term is
   not the failure mode on curvy lines with short episodes.

2. **Context vector: 4 dimensions.** [1, friction, inertia, noise]. The bias
   term is in there so the linear models work out cleanly.

3. **Reward shape: `-MAE - 0.05 * mean(|omega|)`.** MAE is the primary term.
   The motion penalty is the safety term that blocks the reward-hacking
   exploit where the bandit idles the robot to avoid accumulating error.

4. **Holdout protocol: 5 seeds x 30 random chassis.** Each policy sees the
   same 30 chassis per seed so comparisons are paired. Numbers are regenerated
   with `python3 scripts/run_experiments.py`; never hand-copied.

5. **Oracle: per-chassis grid search over all arms with 3 micro-rollouts.**
   Serves as the upper bound and is reported alongside the bandits.

6. **Alignment demo: add a forward-speed knob to the arm space (24 arms total)
   and retrain under `-MAE`.** This turns what was a decision about gains into
   a decision about whether to move at all. LinUCB learns to idle. The fix
   (adding a travel-deficit term) restores the right behavior.

## Things I consciously did not build

- PPO/SAC comparison. The point of the slide is that full RL is the wrong
  shape, not that my PPO implementation lost a race. Reporting a tuned PPO
  number would reward the wrong thing.
- A richer arm space that includes `ki`. Students tune ki by hand rarely
  enough that it is out of scope.
- Sim-to-real transfer. The sim is the deliverable.
- Any dependency beyond numpy and matplotlib. Classroom laptops run Python
  3.9 and I want this to run on day one.

## Success criteria

- Bandit violation rate is meaningfully below fixed-PID violation rate. (Done:
  75% to 21-42%.)
- Alignment demo produces a visible, undeniable reward hack. (Done: hackable
  policy ends at mean x=1.69; aligned policy reaches mean x=6.13.)
- Dashboard runs on GitHub Pages with no build step. (Done: pure HTML, vanilla
  JS, pre-computed JSON.)
- Code is readable enough that an AIPI student could copy `src/bandit.py` into
  their own controller stack. (Intent: readability before cleverness.)

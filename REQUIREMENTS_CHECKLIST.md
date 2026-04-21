# Canvas Rubric Checklist

Maps each element of the Canvas Challenge 4 prompt to where it lives in this
repository.

## Prompt (verbatim)

> Select an applied use case in a structured domain (e.g., healthcare
> treatment policies, fraud detection in fintech, carbon footprint
> reduction). Must justify the use of RL, implement a prototype or
> simulation, and reflect on alignment/safety concerns. Should include
> relevant metrics and literature grounding.

## Checklist

| requirement | satisfied by | evidence |
|---|---|---|
| Applied use case in a structured domain | Beginner robotics education; adaptive PID tuning after chassis swaps | `SCOPE.md#the-use-case`, `README.md#key-decisions` |
| Justification for RL | Decision is per-context, per-episode; reward is delayed and noisy; supervised learning has no label source | `SCOPE.md#why-rl-and-specifically-a-contextual-bandit`, slide 2 of `presentation.md` |
| Prototype or simulation | 2D differential-drive sim + PID + 3 bandit algorithms + oracle baseline | `src/world.py`, `src/control.py`, `src/bandit.py`, `src/eval.py`, `scripts/run_experiments.py` |
| Relevant metrics | Mean tracking error, off-line violation rate, reward per episode, cumulative regret vs oracle | `results/summary.json`, `results/holdout.png`, `results/learning_curve.png`, dashboard |
| Alignment / safety reflection | Reward-hacking demo (bandit learns to idle under `-MAE` reward), fix via travel-deficit term; explicit acknowledgement of distribution shift risk | `scripts/run_alignment.py`, `results/alignment.png`, slide 5 of `presentation.md`, `presentation.md` Q&A (distribution shift), `SCOPE.md#things-i-consciously-did-not-build` (scope boundaries) |
| Literature grounding | Li et al. 2010 (LinUCB origin), Dogru et al. 2021 (RL-PID in control lit), Gal & Ghahramani 2016 (dropout as Bayesian approximation), Amodei et al. 2016 (Concrete Problems in AI Safety) | `README.md#key-literature`, slide 2 and slide 5 of `presentation.md` |

## Deliverable format

| item | location |
|---|---|
| Presentation (6 slides, speaker notes per slide) | [presentation.md](presentation.md) |
| Working simulation | [scripts/run_experiments.py](scripts/run_experiments.py) |
| Live dashboard | <https://aipi590-ggn.github.io/aipi590-challenge-4/> |
| Reproducibility instructions | [README.md#quickstart](README.md) |
| Source code | [src/](src/), [scripts/](scripts/) |
| Raw results | [results/summary.json](results/summary.json), [results/runs.csv](results/runs.csv), [results/alignment.json](results/alignment.json) |

# v12 — Is dynamic expert control necessary here?

**Status:** Pre-registered **development-only design and executable CPU diagnostic**. No v12 empirical results are claimed until the runner has actually completed. This is a planned *minimal falsification experiment* rather than another GRPO run.

## Motivation

v11 showed that GRPO changed its deployed actions, but on four development histories the coordinated controllers matched `global_0` (zero expert displacement) exactly. The relevant unresolved question is **whether different learner states have materially different action values** after controlling for new-task learning progress and actual update count. If not, tuning the controller is unlikely to fix the evidence gap.

## Experiment

Using the *unchanged* v8/v11 CIFAR-100 environment, collect the first active decision of tasks 2–5 from `global_0` rollouts, plus tasks 3–5 from `global_1` rollouts, on six fixed development roots (up to 42 distinct states). The common task-2 state is not duplicated. For every collected state, clone complete learner weights, Adam moments, memory, and RNG state; apply each of six pre-fixed actions for **the same five learner steps**, over **two independently seeded training minibatch streams**. Actions within each stream receive identical new/replay index sequences.

Actions are globally fixed `0`, `0.5`, and `1`, a fixed two-expert-per-layer mask, and two state-dependent two-expert-per-layer masks formed by ranking `(new routing mass − old routing mass)` or the reverse. The mask-based variants have the same number of fully updated experts; no route mask is tuned post hoc. The front-end, heads and Adam moments continue to update when expert displacement is zero.

Independently evaluate the new task and *every previously learned task* on disjoint within-root data drawn only from the **official CIFAR-100 training split**. For each five-step branch report current-task CE reduction (**gain**) and mean positive old-task CE change (**harm**). The locked exploratory utility is `gain − harm`; report its components, not utility alone. This short-window response is **not** end-of-sequence retention.

**Cross-fitting:** stream 0 can select a per-state *hindsight oracle* (privileged, undeployable). Evaluate that action on stream 1. Compare with (a) a globally fixed action and (b) a per-task-stage fixed action, both selected using stream-0 results from **other roots only**. Also evaluate the pre-specified deployable routing-difference action. Average states within each root, then roots; no pseudo-replication. The locked exploratory gate requires adequate learning pressure, an oracle utility difference, no material loss of new-task gain, reduced old-task harm and consistency across ≥4 roots. Meeting this gate is *not* confirmation or proof a learnable controller exists.

## Reproduce

Requires the existing v8 source files and the repository's normal PyTorch/numpy requirements. Run from the repository root:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s experiments/v12_action_value -p 'test_*.py' -v
python experiments/v12_action_value/run_action_value.py --pilot --download --out runs/v12-smoke
python experiments/v12_action_value/run_action_value.py --out runs/v12-development
python experiments/v12_action_value/analyze_action_value.py --run runs/v12-development
```

The pilot downloads the checksum-verified official CIFAR-100 archive if missing, then executes **one** root. The full run reuses that cached archive and executes the six locked roots. Each run must use an output directory that does not exist. CPU single-thread; no GPU, model weights, or outside pretrained LLM required. The runner refuses to write into the frozen `experiments/` tree. The protocol and source hashes are written into each result directory. The official test split is **never** loaded.

Results: `branches.json` contains every paired observation; `analysis/summary.json` stores root-level comparisons, `analysis/report.md` interprets the exploratory gate, and `completion.json` stores work counts. No result in this folder should be called confirmatory without a **new independent** preregistered evaluation.

## What a useful outcome would mean

- **No action-value heterogeneity / failed gate:** state-dependent control is **not demonstrated** in this environment. Diagnose learning pressure and expert-update scale before retraining GRPO; do not call the hypothesis globally false.
- **Cross-stream validated hindsight gap:** a reason to test whether observable features predict good actions on new roots, not proof of adaptive control. The oracle knows stream-0 counterfactual labels no real controller receives.
- **Deployable routing rule improves on held-out roots at matched new-task gain:** better evidence for structured expert decisions, though still a development-only five-step finding and not an RL advantage.

Limits: task-ID-aware fixed raw-image router, stem/head trainable, previously used development class pool, six roots, fixed learning rate/targets, and matched *updates* rather than FLOPs. The v10 pretrained OLMoE extension remains unrun on GPU. See `protocol.json` for the complete frozen design.

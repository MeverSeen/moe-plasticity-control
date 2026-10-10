# MoE Plasticity Control
### Can a learned controller do more than choose a conservative expert-update rule?

**Status (9 October 2026): research-in-progress; not a demonstrated continual-learning improvement.**

With fixed MoE capacity, bounded rehearsal memory, and a finite learning budget, I study whether **the choice of which experts to update** can itself be learned from the state of continual learning. The central claim to test is not that reinforcement learning can alter an optimizer. It is that **state-dependent coordination improves retention at matched new-task learning and cost, compared with strong fixed or scheduled rules**.

A related information question is whether *current degradation of old knowledge* helps a controller beyond the new-task learning curve and remaining budget. These are separate hypotheses, evaluated separately.

## Approach

Treat each adaptation window as a decision point:

1. **Observe:** current-task loss/progress, remaining budget, per-expert old/new routing summaries, recent old-task degradation, and previous actions (the available fields vary by experiment).
2. **Act:** choose the realized parameter-displacement multiplier for an expert (or protected group). In the image experiments, each of two layers has eight experts, with multipliers **0 / 0.5 / 1**.
3. **Update:** run the normal learning step, then scale the expert's **actual Adam parameter displacement**. A zero multiplier **does not freeze the optimizer state**. The shared CNN stem and task-specific head still learn.
4. **Evaluate:** separately record new-task threshold attainment, independent evaluation losses, old-task retention, and physical learner/monitor work. A controller that preserves old tasks by failing to learn the new task is not a success.

The controller is trained by grouped, paired rollouts from matching learner/optimizer/memory states. The latest study compares shared local and layer-coordinated GRPO policies against fixed global multipliers and simple expert-summary rules.

This is a **controlled model-update environment**, not a pretrained language MoE: v1–v7 use synthetic regression with fixed geometric routing; v8/v9/v11 use a small fixed-raw-image-routing, task-ID-aware, two-layer MoE on CIFAR-100. Learned routing, language-model transfer, and full parameter-efficient deployment have **not** been demonstrated.

## What the experiments actually show

| Study | Result | Qualification |
| --- | --- | --- |
| **v7: sequential feedback** | Adding within-segment old-task feedback reduced original-task forgetting from 0.007195 to 0.006205 (**~13.8%**) on 56 jointly successful roots. | Nominal paired interval excluded zero, but Bonferroni sensitivity interval crossed zero; new-task success noninferiority was not established. Only the original task A was protected, not all prior B tasks. |
| **v8: real-image pilot** | Both GRPO controllers selected their initialization; deterministic actions matched global full update. Zero expert displacement preserved more but required **25% more learner updates**. | Three audit roots share one class pool. No established learned-controller gain. |
| **v9: learning-pressure diagnostic** | Under 80 updates per new task, full displacement achieved training/development CE **0.478 / 2.093**, compared with **1.005 / 1.226** for zero displacement. | Six development roots share one class pool. This reveals a training–development gap, not proof that updating experts is intrinsically harmful. |
| **v11: extended GRPO diagnostic** | Across four controller runs (local/coordinated × two seeds), 48 rounds changed policy probabilities and deployed actions. Coordinated policies at rounds 24/48 exactly matched **global_0** on every one of four development histories. | **344 evaluation rows ≠ 344 independent histories.** The coordinated policies did not outperform that fixed rule or establish useful per-expert coordination. |

For v11 greedy deployment, both global_0 and the coordinated policies at rounds 24/48 had **4/4 online sequence successes, 2/4 final all-task successes on independent development examples, mean positive forgetting F = 0.001300, and 63.75 learner updates on average**. Full-displacement initialization had 3/4, 1/4, F = 0.134125, and 66.25 updates. Thus **better than initialization is not better than a fixed policy**. The four roots reuse an existing development class pool; these numbers are not a held-out confirmatory result.

The project records 55,510 actual learner updates across v11 controller training, development evaluation, and fixed observation probes. These are explicit work counters, **not matched FLOPs**.

## What I learned from the negative result

Extending GRPO training solved the narrower diagnosis of *whether optimization was moving the policy*. It did not solve *whether different learning states warrant different expert allocations*. In the current environment, the coordinated policy learned the conservative constant action. This redirects the next experiment toward **identifying a decision-sensitive regime before investing in a more elaborate controller**.

A convincing controller result must clear three increasingly difficult tests:

1. **Action value:** under matched task-success and cost constraints, do different states have different preferred expert-update patterns, with nontrivial gain over a tuned fixed or time-only schedule?
2. **Information value:** with the same policy class and action budget, does old-task degradation feedback improve decisions beyond progress-only signals?
3. **Learning value:** with the same observations, actions, search budget, and evaluation opportunities, does a learned control law outperform hand-crafted rules and state-blind learned action distributions?

These are **proposed future tests**, not results.

## Minimal next falsification test

Before rerunning GRPO, construct paired interventions from complete learner states and common future minibatches. Measure each feasible action's joint **new-task progress, full-history forgetting, and work**. Split at the history/root level and decide thresholds before the confirmatory evaluation.

- First test whether a per-state expert action has material **regret advantage** over the best globally tuned constant, a scheduled policy, and an equally tuned state-blind learned distribution.
- If the oracle action gap is negligible, stop: increased controller capacity is unlikely to help in this regime.
- Only if the action gap exists, train a compact state-aware controller; compare **progress-only vs progress+old-loss** at matched action space and controller compute.
- Report failures and unconditionally aggregated new-task reachability; show retention differences **conditional on the same jointly feasible roots** alongside the overall failure rates. Use new histories/classes for confirmation; do not recycle the four v11 development roots as new test evidence.

## Inspect and reproduce the archived evidence

The private source archive contains the original code, locked protocols, tests, result matrices, checkpoints, logs, and hash manifests. Useful entry points:

- [Technical report](../../docs/technical-report.md) — detailed evidence and limitations.
- [v7 interpretation](../../experiments/v7_sequence_feedback/结果解读.md) — favorable but unconfirmed feedback result.
- [v9 diagnosis](../../experiments/v9_learning_pressure/结果解读.md) — learning pressure and generalization gap.
- [v11 protocol](../../experiments/v11_rl_diagnostics/protocol.json), [diagnosis](../../experiments/v11_rl_diagnostics/结果解读.md), [report](../../experiments/v11_rl_diagnostics/results/development/analysis/研究诊断报告.md) — exact action-vs-fixed-rule comparison.

In the original full archive (Python 3.9.6, PyTorch 2.8.0, NumPy 2.0.2, Matplotlib 3.9.4, one CPU thread):

```bash
python -m pip install -r requirements.txt
python scripts/verify_artifacts.py
python scripts/test_all.py
# Obtain checksum-verified CIFAR-100 first for the image experiments.
python scripts/run_experiment.py v11 --out runs/v11-reproduction
```

**Reproduction status:** the repository includes the previous run's verification and audit records; this publication preparation did not execute a fresh training run. The original v11 runner requires a pre-existing official CIFAR-100 archive at `.cache/datasets/cifar-100-python.tar.gz`; it does not expose a `--download` flag.

**Publication status:** this page is a draft on a private branch. Neither this branch nor the existing 143 MB archive has been published. A public release should be a **new clean-history repository** assembled from an explicitly reviewed allowlist, not a visibility change on the private source repository. See [release manifest](RELEASE_MANIFEST.md).

---
**Non-claims:** no demonstrated coordination advantage over global_0; no confirmed feedback benefit; no pretrained OLMoE 7B GPU experiment (v10 is an engineering preflight only); no confidence interval or statistical superiority inferred from four v11 development roots.

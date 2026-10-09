# MoE Plasticity Control

[中文](README.md) · [Technical report](docs/technical-report.md) · [Experiment index](README.md#实验索引)

Can online measurements of old-task degradation improve mixture-of-experts updates beyond what new-task learning progress already tells us?

This repository studies that question through nine archived experiments, from synthetic regression to a small multilayer MoE on CIFAR-100. Controllers scale expert optimizer steps; v1 additionally controls anchoring regularization. Experiments track new-task success, old-task retention, and the cost of training, branching, and evaluation.

**The strongest feedback result is preliminary. The real-image pilots have not established an advantage for learned expert coordination.** The project preserves the negative results and the diagnostics that changed the research direction.

## Main findings

| Experiment | Finding | Scope |
|---|---|---|
| [v7 Sequential feedback](experiments/v7_sequence_feedback/结果解读.md) | Within-segment feedback reduced original-task forgetting by about **13.8%** on 56 jointly successful test histories. | Success-rate noninferiority was not established, and the interval from a multiple-comparison sensitivity check includes zero. The metric protects the original task, not all previous tasks. |
| [v8 Multilayer image pilot](experiments/v8_multilayer_retention/结果解读.md) | Both GRPO arms selected their initialization and took the same deterministic actions as full expert displacement. | Three audit histories shared one class pool. Zero expert displacement had lower forgetting but required **25% more learner updates**; expert moments, the stem, and heads still updated. |
| [v9 Learning pressure diagnostics](experiments/v9_learning_pressure/结果解读.md) | At 80 updates per subsequent task, full expert displacement reached training/development CE **0.478/2.093**, versus **1.005/1.226** with zero expert displacement. | Both conditions retained trainable stems and task heads. These are development results from six histories, consistent with a generalization or update-scale problem; they do not show that experts are unnecessary. |

The versions use different tasks, metrics, and budgets. Their scores should not be combined into a cross-version performance ranking.

## What is implemented

- Paired action branches restore learner parameters, optimizer moments, random state, and data streams. Actions within a branch comparison receive the same future batches.
- In v2–v4 and v6–v7, Progress and Feedback comparisons separate new-task progress from additional old-task observations. Conditional retention comparisons on shared successful histories are accompanied by failure rates.
- The image experiments use two MoE layers, per-expert update multipliers, a fixed total replay and monitoring memory, and complete task-performance matrices.
- Shared local and coordinated factorized GRPO controllers are implemented in v8. Their deployed behavior is checked separately from training rewards and parameter changes.
- Frozen source and result hashes, independent integrity audits, and physical cost accounting preserve the evidence behind each conclusion.

v1–v7 use synthetic regression with fixed geometric routing. v8/v9 use task-aware multihead classification with routing determined by fixed raw-image features. They do not validate a pretrained language MoE or a learned router.

## Read and reproduce

Start with the [technical report](docs/technical-report.md), then inspect the linked original reports and JSON summaries. Archived results can be read without running training.

Recorded experiments used Python 3.9.6, PyTorch 2.8.0, and one CPU thread. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/verify_artifacts.py
python scripts/test_all.py
```

The integrity check covers **839 archived files**. The default test command runs **112 mechanism and analysis checks** across v1–v9 in separate processes. These checks verify implementation properties; they do not establish the research hypothesis.

To rerun selected experiments, use a fresh output directory for each run:

```bash
python scripts/run_experiment.py v7 --out runs/v7-reproduction
python scripts/run_experiment.py v8 --pilot --download --out runs/v8-pilot
python scripts/run_experiment.py v9 --download --out runs/v9-development
```

v1–v7 need no dataset download or GPU. v8/v9 need the official CIFAR-100 archive; `--download` permits fetching it. They also run on CPU. The runner preserves the frozen archives and refuses to overwrite an existing output directory. Version-specific protocols and results are indexed in the [Chinese README](README.md#实验索引).

## Optional native language extension

As of 2026-10-09, [v10](experiments/v10_native_moe_feedback/README.md) contains a native OLMoE development runner, prepared source-grouped text data, and CPU checks using a tiny randomly initialized Hugging Face OLMoE. **No GPU experiment with pretrained 7B weights has been run.**

The extension is deferred while the decision objective, independent-history evaluation, measurement resolution, and resource limits are completed. The existing two-history configuration is a development probe, not a confirmatory experiment. The v1–v9 report is complete without this extension. See the [readiness review](docs/v10-preflight.md) for the remaining conditions.

v10 has separate dependencies. With them installed, `python scripts/test_all.py --include-v10` runs **140 checks** in total. Its CPU smoke results are engineering evidence only.

## Research boundaries

Lower forgetting alone is insufficient: a policy may simply learn less, fail more often, or use more computation. A common stopping threshold does not guarantee identical final new-task loss. Success on an online monitor also does not guarantee independent evaluation success or retention after later tasks. Update counts and sample passes are recorded cost measures, not exact FLOPs.

The [technical report](docs/technical-report.md) separates findings from possible explanations and future work. Historical design documents remain available in [docs](docs/); their candidate experiments are not claims of completed work. This repository does not currently specify an open-source license.

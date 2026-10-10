# Repository review — 2026-10-10

## Scope actually inspected
- Repository metadata and full `main` tree at `dc1a998` (949 blobs / approx. 143.4 MB); chronological commit subjects.
- Main and English READMEs, technical report, v7/v9/v10/v11 interpretation reports, v11 locked protocol, audit and cost records, six-round regression record.
- v8 model/optimizer/environment/data/control interfaces and v11 diagnostic GRPO loop/runner/analyzer/audit entry points.
- Did **not** download and run the full 143 MB archive, execute its 129 tests, rerun training, deserialize checkpoints, or inspect the complete contents of all tracked blobs. Accordingly, the recorded audits are **claimed by archival records, not independently re-executed here**.

## Verified directly from source
1. v8 MoE has two residual layers of eight experts and a raw-image-derived fixed router. `ScaledAdam.step` applies full Adam and then scales each expert's actual parameter displacement. At m=0 optimizer moments advance and stem/heads can train.
2. v11 imports v8's environment and controller, uses the same one-update-per-group clipped-ratio expression, but adds read-only probes, separate policy-gradient/entropy-gradient measurements and checkpoint diagnostics.
3. The v11 protocol fixes four training and four development roots, two controller arms × two policy seeds, 48 rounds, five checkpoints, greedy+three sampled draws and six baselines. Grid size is 344 rows, with **four** development roots as experimental units.
4. The archived audit records 20 checkpoints, 23 probes, 344 evaluation rows and 55,510 physical learner updates. Original v8 seed-881 first-six-round regression record reports exact agreement in reward/reward std/loss. Original v8 traces do not permit checking per-action/logprob equality.
5. No model weights from pretrained 7B OLMoE were run on GPU; v10 is a separate optional CPU-engineering-only preflight.
6. `run_experiment.py v11 --download` is invalid: v11 has no download flag; official CIFAR-100 archive must already be cached.
7. Current repository has no open-source license and includes many checkpoints/results. Publishing this repository as-is is **not recommended**.

## Interpretation guardrails
- **v7 ~13.8%**: common-success subset, original A only, success-rate noninferiority unproven, multiple-comparison sensitivity interval crosses zero.
- **v9 generalization gap**: one small fixed-routing CIFAR environment, six roots; not evidence that experts can generally be discarded.
- **v11**: optimizer updates and actions change, but learned coordination matches fixed `global_0` exactly on four development histories; online success ≠ final independent-development all-task success. Results **do not prove** control outperforming a strong fixed baseline.
- No matched FLOPs, no held-out confirmatory study for v11, no evidence of pretrained language-MoE continual learning.

## Blocking release items
- Authorization/ownership/license review, including dataset processing and work-product provenance.
- Detailed content scan of **every** allowed file and current git history leakage prevention via new clean-history repo.
- Ensure all v8/v11 transitive code and archived six-round replay references are either present or explicitly refactored and retested.
- Produce standalone minimal public summary/reproduction with no dead private links, then rerun the entire thin-repo test set and at least a smoke task.
- No public action has been taken.

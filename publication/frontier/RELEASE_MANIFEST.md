# Frontier public release manifest (draft)

## Decision
**Not approved for publication.** Original repository `MeverSeen/moe-plasticity-control` is private and contains ~143.4 MB across 949 tracked blobs in the inspected `main` tree (commit `dc1a998`). This checklist records **candidate files**, not disclosure authorization.

**Never flip the source repository's visibility or fork its history.** Create a **new repository with an orphan/clean initial commit** and import only reviewed files. An ordinary branch of the old repository still retains reachable historical objects. Do not push an archive, preexisting `.git`, or all `results/` directories.

## Candidate narrow demonstration (v8 + v11)
All path names below are relative to the private repository root.

**Candidate code, subject to license and content review**
- `experiments/v8_multilayer_retention/{data.py,model.py,environment.py,control.py,run_retention.py}`
- `experiments/v11_rl_diagnostics/{protocol.json,diagnostic_control.py,run_diagnostics.py,analyze.py}`
- `experiments/v8_multilayer_retention/test_*.py` and `experiments/v11_rl_diagnostics/test_*.py` (manually enumerate and audit each).
- `requirements.txt`
- A **new** compact readme and a new standalone runner instead of copying `scripts/run_experiment.py` unchanged (that dispatcher contains paths for all archived versions).
- A **new** minimal numerical summary, curated from audited v11 results; no unreviewed full JSON logs, checkpoints or datasets.

**Important dependency closure:** `run_diagnostics.py` imports v8 and reads `experiments/v8_multilayer_retention/results/pilot/models/learned_{arm}_881_training.json` for six-round regression. For a self-contained public demo, either allowlist these **two specific training logs** after inspection, or explicitly refactor the regression into an optional separate verification mode, label that change, and rerun tests. Do **not** silently delete this dependency while claiming the original run is reproducible.

**Candidate references**
- `publication/frontier/README.md` (draft)
- Carefully reviewed excerpts of `docs/technical-report.md` and `experiments/v11_rl_diagnostics/结果解读.md`, with private-repo links replaced by valid public paths.
- `experiments/v11_rl_diagnostics/results/development/{locked_protocol.json,original_replay_check.json,total_cost.json}` and the small `results/development/analysis/audit.json`, **only after checking file-level provenance** and verifying the reproduction claim.
- `experiments/v11_rl_diagnostics/results/development/analysis/研究诊断报告.md` after link repair.
- New minimal `EVIDENCE.md` and `REPRODUCIBILITY.md`.

**Default exclude**
- All `.cache/`, source corpora, model/pretrained weights, raw dataset copies and tokenizers.
- All `results/` and `*.pt` by default; make individual written exceptions after inspecting contents and ownership.
- Large raw training logs, full snapshots, random samples, intermediate branch data, debug dumps.
- Work product of third parties or employers unless independently authorized. Remove company names, internal server paths, usernames/emails, credentials, device identifiers and business materials.
- v10 OLMoE experimental corpora, preprocessing archives and snapshots (not needed to demonstrate v11).
- `.git/` and existing git history, including deleted historical secrets.

## Before publication
1. Enumerate precisely copied paths; inspect **content**, not just filenames, for secrets, identifiers, licenses, copyrighted text and contractual constraints. A regex-only scan is insufficient.
2. Resolve rights to release original research code and derivative dataset-processing code. Verify the official CIFAR-100 use terms and avoid redistributing the dataset itself. The current private repository has **no open-source license**; choose one deliberately for the new distribution if rights permit.
3. Prove dependency closure with a fresh isolated environment and only allowlisted files. Check Python/PyTorch versions; use official training split without accidental test exposure.
4. Re-run relevant unit tests, original-result integrity checks (where artifacts are included), and at least the smoke reproduction. Don't advertise 129 full-archive tests for a new thin demo if they were not run there.
5. Check that each numeric claim links to a protocol and a reviewable summary or regenerated result. State root-level sample size, success/failure outcomes and development-only status.
6. Confirm public links work without GitHub sign-in. The draft currently contains **private-relative paths**; copy or rewrite them in the new repository.
7. Verify the clean repository's first commit has no ancestor containing unreviewed material. Explicitly inspect `git rev-list --all` and `git ls-files` before making it public.
8. Run a full manual release approval step. No visibility changes, uploads, license grants or public links have been approved by this draft.

## Repro caveats
- v11 is a four-root **development** comparison, not a confirmation on held-out class pools.
- Zero displacement leaves the stem/head trainable and Adam moments advancing.
- The only training-data source in v11 is the official CIFAR-100 *train* split. Original v11 runner does **not** auto-download; place checksum-verified `cifar-100-python.tar.gz` into `.cache/datasets/`.
- The archived checks cover implementation/audit behavior, not proof of the research hypothesis.

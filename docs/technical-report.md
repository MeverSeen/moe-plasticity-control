# MoE Plasticity Control Technical Report

Research status: 2026-10-09. [Project overview](../README.en.md).

## Scope and current conclusion

This project studies expert updates in continual learning: with fixed mixture-of-experts capacity, limited retained data, and an explicit work budget, can a controller help a model learn new tasks while preserving earlier knowledge? The central information comparison asks whether current old-task degradation improves decisions beyond new-task learning progress. Individual expert coordination is a separate extension of that question.

The evidence does **not** establish a robust incremental benefit from old-task feedback or a learned-coordination advantage in the image pilot. One sequential synthetic result is encouraging but remains unconfirmed. The later image studies show unchanged deployed controller actions and a large training–development gap. They motivate checking adaptation and measurement before expanding controller training; they do not identify a single cause of failure.

Results from different versions should not be ranked as though they share a benchmark. They use different tasks, models, targets, memories, and evaluation protocols. The comparisons below are evidence within each locked study, with limitations attached to the relevant claim.

## Mechanism and fair comparison

v1 controls each expert's optimizer step and anchoring regularization strength. v2–v7 instead use one multiplier for a protected expert group; zero prevents weight displacement while optimizer moments still advance. v8/v9 expose two layers of eight experts with global, grouped, local, and coordinated displacement multipliers. A comparison needs more than a lower forgetting number:

- New-task reachability is reported alongside preservation. A method that fails to learn cannot be treated as a successful retention method.
- The synthetic experiments keep an original-task reference. In v8/v9, forgetting uses each task's own learning-time endpoint; the full-history matrix separates initial generalization errors from subsequent damage.
- Failures remain in all-root summaries. Paired forgetting comparisons on jointly successful histories are conditional results and do not replace the reachability record.
- Development selection and final evaluation are separated when a study has an audit split. Work counters include learner updates, sample passes, monitoring, and endpoint evaluation; they are explicit proxies, not hardware FLOPs or matched wall-clock cost.

These choices matter because apparent control gains can otherwise arise from early stopping, different successful subsets, reset baselines, or free monitoring. The detailed contracts and checks are preserved with the original runs: [v1](../experiments/v1_expert_control/实验报告.md), [v2](../experiments/v2_feedback_control/实验报告.md), [v7](../experiments/v7_sequence_feedback/results/main/analysis/实验报告.md), [v8 validation](v8-validation.md), and [v9 validation](v9-validation.md).

## Evidence from the synthetic control sequence

Versions 1–6 progressively asked whether the data and interface supplied enough decision value before making stronger learning claims. v1 did not show a PPO expert controller outperforming an importance-rule baseline. In v2, both progress-only and old-knowledge-feedback PPO arms selected epoch-zero actors, which deterministically chose the same protective action; the equal test result is evidence of identical deployed policies, not evidence that feedback has zero value. The [v2 report](../experiments/v2_feedback_control/实验报告.md) records 77/80 reached histories for the selected fixed protection action versus 78/80 for full updating, with a preservation–reachability trade-off rather than controller superiority.

v3 introduced paired action branches from the same complete learner state and continuation batches. Its principal branch and closed-loop comparisons still had intervals crossing zero. v4 added learner-state aggregation and equal-query non-learning controls; all groups reached the target on its 80 test histories, but aggregation did not surpass the matched controls. These are important checks against a tempting but unsupported story that merely adding state, old-loss observations, or more controller capacity creates a benefit.

v5 calibrated targets and budgets while searching fixed choices and schedules. A privileged per-history fixed choice had localized value, but privileged plan selection did not reliably beat the locked time schedule, and the fully feasible subset did not beat direct freezing. v6 then separated three possible failures: inability to learn a mapping, weak supervision, and poor model selection. The basic learning checks passed; selecting candidates by actual closed-loop development behavior reduced forgetting from 0.007971 to 0.006028 on 96 new histories at the same reachability. That is a selection result, not an online-feedback result. The primary Feedback–Progress gap was -0.000101 with a 95% interval of [-0.000376, 0.000181], and strong time policies remained unbeaten. See the [v5 interpretation](../experiments/v5_calibration/结果解读.md) and [v6 diagnostics](../experiments/v6_learning_diagnostics/结果解读.md).

The v5 privileged references use future evaluation labels for selection; their separate continuation resamples training batches, not task identity or evaluation data. They are diagnostic references, not deployable policies. Together, v1–v6 show that fixed protection and closed-loop selection can matter without establishing incremental online-feedback value.

## v7: a cautious sequential signal

v7 continued through B1, B2, and B3 without resetting parameters or Adam state, while the original A reference and memory remained fixed. Within-segment Feedback added current A-loss signals to a matched Progress controller. There were 64 test roots and three controller initializations; these are not 192 independent histories. On 56 roots where both arms succeeded under every initialization, A forgetting fell from 0.007195 to 0.006205, about 13.8%. The Feedback–Progress difference was -0.000990, with nominal paired 95% interval [-0.001963, -0.000071]. Forgetting here is a positive increase in independent A-evaluation MSE.

Feedback's mean reachability was slightly lower (89.06% versus 89.58%); noninferiority was not established. A post-run Bonferroni sensitivity interval, [-0.002125, 0.000065], crossed zero. These intervals describe root sampling, with limited coverage of controller-training randomness. The B tasks were not added to replay; only about 48%–49% of learning-group histories still had all three B tasks at threshold at the end. Segment-level Feedback produced a different reachability–forgetting trade-off. See the [v7 interpretation](../experiments/v7_sequence_feedback/结果解读.md), [primary summary](../experiments/v7_sequence_feedback/results/main/analysis/summary.json), and [additional diagnostics](../experiments/v7_sequence_feedback/results/main/analysis/additional_diagnostics.json).

Within-segment Feedback also had lower forgetting than the development-selected schedule on their common successful roots. That favorable comparison with a finite schedule search must be distinguished from the incremental value of feedback over Progress. Offline work was substantial: branch queries used 541,302 subsequent learner updates, and development candidate selection used another 181,139. These costs are separate from deployed-policy execution.

## v8: a stronger image environment, no learned-controller gain

v8 moved to CIFAR-100 and a small two-layer MoE. Each history contains five 5-way tasks, total old memory is fixed at 128 replay plus 128 monitor examples, and task matrices record every learned task after each endpoint. All methods receive the same environment observation and action levels; rules use selected fields, the local controller uses expert summaries, and the coordinated controller adds learned layer/global pooling. Online monitor success and final held-out performance are evaluated separately.

Both learned arms selected their epoch-zero checkpoint, whose deterministic action was full expert displacement. On the three audit roots, they were behaviorally identical to `global_1`; learning had not improved the deployed policy. Zero expert displacement had lower endpoint positive forgetting (0.000434 versus 0.016931) but used 25% more learner updates, so it is not a compute-matched win. Expert moments still advanced in this condition. There was one controller seed per arm, and the three audit roots shared a 25-class pool. The [v8 result note](../experiments/v8_multilayer_retention/结果解读.md) and [raw report](../experiments/v8_multilayer_retention/results/pilot/analysis/实验报告.md) retain the favorable preservation point estimate and its limits.

v8 is not a native learned-router result. Routing is derived from fixed raw-image features; task identity is visible through separate classification heads. The CNN stem and task heads remain trainable even when expert displacement is zero. Those design choices make the study a useful controlled MoE-update environment, but they prevent claims about learned routing or language-model MoEs.

## v9: learning pressure exposed a generalization problem

v9 ran 30 conditions for each of six roots, yielding 180 development trajectories that share the same 25-class pool. Conditions within a root branch from its common first-task checkpoint; they are not independent histories. The study tightened subsequent-task targets from CE 1.55 to 1.25, compared six update rules, and added fixed-budget and module-scope interventions. It did not load official test data or train a new learned controller.

Stricter targets did not make full updating outperform zero expert displacement on the reported reach rates. At target 1.35, early-stop `global_0` achieved 18/24 online monitor hits and 11/24 simultaneous held-out-development hits; `global_1` achieved 10/24 and 6/24. These 24 tasks are nested within six roots. `global_0` still trains the stem and task heads. In separate runs using 80 updates per subsequent task, full updating reached mean terminal training/development CE of 0.478/2.093, versus 1.005/1.226 with zero expert displacement. The full-updating curve continued to reduce training loss while development error rose.

This pattern is compatible with overfitting or unstable updates, but it does not identify a cause. Hard-freezing the stem changes both representation adaptation and its interaction with expert updates; low forgetting for heads-only training also comes with weak new-task learning. Fixed-budget and early-stop rollouts are not interchangeable prefixes, and matching update counts does not match FLOPs. The [v9 interpretation](../experiments/v9_learning_pressure/结果解读.md) and [development audit](../experiments/v9_learning_pressure/results/development/analysis/diagnostics.json) retain these distinctions. Physical work totaled 31,735 learner updates, counting each shared prefix once; summing the records naively would give 35,070.

## Limitations that govern interpretation

The completed environments are deliberately small. v1–v7 use fixed geometric routing in synthetic regression; v8–v9 use fixed raw-image routing and task-ID-visible multi-head classification. Neither is a learned-router continual-learning result, and neither supports a language-model claim. Small numbers of controller seeds and shared class pools limit uncertainty estimates. In v9, development data diagnose an effect but cannot provide a final generalization confirmation. Most importantly, a lower retained-task loss cannot be separated from an inability to acquire the new task unless the reachability record, stopping rule, and module scope are read with it. These limitations constrain the claim, rather than erase the reproducible negative evidence.

## Reproduction and next decision

Archived v1–v9 results can be checked without rerunning training. The [English README](../README.en.md) gives installation and reproduction commands: 839 archived files have integrity hashes, and the default suite runs 112 checks. v8/v9 reruns require official CIFAR-100 data; v1–v7 need no external dataset. The image result directories include full task matrices alongside protocols, costs, and analysis. All completed experiments ran on CPU. Reproduction writes to a fresh `runs/` directory and leaves archives unchanged.

Optional v10 targets a fixed expert group in a pretrained language MoE. Its runner implements B-training route-occupancy logs, optimizer-state restoration, and separate old-domain evaluation. Only CPU engineering checks using a tiny random model have run; pretrained 7B GPU execution remains untested. A decision objective, analysis controls, independent-history plan, and bounded pilot must precede any larger study. The [v10 readiness review](v10-preflight.md) separates CPU preparation from measurements requiring a GPU.

The v1–v9 study stands without v10. It offers a preliminary A-retention signal, a real-image pilot without learned-control improvement, and evidence that training progress can diverge sharply from development performance. Confirming the v7 feedback effect requires fresh histories and a predeclared success-rate noninferiority margin. Extending that claim to all-history retention or native language MoEs requires separate evidence.

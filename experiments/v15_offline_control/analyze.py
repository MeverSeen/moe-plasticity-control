"""Dataset coverage and reward diagnostics, never used for policy fitting."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from common import save_json


def ranks(values):
    values = np.asarray(values, dtype=float)
    out = np.empty(len(values)); order = np.argsort(values, kind="stable")
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        out[order[i:j]] = (i + j - 1) / 2
        i = j
    return out


def spearman(a, b):
    if len(a) < 3:
        return None
    a, b = ranks(a), ranks(b)
    if np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def analyze(root, output):
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows, unique_actions, levels, roots = [], set(), [], {}
    for path in sorted((root / "results/train").glob("*.json")):
        r = json.loads(path.read_text())
        old = r["raw_terminal"]["old_tasks"]
        row = {"file": path.name, "root": r["root"]["id"], "behavior": r["behavior"]["name"],
               "success": r["success"], "failures": r["failure_count"],
               "decisions": len(r["trace"]), "updates": r["cost_actual"]["updates"],
               "monitor_positive_F": r["monitor"]["forgetting"],
               "monitor_signed_F": r["monitor"]["signed_forgetting"],
               "old_ce": float(np.mean([x["mean_ce"] for x in old])),
               "effective_distance": r["parameter_distance"]["effective_sq"],
               "eval_F_diagnostic_only": r["evaluation"]["forgetting"],
               "final_eval_ce_diagnostic_only": r["evaluation"]["final_loss"],
               "wall_seconds": r["total_wall_seconds_including_logging"]}
        rows.append(row); roots.setdefault(row["root"], []).append(row)
        for item in r["trace"]:
            action = np.array(item["experts"], dtype=np.float32)
            unique_actions.add(hashlib.sha256(action.tobytes()).hexdigest())
            levels.append(action)
    if not rows:
        raise ValueError("No completed training trajectories")
    with (output / "training_histories.csv").open("w") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    diagnostics = []
    for name, group in roots.items():
        success = [x for x in group if x["success"]]
        diagnostics.append({"root": name, "histories": len(group), "successful": len(success),
                            "monitor_zero_fraction": float(np.mean([x["monitor_positive_F"] == 0 for x in group])),
                            "ranges": {k: [min(x[k] for x in group), max(x[k] for x in group)]
                                       for k in ["monitor_positive_F", "old_ce", "effective_distance"]},
                            "successful_within_root_cost_vs_eval_F_spearman": {
                                k: spearman([x[k] for x in success], [x["eval_F_diagnostic_only"] for x in success])
                                for k in ["monitor_positive_F", "old_ce", "effective_distance"]}})
    summary = {"histories": len(rows), "roots": len(roots),
               "decisions": sum(x["decisions"] for x in rows), "unique_executed_joint_actions": len(unique_actions),
               "learner_updates_in_branches": sum(x["updates"] for x in rows),
               "action_entry_fractions_not_independent_samples": {
                   str(x): float((np.stack(levels) == x).mean()) for x in [0., .25, .5, .75, 1.]},
               "reward_diagnostics": diagnostics,
               "limits": "Training-pool observational diagnostics only. Fixed initial state does not isolate one factor across different action sequences. Correlations are not held-out efficacy or statistical significance; expert entries are not independent samples."}
    save_json(output / "dataset_summary.json", summary)
    if (root / "evaluation_rows.json").exists():
        evaluation = json.loads((root / "evaluation_rows.json").read_text())
        with (output / "closed_loop.csv").open("w") as file:
            writer = csv.DictWriter(file, fieldnames=list(evaluation[0])); writer.writeheader(); writer.writerows(evaluation)
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--root", required=True)
    p.add_argument("--output", required=True); a = p.parse_args()
    print(json.dumps(analyze(a.root, a.output), indent=2))

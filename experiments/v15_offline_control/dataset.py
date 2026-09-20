"""Immutable complete-history import and reward relabelling for offline control."""
import argparse
import json
from pathlib import Path
import torch
from common import save_json, file_sha

REWARDS = ("positive_forgetting", "old_ce", "effective_distance")


def endpoint_reward(result, name, loss_scale=.1, distance_scale=1., kappa=3.):
    if not result.get("history_completed", True):
        raise ValueError("An interrupted history is not a terminal training example")
    if loss_scale <= 0 or distance_scale <= 0:
        raise ValueError("Reward scales must be fixed and positive")
    if name == "positive_forgetting":
        cost = float(result["monitor"]["forgetting"]) / loss_scale
    elif name == "old_ce":
        raw = result.get("raw_terminal", {}).get("old_tasks")
        if not raw:
            raise ValueError("Raw per-task old CE was not logged; cannot infer it from forgetting")
        cost = sum(x["mean_ce"] for x in raw) / len(raw) / loss_scale
    elif name == "effective_distance":
        cost = float(result["parameter_distance"]["effective_sq"]) / distance_scale
    else:
        raise ValueError(name)
    return -cost - kappa * int(result["failure_count"])


def convert(result, reward_name, source="", loss_scale=.1, distance_scale=1.):
    split = result["root"]["split"]
    if split not in ("train", "development"):
        raise ValueError("Confirmation/evaluation histories cannot enter an offline training dataset")
    reward = endpoint_reward(result, reward_name, loss_scale, distance_scale)
    trace = result["trace"]
    transitions = []
    for i, item in enumerate(trace):
        if item["duration"] <= 0:
            raise ValueError("Unexecuted action cannot be a transition")
        terminal = i == len(trace) - 1
        obs = tuple(torch.tensor(x, dtype=torch.float32) for x in item["observation"])
        nxt = tuple(torch.zeros_like(x) for x in obs) if terminal else tuple(
            torch.tensor(x, dtype=torch.float32) for x in trace[i + 1]["observation"])
        transitions.append({"obs": obs, "next_obs": nxt,
                            "action": torch.tensor(item["experts"], dtype=torch.float32),
                            "latent": list(item["latent"]),
                            "behavior_logp": item.get("behavior_logp", item.get("policy_logp")),
                            "duration": item["duration"],
                            "terminated": terminal, "truncated": False,
                            "reward": reward if terminal else 0.,
                            "root": result["root"]["id"], "split": split,
                            "trajectory": source, "index": i})
    return transitions


def build(paths, output, reward="positive_forgetting", loss_scale=.1, distance_scale=1.):
    rows, manifest, rejected = [], [], []
    for path in sorted(map(Path, paths)):
        result = json.loads(path.read_text())
        try:
            converted = convert(result, reward, str(path), loss_scale, distance_scale)
        except (ValueError, KeyError, TypeError) as error:
            rejected.append({"path": str(path), "reason": str(error)})
            continue
        rows.extend(converted)
        manifest.append({"path": str(path), "sha256": file_sha(path),
                         "root": result["root"]["id"], "split": result["root"]["split"],
                         "decisions": len(converted), "reward": endpoint_reward(result, reward, loss_scale, distance_scale)})
    if not rows:
        raise ValueError("No eligible executed transitions")
    roots = {}
    for row in rows:
        old = roots.setdefault(row["root"], row["split"])
        if old != row["split"]:
            raise ValueError("Shared root crossed a split boundary")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"schema": 1, "reward": reward, "gamma": 1.,
                "loss_scale": loss_scale, "distance_scale": distance_scale,
                "transitions": rows, "sources": manifest}, output)
    save_json(output.with_suffix(".json"), {"schema": 1, "reward": reward,
        "transitions": len(rows), "histories": len(manifest), "roots": roots,
        "terminal_rewards": [x["reward"] for x in manifest],
        "sources": manifest, "rejected": rejected, "dataset_sha256": file_sha(output)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reward", choices=REWARDS, default=REWARDS[0])
    parser.add_argument("--distance-scale", type=float, default=1.)
    args = parser.parse_args()
    build(args.input.glob("*.json"), args.output, args.reward, distance_scale=args.distance_scale)

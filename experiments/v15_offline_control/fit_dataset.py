"""Lock reward scales using train-only records, then fit BC and three IQLs."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from common import file_sha, save_json
from dataset import build
from offline import train


def fit(root, protocol):
    root = Path(root)
    plan = json.loads(Path(protocol).read_text())
    inputs = sorted((root / "results/train").glob("*.json"))
    if not (root / "collection_summary.json").exists():
        raise RuntimeError("Collection must be finalized before the immutable dataset is locked")
    records = [json.loads(p.read_text()) for p in inputs]
    if any(x["root"]["split"] != "train" for x in records):
        raise ValueError("Non-training root in collection folder")
    distances = [r["parameter_distance"]["effective_sq"] for r in records
                 if r["behavior"]["name"] == "fixed050" and r["parameter_distance"]["effective_sq"] > 0]
    rewards = ["positive_forgetting", "old_ce"]
    scale = float(np.median(distances) / 10) if distances else None
    if scale is not None:
        rewards.append("effective_distance")
    lock = {"locked_epoch": time.time(), "source_protocol_sha256": file_sha(protocol),
            "histories": len(inputs), "sources": {p.name: file_sha(p) for p in inputs},
            "distance_scale": scale, "scale_rule": plan["rewards"]["distance_scale"],
            "rewards": rewards, "training_only": True}
    save_json(root / "offline/dataset_lock.json", lock)
    outputs = []
    for reward in rewards:
        path = root / "offline" / (reward + ".pt")
        build(inputs, path, reward, distance_scale=scale or 1.)
        result = train(path, root / "offline" / ("iql_" + reward), "iql",
                       steps=plan["offline"]["steps"], seed=plan["offline"]["seed"],
                       batch_size=plan["offline"]["batch_size"],
                       actor_every=plan["offline"]["actor_every"],
                       warmup=plan["offline"]["bc_warmup_steps"])
        outputs.append(result)
    outputs.append(train(root / "offline/old_ce.pt", root / "offline/bc", "bc",
                         steps=plan["offline"]["steps"], seed=plan["offline"]["seed"],
                         batch_size=plan["offline"]["batch_size"],
                         actor_every=plan["offline"]["actor_every"]))
    save_json(root / "offline/summary.json", {"fits": outputs, "dataset_lock": lock})


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--root", required=True)
    p.add_argument("--protocol", required=True); args = p.parse_args()
    fit(args.root, args.protocol)

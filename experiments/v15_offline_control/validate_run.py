"""Post-run integrity checks; never used to select a controller."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from common import file_sha, save_json
from dataset import convert, endpoint_reward
from structured_policy import draw_action


def validate_history(record):
    assert record["history_completed"]
    assert record["failure_count"] == sum(t["success"] is False for t in record["tasks"])
    assert len(record["learner_steps"]) == record["cost_actual"]["updates"]
    assert sum(t["duration"] for t in record["trace"]) == len(record["learner_steps"])
    assert abs(record["reward"]["telescope_error"]) < 1e-8
    assert abs(record["restore_check"]["fresh_ce"] - record["restore_check"]["cached_ce"]) <= 1e-7
    signed = []
    for old in record["raw_terminal"]["old_tasks"]:
        assert len(old["sample_ids"]) == len(old["ce"]) == len(old["baseline"])
        assert np.isclose(np.mean(old["ce"]), old["mean_ce"], atol=1e-12)
        change = float(np.mean(np.asarray(old["ce"]) - old["baseline"]))
        assert np.isclose(change, old["signed_change"], atol=1e-12)
        signed.append(change)
    assert np.isclose(np.maximum(signed, 0).mean(), record["monitor"]["forgetting"], atol=1e-12)
    assert np.isclose(np.mean(signed), record["monitor"]["signed_forgetting"], atol=1e-12)
    for item in record["trace"]:
        obs = tuple(torch.tensor(v) for v in item["observation"])
        assert obs[0].shape == (16, 64, 10) and obs[1].shape == (10,)
        assert all(torch.isfinite(v).all() for v in obs)
        action, _, _, _ = draw_action(
            obs, lambda _: torch.zeros(5), lambda *_: torch.zeros(3),
            lambda *_: torch.zeros(64), lambda *_: torch.zeros(5), 0.,
            latent=[torch.tensor(v) for v in item["latent"]])
        assert torch.equal(action, torch.tensor(item["experts"]))
        assert 1 <= item["duration"] <= 5
        assert np.isfinite(item["behavior_logp"])
        if not record["behavior"]["stochastic"]:
            assert item["behavior_logp"] == 0.
    if record["root"]["split"] == "train":
        for reward in ("positive_forgetting", "old_ce", "effective_distance"):
            transitions = convert(record, reward)
            if transitions:
                assert sum(t["terminated"] for t in transitions) == 1
                assert np.isclose(sum(t["reward"] for t in transitions), endpoint_reward(record, reward))


def validate(outer, output):
    outer = Path(outer)
    root = outer / "data"
    verified_sources = {}
    for name in ("source_lock.json", "offline_source_lock.json"):
        lock = json.loads((outer / name).read_text())
        for remote, sha in lock["files"].items():
            relative = remote.split("/code/", 1)[1]
            local = outer / "code" / relative
            assert file_sha(local) == sha, str(local)
            verified_sources[relative] = sha
        assert file_sha(outer / "code/experiments/v15_offline_control/protocol.json") == lock["protocol_sha256"]
    restore = [json.loads(p.read_text()) for p in sorted((root / "restore_checks").glob("*.json"))]
    assert len(restore) == 5 and all(x["equal"] for x in restore)
    records = []
    for phase in ("train", "restore_diagnostic", "development", "confirmation"):
        for path in sorted((root / "results" / phase).glob("*.json")):
            if path.stem.endswith(("__general", "__incomplete")):
                continue
            r = json.loads(path.read_text())
            validate_history(r)
            records.append({"path": str(path.relative_to(outer)), "sha256": file_sha(path),
                            "phase": phase, "root": r["root"]["id"], "decisions": len(r["trace"])})
    dataset_lock = json.loads((root / "offline/dataset_lock.json").read_text())
    for name, sha in dataset_lock["sources"].items():
        assert file_sha(root / "results/train" / name) == sha
    selected = json.loads((root / "selection_lock.json").read_text())
    assert file_sha(root / "offline" / selected["selected"] / "controller.pt") == selected["controller_sha256"]
    summary = json.loads((root / "evaluation_summary.json").read_text())
    assert summary["base_start"] == summary["base_end"]
    result = {"status": "passed", "sources": verified_sources, "restore_checks": len(restore),
              "complete_histories": records, "decisions": sum(r["decisions"] for r in records),
              "scope": "Integrity and reward/action semantics; not evidence of controller efficacy."}
    save_json(output, result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--outer", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    result = validate(args.outer, args.output)
    print(json.dumps({k: v for k, v in result.items() if k not in ("sources", "complete_histories")}))

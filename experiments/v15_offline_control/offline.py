"""CPU BC and IQL on saved transitions; no learner/model interaction."""
import argparse
import copy
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from common import save_json, file_sha
from structured_policy import ActorCritic


class ValueNetwork(nn.Module):
    def __init__(self, action=False):
        super().__init__()
        self.action = action
        self.encode = nn.Sequential(nn.Linear(22 if action else 21, 48), nn.Tanh(),
                                    nn.Linear(48, 32), nn.Tanh())
        self.head = nn.Sequential(nn.Linear(74, 64), nn.Tanh(), nn.Linear(64, 1))

    def forward(self, local, global_, action=None):
        n, layers, experts, _ = local.shape
        position = torch.linspace(0, 1, layers, device=local.device)[None, :, None, None]
        parts = [local, global_[:, None, None, :].expand(n, layers, experts, -1),
                 position.expand(n, layers, experts, 1)]
        if self.action:
            if action is None:
                raise ValueError("Q requires the actually executed action")
            parts.append(action[..., None])
        h = self.encode(torch.cat(parts, -1))
        pooled = torch.cat([h.mean((1, 2)), h.amax((1, 2)), global_], -1)
        return self.head(pooled).squeeze(-1)


def tensor_batch(rows):
    return {"local": torch.stack([r["obs"][0] for r in rows]),
            "global": torch.stack([r["obs"][1] for r in rows]),
            "next_local": torch.stack([r["next_obs"][0] for r in rows]),
            "next_global": torch.stack([r["next_obs"][1] for r in rows]),
            "actions": torch.stack([r["action"] for r in rows]),
            "reward": torch.tensor([r["reward"] for r in rows]),
            "terminal": torch.tensor([r["terminated"] for r in rows], dtype=torch.float32)}


def expectile_loss(residual, expectile=.7):
    weight = torch.where(residual > 0, expectile, 1 - expectile)
    return (weight * residual.square()).mean()


def bellman_target(reward, terminal, next_value):
    # gamma=1 per decision; variable window duration remains recorded in the data.
    return reward + (1 - terminal) * next_value


def train(dataset_path, output, method="iql", steps=1200, seed=1515,
          batch_size=16, actor_every=4, warmup=200):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    package = torch.load(dataset_path, map_location="cpu", weights_only=False)
    rows = [r for r in package["transitions"] if r["split"] == "train"]
    if not rows:
        raise ValueError("No training roots")
    if any(r.get("truncated") for r in rows):
        raise ValueError("This first implementation accepts complete histories only")
    actor = ActorCritic()
    q1, q2, value = ValueNetwork(True), ValueNetwork(True), ValueNetwork()
    tq1, tq2 = copy.deepcopy(q1), copy.deepcopy(q2)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)
    q_optimizer = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=3e-4)
    v_optimizer = torch.optim.Adam(value.parameters(), lr=3e-4)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    log = []
    started = time.monotonic()
    for step in range(steps):
        selected = [rows[i] for i in rng.integers(len(rows), size=batch_size)]
        batch = tensor_batch(selected)
        l, g, a = batch["local"], batch["global"], batch["actions"]
        metrics = {"step": step + 1}
        if method == "iql":
            with torch.no_grad():
                q = torch.minimum(tq1(l, g, a), tq2(l, g, a))
            v_loss = expectile_loss(q - value(l, g))
            v_optimizer.zero_grad(); v_loss.backward()
            nn.utils.clip_grad_norm_(value.parameters(), 5.)
            v_optimizer.step()
            with torch.no_grad():
                target = bellman_target(batch["reward"], batch["terminal"],
                                        value(batch["next_local"], batch["next_global"]))
            q_loss = ((q1(l, g, a) - target).square().mean()
                      + (q2(l, g, a) - target).square().mean())
            q_optimizer.zero_grad(); q_loss.backward()
            nn.utils.clip_grad_norm_(list(q1.parameters()) + list(q2.parameters()), 5.)
            q_optimizer.step()
            with torch.no_grad():
                for target_net, net in ((tq1, q1), (tq2, q2)):
                    for p, current in zip(target_net.parameters(), net.parameters()):
                        p.lerp_(current, .005)
            metrics.update(q_loss=float(q_loss.detach()), value_loss=float(v_loss.detach()))
        if step % actor_every == 0:
            with torch.no_grad():
                if method == "bc" or step < warmup:
                    weights = torch.ones(len(selected))
                else:
                    advantage = torch.minimum(tq1(l, g, a), tq2(l, g, a)) - value(l, g)
                    weights = torch.exp((advantage / 3.).clamp(max=np.log(20.)))
            logps = torch.stack([actor(r["obs"], [torch.tensor(z) for z in r["latent"]])[2]
                                 for r in selected])
            # Exact JOINT ordered latent likelihood; not independent expert labels.
            actor_loss = -(weights * logps).mean()
            actor_optimizer.zero_grad(); actor_loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), 5.)
            actor_optimizer.step()
            metrics.update(actor_loss=float(actor_loss.detach()),
                           mean_weight=float(weights.mean()))
        if (step + 1) % 100 == 0 or step == 0 or step + 1 == steps:
            metrics["wall_seconds"] = time.monotonic() - started
            log.append(metrics)
            save_json(output / "learning.json", log)
            print(json.dumps(metrics), flush=True)
    torch.save({"policy": actor.state_dict(), "q1": q1.state_dict(), "q2": q2.state_dict(),
                "value": value.state_dict(), "method": method,
                "reward": package["reward"], "dataset_sha256": file_sha(dataset_path),
                "seed": seed, "steps": steps}, output / "controller.pt")
    with torch.no_grad():
        # Fixed diagnostic subset; never a deployment-performance estimate.
        diagnostic = rows[::max(1, len(rows) // 32)][:32]
        match = [bool(torch.equal(actor(r["obs"], deterministic=True)[0], r["action"]))
                 for r in diagnostic]
    summary = {"method": method, "reward": package["reward"], "seed": seed,
               "steps": steps, "actor_steps": (steps + actor_every - 1) // actor_every,
               "transitions": len(rows), "roots": sorted({r["root"] for r in rows}),
               "wall_seconds": time.monotonic() - started,
               "diagnostic_joint_action_match": float(np.mean(match)),
               "controller_sha256": file_sha(output / "controller.pt"),
               "scope": "offline fit only; no claim of closed-loop improvement"}
    save_json(output / "summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", choices=("bc", "iql"), default="iql")
    parser.add_argument("--steps", type=int, default=1200)
    args = parser.parse_args()
    train(args.dataset, args.output, args.method, args.steps)

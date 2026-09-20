"""Instrument the existing learner; keep observations and actions unchanged."""
import copy
import time
import numpy as np
import torch
from common import save_json
from environment import Environment, hash_tree, cost_difference
from lora_native import NativeLearner, cpu_tree
from optimizer import GatedAdamW
from metrics import effective_distance


class LoggedAdamW(GatedAdamW):
    @torch.no_grad()
    def step(self, multipliers):
        # Same order and arithmetic as v14; record actual post-gating movements.
        m = torch.as_tensor(multipliers).detach().cpu().flatten()
        assert len(m) == len(self.groups) and torch.isfinite(m).all()
        assert ((m >= 0) & (m <= 1)).all()
        self.active_parameters = sum(p.grad is not None for p in self.parameters)
        previous = [[p.detach().clone() for p in group] for group in self.groups]
        self.optimizer.step()
        relative, actual, changed = [], [], []
        for index, (group, old) in enumerate(zip(self.groups, previous)):
            ds, ws, actual_ds = [], [], []
            scale = float(m[index])
            for p, before in zip(group, old):
                delta = p - before
                ds.append(delta.square().sum()); ws.append(before.square().sum())
                if scale == 0:
                    p.copy_(before)
                elif scale != 1:
                    p.copy_(before + scale * delta)
                actual_ds.append((p - before).square().sum())
            relative.append((torch.stack(ds).sum() / (torch.stack(ws).sum() + 1e-20)).sqrt())
            actual.append(torch.stack(actual_ds).sum())
        self.relative = torch.stack(relative).cpu()
        actual = torch.stack(actual).cpu()
        self.last_displacement = {"factor_step_sq_per_expert": actual.tolist(),
                                  "factor_step_sq": float(actual.sum()),
                                  "changed_experts": int((actual > 0).sum())}


class LoggedLearner(NativeLearner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optim = LoggedAdamW(self.groups, self.lr)
        self.last_train_loss = None

    def reset(self):
        super().reset()
        self.optim = LoggedAdamW(self.groups, self.lr)

    def restore(self, state):
        # Defensively detach Adam state from reusable checkpoint objects.
        super().restore(cpu_tree(state))
        self.optim.zero_grad()

    def step(self, *args, **kwargs):
        self.last_train_loss = super().step(*args, **kwargs)
        return self.last_train_loss


class LoggedEnvironment(Environment):
    def _measure(self, rows, role="monitor", routes=False):
        ll, rr = super()._measure(rows, role, routes)
        if not hasattr(self, "measurements"):
            self.measurements = []
        self.measurements.append({"role": role, "task_position": self.k,
                                  "task_step": self.step, "routes": routes,
                                  "sample_ids": [str(r.get("id", r.get("input_hash", "unknown"))) for r in rows],
                                  "target_tokens": [r["target_tokens"] for r in rows],
                                  "ce": ll})
        return ll, rr

    def _update(self, action):
        if not hasattr(self, "step_log"):
            self.step_log = []
        task, step = self.k, self.step
        before = copy.deepcopy(self.learner.cost)
        super()._update(action)
        self.step_log.append({"task_position": task, "step": step + 1,
                              "train_ce": self.learner.last_train_loss,
                              "stop_ce": self.current_loss,
                              "displacement": copy.deepcopy(self.learner.optim.last_displacement),
                              "cost": cost_difference(self.learner.cost, before)})

    def observation(self):
        fresh = self.cached_obs is None
        obs, z = super().observation()
        if fresh:
            raw = []
            # super().observation measured one batch for each old task, in order.
            measurements = self.measurements[-len(self.memory):] if self.memory else []
            for (j, memory), measured in zip(sorted(self.memory.items()), measurements):
                rows = memory["monitor"]
                start = (self.spec["seed"] * 97 + self.k * 13 + self.step * 2 + j * 31) % len(rows)
                indices = [(start + i) % len(rows) for i in range(min(self.cfg.old_observation_examples, len(rows)))]
                baseline = [memory["baseline"][i] for i in indices]
                assert [str(rows[i].get("id", rows[i].get("input_hash", "unknown"))) for i in indices] == measured["sample_ids"]
                raw.append({"task": self.spec["tasks"][j], "sample_ids": measured["sample_ids"],
                            "ce": measured["ce"], "baseline": baseline,
                            "mean_ce": float(np.mean(measured["ce"])),
                            "signed_change": float(np.mean(np.array(measured["ce"]) - baseline))})
            if not hasattr(self, "raw_boundaries"):
                self.raw_boundaries = []
            self.raw_boundaries.append({"task_position": self.k, "step": self.step,
                                        "current_ce": self.current_loss, "target": self.target(),
                                        "old_tasks": raw})
        return obs, z

    def run_logged(self, policy, anchor_weights, output, *, stochastic=False, action_seed=0,
                   behavior_name="", checkpoint_path=None):
        # Clear inherited prefix diagnostic logs; A's observations remain in its file.
        self.measurements = []; self.step_log = []; self.raw_boundaries = []
        self.learner.optim.zero_grad()
        restored = self.learner.snapshot()
        restored_hash = hash_tree({"weights": restored["weights"], "optimizer": restored["optimizer"]})
        # Recompute, do not trust the cached starting monitor value as a restore check.
        ll, _ = self._measure(self.stop_rows)
        initial_check = {"fresh_ce": float(np.mean(ll)), "cached_ce": self.current_loss,
                         "weights_optimizer_sha256": restored_hash}
        if abs(initial_check["fresh_ce"] - initial_check["cached_ce"]) > 1e-7:
            raise RuntimeError("Fresh restored monitor CE differs from saved CE")
        result = super().run(policy, stochastic=stochastic, action_seed=action_seed,
                             checkpoint_path=checkpoint_path)
        old_raw = []
        terminal_measurements = self.measurements[-(self.cfg.tasks - 1):]
        for j, measured in enumerate(terminal_measurements):
            memory = self.memory[j]
            assert len(measured["ce"]) == len(memory["baseline"])
            old_raw.append({"task": self.spec["tasks"][j], "sample_ids": measured["sample_ids"],
                            "ce": measured["ce"], "baseline": list(memory["baseline"]),
                            "mean_ce": float(np.mean(measured["ce"])),
                            "signed_change": float(np.mean(np.array(measured["ce"]) - memory["baseline"]))})
        metric_started = time.perf_counter()
        distance = effective_distance(self.learner.parameters, anchor_weights)
        metric_seconds = time.perf_counter() - metric_started
        distance["anchor"] = "same root's post-A prefix, before controlled B/C learning"
        for item in result["trace"]:
            item["behavior_logp"] = item["policy_logp"] if stochastic else 0.
        result.update(schema=1, protocol_revision="v15_offline_control",
                      behavior={"name": behavior_name, "stochastic": stochastic,
                                "epsilon": policy.epsilon, "seed": action_seed},
                      raw_terminal={"old_tasks": old_raw}, raw_boundaries=self.raw_boundaries,
                      raw_measurements=self.measurements, learner_steps=self.step_log,
                      parameter_distance=distance, parameter_metric_seconds=metric_seconds,
                      restore_check=initial_check,
                      logging_scope="unchanged v14 observations/actions; eval metrics diagnostic only, never reward")
        save_json(output, result)
        return result

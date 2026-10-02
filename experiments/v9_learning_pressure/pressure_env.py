"""Learning-pressure environment built on v8's fixed-routing MoE.

The held-out ``eval`` split is measured only for diagnostics.  In particular it
never appears in ``observe``, a chooser call, or the early-stop condition.
"""
import sys
import copy
import time
from pathlib import Path

import torch
from torch.nn import functional as F

V8 = Path(__file__).resolve().parents[1] / "v8_multilayer_retention"
if str(V8) not in sys.path:
    sys.path.insert(0, str(V8))

from environment import RetentionEnv, measure, LEVELS  # noqa: E402
from control import rule_chooser  # noqa: E402


PREFIX_TARGET = 1.55


class PressureEnv(RetentionEnv):
    """v8 learner with independent train/dev diagnostic curves.

    ``mode`` is ``early_stop`` or ``fixed_budget``.  The latter always runs the
    full budget, including when the monitor loss already meets the target.
    """
    def __init__(self, history, root, config, capture=True):
        config = dict(config)
        self.mode = config.pop("mode", "early_stop")
        if self.mode not in {"early_stop", "fixed_budget"}:
            raise ValueError("mode must be 'early_stop' or 'fixed_budget'")
        super().__init__(history, root, config, capture=capture)
        self.cost.setdefault("diagnostic_train_examples", 0)
        self.cost.setdefault("diagnostic_eval_examples", 0)
        self.cost.setdefault("diagnostic_seconds", 0.0)

    def _diagnostic_loss(self, name):
        # Timed separately so full-dataset curves are never described as
        # learner data passes.  They use no RNG and no gradient state.
        import time
        batch = self.current[name]
        start = time.perf_counter()
        values, _ = measure(self.model, batch)
        self.cost["diagnostic_%s_examples" % ("train" if name == "train" else "eval")] += len(batch)
        self.cost["diagnostic_seconds"] += time.perf_counter() - start
        return float(values.mean())

    def _start(self):
        super()._start()
        # v8's _start invokes virtual methods before this subclass has added
        # its extra cost keys, so create them lazily here.
        self.cost.setdefault("diagnostic_train_examples", 0)
        self.cost.setdefault("diagnostic_eval_examples", 0)
        self.cost.setdefault("diagnostic_seconds", 0.0)
        self.initial_train_loss = self._diagnostic_loss("train")
        self.initial_eval_loss = self._diagnostic_loss("eval")
        self.min_train_loss = self.initial_train_loss
        self.min_eval_loss = self.initial_eval_loss
        self.min_monitor_loss = self.current_loss
        self.first_hit_step = 0 if self.current_loss <= self.c["target_loss"] else None
        self.ever_hit = self.first_hit_step is not None
        self.first_hit_eval_loss = self.initial_eval_loss if self.ever_hit else None

    def _clear_frozen_grads(self):
        for parameter in self.model.parameters():
            if not parameter.requires_grad:
                parameter.grad = None

    def advance(self, multipliers):
        if self.done:
            raise RuntimeError("terminal environment")
        multipliers = multipliers.detach().clone()
        if multipliers.shape != self.last.shape or not torch.isin(multipliers, LEVELS).all():
            raise ValueError("actions must be [2, experts] with values 0, .5, 1")
        self.cost["controller_calls"] += 1
        self.previous_loss = self.current_loss
        previous_old = self.old_mean
        new, old = self.current["train"], self.memory.replay()
        for _ in range(min(self.c["interval"], self.c["max_steps"] - self.local_steps)):
            ni = torch.randint(len(new), (self.c["batch"],), generator=self.rng)
            self.optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(self.model(new.x[ni], new.task[ni]), new.y[ni])
            self.cost["train_examples"] += len(ni)
            if old is not None:
                oi = torch.randint(len(old), (self.c["batch"],), generator=self.rng)
                loss = loss + self.c["replay_weight"] * F.cross_entropy(
                    self.model(old.x[oi], old.task[oi]), old.y[oi])
                self.cost["train_examples"] += len(oi)
            loss.backward()
            # Freezing before the Adam step preserves both weights and moments.
            self._clear_frozen_grads()
            self.optimizer.step(multipliers)
            self.local_steps += 1
            self.cost["learner_updates"] += 1
        self.current_loss = self._new_loss()
        self._old_loss()
        self.old_change = self.old_mean - previous_old
        self.last = multipliers
        train_loss = self._diagnostic_loss("train")
        eval_loss = self._diagnostic_loss("eval")
        self.min_train_loss = min(self.min_train_loss, train_loss)
        self.min_eval_loss = min(self.min_eval_loss, eval_loss)
        self.min_monitor_loss = min(self.min_monitor_loss, self.current_loss)
        if not self.ever_hit and self.current_loss <= self.c["target_loss"]:
            self.ever_hit, self.first_hit_step, self.first_hit_eval_loss = True, self.local_steps, eval_loss
        if self.capture:
            self.events.append(dict(task=self.stage, steps=self.local_steps,
                new_loss=self.current_loss, train_loss=train_loss, eval_loss=eval_loss,
                memory_forgetting=self.forgetting, multipliers=multipliers.tolist()))

    def finish_task(self):
        hit = self.current_loss <= self.c["target_loss"]
        if self.mode == "fixed_budget" and self.local_steps < self.c["max_steps"]:
            raise RuntimeError("fixed_budget task must exhaust its budget")
        if self.mode == "early_stop" and not hit and self.local_steps < self.c["max_steps"]:
            raise RuntimeError("task must reach target or exhaust its budget")
        terminal_train = self._diagnostic_loss("train")
        terminal_eval = self._diagnostic_loss("eval")
        self.completed.append(dict(task=self.stage, initial_loss=self.initial_loss,
            terminal_loss=self.current_loss, success=hit, steps=self.local_steps,
            memory_forgetting=self.forgetting, initial_train_loss=self.initial_train_loss,
            initial_monitor_loss=self.initial_loss, initial_eval_loss=self.initial_eval_loss,
            min_monitor_loss=self.min_monitor_loss, terminal_monitor_loss=self.current_loss,
            min_train_loss=self.min_train_loss, min_eval_loss=self.min_eval_loss,
            terminal_train_loss=terminal_train, terminal_eval_loss=terminal_eval,
            first_hit_step=self.first_hit_step, ever_hit=self.ever_hit,
            first_hit_eval_loss=self.first_hit_eval_loss, target_loss=self.c["target_loss"]))
        refs, _ = measure(self.model, self.current["monitor"])
        self.cost["monitor_examples"] += len(refs)
        self.memory.observe_task(self.stage, self.current["train"], self.current["monitor"], refs)
        if self.capture:
            import copy
            self.snapshots.append(copy.deepcopy(self.model.state_dict()))
        self.stage += 1
        if not self.done:
            self._start()

    def run_task(self, chooser, trace=None):
        while self.local_steps < self.c["max_steps"] and (self.mode == "fixed_budget" or self.current_loss > self.c["target_loss"]):
            obs = self.observe()
            action, record = chooser(obs)
            if trace is not None and record is not None:
                trace.append(record)
            self.advance(action)
        self.finish_task()


def _set_scope(env, scope):
    if scope not in {"full", "frozen_stem", "heads_only"}:
        raise ValueError("scope must be 'full', 'frozen_stem', or 'heads_only'")
    for parameter in env.model.parameters():
        parameter.requires_grad_(True)
    if scope in {"frozen_stem", "heads_only"}:
        for parameter in env.model.stem.parameters():
            parameter.requires_grad_(False)
    if scope == "heads_only":
        for layer in env.model.layers:
            for parameter in layer.parameters():
                parameter.requires_grad_(False)
    env._clear_frozen_grads()


def _retarget_current_task(env):
    """Task 1 was initialized during the prefix under its prefix threshold."""
    env.min_train_loss = env.initial_train_loss
    env.min_eval_loss = env.initial_eval_loss
    env.min_monitor_loss = env.initial_loss
    env.first_hit_step = 0 if env.initial_loss <= env.c["target_loss"] else None
    env.ever_hit = env.first_hit_step is not None
    env.first_hit_eval_loss = env.initial_eval_loss if env.ever_hit else None


def run_prefix(env):
    """Run the shared, fully plastic task-0 checkpoint."""
    if env.stage != 0:
        raise ValueError("prefix must start at task 0")
    start = time.perf_counter()
    env.mode, env.c["target_loss"] = "early_stop", PREFIX_TARGET
    _retarget_current_task(env)
    _set_scope(env, "full")
    env.run_task(rule_chooser("global_1"))
    env.cost["seconds"] = env.cost.get("seconds", 0.0) + time.perf_counter() - start
    # Numeric snapshot lets a result harness charge each cloned branch only
    # for its suffix, while `cost.seconds` remains total logical rollout time.
    env.prefix_cost = copy.deepcopy(env.cost)
    return env


def run_suffix(env, method, target, mode, scope):
    """Continue a cloned prefix over tasks 1..N under a pressure condition."""
    if env.stage != 1:
        raise ValueError("suffix requires a completed task-0 prefix")
    if mode not in {"early_stop", "fixed_budget"}:
        raise ValueError("unknown mode")
    if not isinstance(target, (int, float)) or not torch.isfinite(torch.tensor(float(target))) or target <= 0:
        raise ValueError("target must be a finite positive number")
    if scope == "heads_only" and method != "global_0":
        raise ValueError("heads_only requires method='global_0'")
    start = time.perf_counter()
    env.mode, env.c["target_loss"] = mode, float(target)
    _retarget_current_task(env)
    _set_scope(env, scope)
    # A heads-only condition isolates representation/expert plasticity; its
    # expert multipliers are deliberately irrelevant and fixed for auditability.
    chooser = rule_chooser("global_0" if scope == "heads_only" else method)
    while not env.done:
        env.run_task(chooser)
    env.cost["seconds"] = env.cost.get("seconds", 0.0) + time.perf_counter() - start
    return env

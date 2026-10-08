"""Small, explicit runtime primitives for native-MoE feedback experiments.

The module intentionally has no model-loading code.  It operates on a supplied
``OlmoeForCausalLM`` (or a compatible module) so experiments can construct a
tiny offline configuration in tests and choose their own checkpoint source in
production.
"""

from __future__ import annotations

import copy
import hashlib
import random
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, MutableMapping, Tuple

import torch

try:  # numpy is optional for the narrow torch-only runtime use case.
    import numpy as np
except ImportError:  # pragma: no cover - exercised only in reduced installs
    np = None


_EXPERT_RE = re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\..+$")
_FROZEN_FINGERPRINT_CACHE: Dict[Tuple[int, int, int, Tuple[int, ...], str], dict] = {}


def _parameter_schema(named_parameters: Iterable[Tuple[str, torch.nn.Parameter]]) -> List[dict]:
    return [
        {"name": name, "shape": list(parameter.shape), "dtype": str(parameter.dtype)}
        for name, parameter in named_parameters
    ]


def _cpu_clone(value):
    """Deep-copy an optimizer/RNG value while keeping all tensors on CPU."""
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu").clone()
    if isinstance(value, dict):
        return {key: _cpu_clone(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_clone(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_clone(item) for item in value)
    return copy.deepcopy(value)


def _to_device(value, device: torch.device):
    if isinstance(value, torch.Tensor):
        # ``Tensor.to`` is allowed to return its input for an unchanged device;
        # restoring optimizer moments must never alias checkpoint storage.
        return value.to(device=device).clone()
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_device(item, device) for item in value)
    return copy.deepcopy(value)


def configure_trainable_experts(model: torch.nn.Module, layers: List[int], experts: List[int]) -> dict:
    """Freeze a model except for the exact Cartesian product of MoE experts.

    Parameter names must follow Hugging Face OLMoE's
    ``model.layers.N.mlp.experts.K.*`` layout.  Rejecting a partial or invalid
    selector is deliberate: a typo must never silently produce a no-op branch.
    """
    if not isinstance(layers, list) or not layers or not all(isinstance(x, int) for x in layers):
        raise ValueError("layers must be a non-empty list of integers")
    if not isinstance(experts, list) or not experts or not all(isinstance(x, int) for x in experts):
        raise ValueError("experts must be a non-empty list of integers")
    if len(set(layers)) != len(layers) or len(set(experts)) != len(experts):
        raise ValueError("layers and experts must not contain duplicates")
    if min(layers) < 0 or min(experts) < 0:
        raise ValueError("layers and experts must be non-negative")

    named = list(model.named_parameters())
    groups: Dict[Tuple[int, int], List[Tuple[str, torch.nn.Parameter]]] = {}
    for name, parameter in named:
        match = _EXPERT_RE.match(name)
        if match:
            groups.setdefault((int(match.group(1)), int(match.group(2))), []).append((name, parameter))
    if not groups:
        raise ValueError("model has no OLMoE expert parameters matching the required naming scheme")

    requested = {(layer, expert) for layer in layers for expert in experts}
    missing = sorted(requested.difference(groups))
    if missing:
        raise ValueError("invalid layer/expert selection: %s" % (missing,))

    selected = {name for pair in requested for name, _ in groups[pair]}
    for name, parameter in named:
        parameter.requires_grad_(name in selected)

    names = [name for name, _ in named if name in selected]
    trainable = sum(parameter.numel() for name, parameter in named if name in selected)
    total = sum(parameter.numel() for _, parameter in named)
    return {
        "names": names,
        "trainable_parameters": trainable,
        "total_parameters": total,
        "layers": list(layers),
        "experts": list(experts),
    }


class ExpertAdamW:
    """AdamW with FP32 master weights and an externally controlled displacement.

    The Adam moments always advance when a gradient is present.  ``multiplier``
    only scales the resulting parameter displacement; consequently multiplier
    zero is a true no-weight-update control, rather than a skipped optimizer
    step.  This distinction matters when comparing plasticity controls.
    """

    def __init__(self, named_parameters: Iterable[Tuple[str, torch.nn.Parameter]], lr: float, weight_decay: float = 0.0,
                 betas: Tuple[float, float] = (0.9, 0.999), eps: float = 1e-8):
        # Callers commonly pass model.named_parameters(); keep only the exact
        # trainable expert tensors rather than allocating FP32 masters for a
        # whole 7B model.
        items = [(name, parameter) for name, parameter in named_parameters if parameter.requires_grad]
        if not items:
            raise ValueError("ExpertAdamW requires at least one named parameter")
        if lr <= 0 or weight_decay < 0 or eps <= 0 or not (0 <= betas[0] < 1 and 0 <= betas[1] < 1):
            raise ValueError("invalid AdamW hyperparameters")
        names = [name for name, _ in items]
        if len(names) != len(set(names)):
            raise ValueError("parameter names must be unique")
        self.items = items
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.betas = tuple(float(value) for value in betas)
        self.eps = float(eps)
        self.master = {name: parameter.detach().float().clone() for name, parameter in items}
        self.state: Dict[str, dict] = {}
        self.schema = _parameter_schema(items)

    def zero_grad(self) -> None:
        for _, parameter in self.items:
            parameter.grad = None

    def step(self, multiplier: float) -> dict:
        if not isinstance(multiplier, (int, float)) or not 0.0 <= float(multiplier) <= 1.0:
            raise ValueError("multiplier must be in [0, 1]")
        multiplier = float(multiplier)
        beta1, beta2 = self.betas
        master_displacement_sq = 0.0
        model_displacement_sq = 0.0
        gradient_sq = 0.0
        parameters_with_grad = 0
        with torch.no_grad():
            for name, parameter in self.items:
                if parameter.grad is None:
                    continue
                parameters_with_grad += 1
                grad = parameter.grad.detach()
                if grad.is_sparse:
                    raise RuntimeError("ExpertAdamW does not support sparse gradients")
                master = self.master[name]
                if master.device != parameter.device:
                    master = master.to(device=parameter.device)
                    self.master[name] = master
                grad = grad.to(device=master.device, dtype=torch.float32)
                gradient_sq += float(torch.sum(grad.square()).item())
                state = self.state.setdefault(name, {
                    "step": 0,
                    "exp_avg": torch.zeros_like(master, dtype=torch.float32),
                    "exp_avg_sq": torch.zeros_like(master, dtype=torch.float32),
                })
                state["step"] += 1
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                # Form the same AdamW proposal in FP32, then interpolate from
                # the actual master value to that proposal.
                proposal = master.clone()
                if self.weight_decay:
                    proposal.mul_(1.0 - self.lr * self.weight_decay)
                bias_correction1 = 1.0 - beta1 ** state["step"]
                bias_correction2 = 1.0 - beta2 ** state["step"]
                denom = exp_avg_sq.sqrt().div_(bias_correction2 ** 0.5).add_(self.eps)
                proposal.addcdiv_(exp_avg, denom, value=-(self.lr / bias_correction1))
                if multiplier:
                    displacement = proposal.sub(master).mul_(multiplier)
                    master_displacement_sq += float(torch.sum(displacement.square()).item())
                    master.add_(displacement)
                    represented = master.to(dtype=parameter.dtype)
                    model_displacement_sq += float(torch.sum(represented.float().sub(parameter.detach().float()).square()).item())
                    parameter.copy_(represented)
        return {
            "master_displacement_l2": master_displacement_sq ** 0.5,
            "model_displacement_l2": model_displacement_sq ** 0.5,
            "parameters_with_grad": parameters_with_grad,
            "gradient_l2": gradient_sq ** 0.5,
        }

    def state_dict(self) -> dict:
        return {
            "schema": copy.deepcopy(self.schema),
            "hyperparameters": {"lr": self.lr, "weight_decay": self.weight_decay, "betas": self.betas, "eps": self.eps},
            "master": _cpu_clone(self.master),
            "state": _cpu_clone(self.state),
        }

    def load_state_dict(self, state_dict: Mapping) -> None:
        if state_dict.get("schema") != self.schema:
            raise ValueError("optimizer parameter schema does not match")
        hyper = state_dict.get("hyperparameters", {})
        expected = {"lr": self.lr, "weight_decay": self.weight_decay, "betas": self.betas, "eps": self.eps}
        if hyper != expected:
            raise ValueError("optimizer hyperparameters do not match")
        masters, state = state_dict.get("master"), state_dict.get("state")
        if set(masters or {}) != {name for name, _ in self.items} or not isinstance(state, dict):
            raise ValueError("optimizer state is incomplete")
        self.master = {}
        self.state = {}
        for name, parameter in self.items:
            master = masters[name]
            if not isinstance(master, torch.Tensor) or tuple(master.shape) != tuple(parameter.shape):
                raise ValueError("master tensor schema does not match for %s" % name)
            self.master[name] = master.to(device=parameter.device, dtype=torch.float32).clone()
            if name in state:
                item = _to_device(state[name], parameter.device)
                if set(item) != {"step", "exp_avg", "exp_avg_sq"} or tuple(item["exp_avg"].shape) != tuple(parameter.shape) or tuple(item["exp_avg_sq"].shape) != tuple(parameter.shape):
                    raise ValueError("optimizer moment schema does not match for %s" % name)
                self.state[name] = item


def _frozen_fingerprint(parameter: torch.nn.Parameter) -> dict:
    # The digest is stored, not a duplicate tensor.  Viewing as bytes also
    # works for bfloat16, which NumPy cannot represent directly.
    storage_pointer = parameter.untyped_storage().data_ptr()
    key = (id(parameter), storage_pointer, parameter._version, tuple(parameter.shape), str(parameter.dtype))
    cached = _FROZEN_FINGERPRINT_CACHE.get(key)
    if cached is not None:
        return dict(cached)
    data = parameter.detach().to(device="cpu").contiguous().view(torch.uint8).numpy().tobytes()
    fingerprint = {
        "object_id": id(parameter),
        "storage_pointer": storage_pointer,
        "version": parameter._version,
        "sha256": hashlib.sha256(data).hexdigest(),
        "shape": list(parameter.shape),
        "dtype": str(parameter.dtype),
        "device": str(parameter.device),
    }
    _FROZEN_FINGERPRINT_CACHE[key] = dict(fingerprint)
    return fingerprint


@dataclass
class LearnerCheckpoint:
    mutable_parameters: Dict[str, torch.Tensor]
    mutable_schema: List[dict]
    frozen_fingerprints: Dict[str, dict]
    optimizer_state: dict
    python_rng: object
    numpy_rng: object
    torch_rng: torch.Tensor
    cuda_rng: object
    training: bool
    stream_state: dict

    @classmethod
    def capture(cls, model: torch.nn.Module, optimizer: ExpertAdamW, stream_state: dict) -> "LearnerCheckpoint":
        named = list(model.named_parameters())
        mutable = [(name, parameter) for name, parameter in named if parameter.requires_grad]
        if not mutable:
            raise ValueError("checkpoint requires at least one mutable parameter")
        return cls(
            mutable_parameters={name: parameter.detach().to("cpu").clone() for name, parameter in mutable},
            mutable_schema=_parameter_schema(mutable),
            frozen_fingerprints={name: _frozen_fingerprint(parameter) for name, parameter in named if not parameter.requires_grad},
            optimizer_state=optimizer.state_dict(),
            python_rng=copy.deepcopy(random.getstate()),
            numpy_rng=copy.deepcopy(np.random.get_state()) if np is not None else None,
            torch_rng=torch.random.get_rng_state().to("cpu").clone(),
            cuda_rng=[state.to("cpu").clone() for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else None,
            training=bool(model.training),
            stream_state=copy.deepcopy(stream_state),
        )

    def restore(self, model: torch.nn.Module, optimizer: ExpertAdamW) -> dict:
        named = dict(model.named_parameters())
        mutable = [(name, named[name]) for name in self.mutable_parameters if name in named and named[name].requires_grad]
        if _parameter_schema(mutable) != self.mutable_schema or len(mutable) != len(self.mutable_schema):
            raise ValueError("model mutable parameter schema does not match checkpoint")
        expected_frozen = set(self.frozen_fingerprints)
        if expected_frozen != {name for name, parameter in named.items() if not parameter.requires_grad}:
            raise ValueError("model frozen parameter schema does not match checkpoint")
        for name, fingerprint in self.frozen_fingerprints.items():
            # A normal restore needs only O(1) identity, storage, schema and
            # version checks. The digest is captured for auditability, without
            # rereading frozen 7B weights on every branch restore. Deliberate
            # `.data` writes bypass PyTorch's version guard and are unsupported.
            parameter = named[name]
            if (
                id(parameter) != fingerprint["object_id"]
                or parameter.untyped_storage().data_ptr() != fingerprint["storage_pointer"]
                or list(parameter.shape) != fingerprint["shape"]
                or str(parameter.dtype) != fingerprint["dtype"]
                or str(parameter.device) != fingerprint["device"]
                or parameter._version != fingerprint["version"]
            ):
                raise RuntimeError("frozen parameter mutated since checkpoint: %s" % name)
        with torch.no_grad():
            for name, saved in self.mutable_parameters.items():
                named[name].copy_(saved.to(device=named[name].device, dtype=named[name].dtype))
        optimizer.load_state_dict(self.optimizer_state)
        random.setstate(copy.deepcopy(self.python_rng))
        if np is not None and self.numpy_rng is not None:
            np.random.set_state(copy.deepcopy(self.numpy_rng))
        torch.random.set_rng_state(self.torch_rng)
        if self.cuda_rng is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(self.cuda_rng)
        model.train(self.training)
        return copy.deepcopy(self.stream_state)

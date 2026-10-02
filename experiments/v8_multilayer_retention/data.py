"""Deterministic CIFAR-100 streams and bounded replay/monitor memory for v8.

The archive is deliberately read in-place: extracting it would make the source
state harder to audit and is unnecessary for these relatively small pickle files.
"""

from __future__ import annotations

import hashlib
import pickle
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch


CIFAR100_URL = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
CIFAR100_ARCHIVE = "cifar-100-python.tar.gz"
CIFAR100_MD5 = "eb9058c3a382ffc7106e4002c42a8d85"
# Hashes published by torchvision for the files inside the official archive.
_PICKLE_MD5 = {
    "train": "16019d7e3df5f24257cddd939b257f8d",
    "test": "f0ef6b0ae62326f3e7ffdfab6717acfc",
}
_ID_OFFSET = 1_000_000_000


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_cifar100(cache_dir: str | Path, download: bool = False) -> Path:
    """Return a checksum-verified official archive, optionally downloading it."""
    cache = Path(cache_dir)
    archive = cache / CIFAR100_ARCHIVE
    if archive.exists() and _md5(archive) == CIFAR100_MD5:
        return archive
    if archive.exists() and not download:
        raise ValueError("CIFAR-100 archive exists but its MD5 does not match the official release")
    if not archive.exists() and not download:
        raise FileNotFoundError(f"Missing {archive}; call with download=True to fetch the official archive")
    cache.mkdir(parents=True, exist_ok=True)
    temporary = archive.with_suffix(archive.suffix + ".part")
    try:
        urllib.request.urlretrieve(CIFAR100_URL, temporary)
        if _md5(temporary) != CIFAR100_MD5:
            raise ValueError("Downloaded CIFAR-100 archive failed official MD5 verification")
        temporary.replace(archive)
    finally:
        if temporary.exists():
            temporary.unlink()
    return archive


def load_dataset(cache_dir: str | Path, split: str, download: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load CIFAR-100 ``split`` from its tarball as images and fine labels."""
    if split not in _PICKLE_MD5:
        raise ValueError("split must be 'train' or 'test'")
    archive = ensure_cifar100(cache_dir, download=download)
    member_name = "cifar-100-python/" + split
    with tarfile.open(archive, "r:gz") as tar:
        member = tar.getmember(member_name)
        source = tar.extractfile(member)
        if source is None:
            raise ValueError(f"Missing {member_name} in CIFAR-100 archive")
        raw = source.read()
    if hashlib.md5(raw).hexdigest() != _PICKLE_MD5[split]:
        raise ValueError(f"{member_name} failed official pickle MD5 verification")
    payload = pickle.loads(raw, encoding="latin1")
    data = np.asarray(payload["data"], dtype=np.uint8)
    labels = np.asarray(payload["fine_labels"], dtype=np.int64)
    if data.ndim != 2 or data.shape[1] != 3 * 32 * 32 or len(data) != len(labels):
        raise ValueError("Malformed CIFAR-100 pickle payload")
    return torch.from_numpy(data.reshape(-1, 3, 32, 32).copy()), torch.from_numpy(labels.copy()).long()


@dataclass(frozen=True)
class Batch:
    x: torch.Tensor
    y: torch.Tensor
    task: torch.Tensor
    ids: torch.Tensor

    def __post_init__(self) -> None:
        n = len(self.x)
        if self.x.dtype != torch.float32 or self.x.ndim != 4 or tuple(self.x.shape[1:]) != (3, 32, 32):
            raise ValueError("x must be float32 [N, 3, 32, 32]")
        if not torch.all((self.x >= -1) & (self.x <= 1)):
            raise ValueError("x must be normalized to [-1, 1]")
        for name in ("y", "task", "ids"):
            value = getattr(self, name)
            if value.dtype != torch.long or value.ndim != 1 or len(value) != n:
                raise ValueError(f"{name} must be long [N]")
        if len(torch.unique(self.ids)) != n:
            raise ValueError("Batch ids must be unique")

    def __len__(self) -> int:
        return len(self.x)

    def subset(self, indices: torch.Tensor | Sequence[int]) -> "Batch":
        idx = torch.as_tensor(indices, dtype=torch.long)
        return Batch(self.x[idx], self.y[idx], self.task[idx], self.ids[idx])

    @staticmethod
    def concat(*batches: Optional["Batch"]) -> Optional["Batch"]:
        valid = [b for b in batches if b is not None and len(b)]
        if not valid:
            return None
        ids = torch.cat([b.ids for b in valid])
        if len(torch.unique(ids)) != len(ids):
            raise ValueError("Cannot concatenate batches with duplicate ids")
        return Batch(torch.cat([b.x for b in valid]), torch.cat([b.y for b in valid]),
                     torch.cat([b.task for b in valid]), ids)


def class_partition(seed: int = 8800) -> Dict[str, List[int]]:
    """Create a reproducible, non-overlapping 50/25/25 fine-label split."""
    generator = np.random.default_rng(seed)
    classes = generator.permutation(100).tolist()
    return {"train": classes[:50], "dev": classes[50:75], "audit": classes[75:]}


def _batch(images: torch.Tensor, labels: torch.Tensor, indices: torch.Tensor, task: int,
           class_to_local: Dict[int, int], offset: int = 0) -> Batch:
    x = images[indices].to(torch.float32).div_(127.5).sub_(1.0)
    original = labels[indices].tolist()
    try:
        local_y = torch.tensor([class_to_local[int(label)] for label in original], dtype=torch.long)
    except KeyError as exc:
        raise ValueError(f"label {exc.args[0]} is absent from this task's classes") from exc
    return Batch(x, local_y, torch.full((len(indices),), task, dtype=torch.long),
                 indices.long() + offset)


def _interleave_class_parts(parts: Sequence[torch.Tensor]) -> torch.Tensor:
    """Round-robin equal-size class samples so every prefix stays balanced."""
    if not parts:
        return torch.empty(0, dtype=torch.long)
    lengths = {len(part) for part in parts}
    if len(lengths) != 1:
        raise ValueError("class parts must have equal lengths for round-robin sampling")
    return torch.stack(tuple(parts), dim=1).reshape(-1)


def make_history(images: torch.Tensor, labels: torch.Tensor, classes: Sequence[int], root: int,
                 tasks: int = 5, classes_per_task: int = 5, train_per_class: int = 32,
                 monitor_per_class: int = 16, eval_per_class: int = 32) -> List[dict]:
    """Sample disjoint train/monitor/dev-eval examples for a task history.

    ``eval`` is held out from the official training split and is suitable for
    development selection only.  The official test split remains audit-only.
    """
    if len(classes) != tasks * classes_per_task:
        raise ValueError("classes must contain exactly tasks * classes_per_task distinct labels")
    if len(set(classes)) != len(classes):
        raise ValueError("classes must be unique")
    generator = torch.Generator().manual_seed(root)
    result = []
    for task_idx in range(tasks):
        group = list(classes[task_idx * classes_per_task:(task_idx + 1) * classes_per_task])
        class_to_local = {cls: local for local, cls in enumerate(group)}
        train_parts, monitor_parts, eval_parts = [], [], []
        for cls in group:
            available = torch.where(labels == cls)[0]
            required = train_per_class + monitor_per_class + eval_per_class
            if len(available) < required:
                raise ValueError(f"class {cls} has only {len(available)} examples, need {required}")
            chosen = available[torch.randperm(len(available), generator=generator)[:required]]
            train_parts.append(chosen[:train_per_class])
            monitor_parts.append(chosen[train_per_class:train_per_class + monitor_per_class])
            eval_parts.append(chosen[train_per_class + monitor_per_class:])
        result.append({"classes": group,
                       "train": _batch(images, labels, _interleave_class_parts(train_parts), task_idx, class_to_local),
                       "monitor": _batch(images, labels, _interleave_class_parts(monitor_parts), task_idx, class_to_local),
                       "eval": _batch(images, labels, _interleave_class_parts(eval_parts), task_idx, class_to_local)})
    return result


def build_audit(test_images: torch.Tensor, test_labels: torch.Tensor, task_classes: Sequence[Sequence[int]],
                per_class: int = 100) -> List[Batch]:
    """Build an offline-only per-task test set with IDs separate from training IDs."""
    output = []
    for task_idx, classes in enumerate(task_classes):
        class_to_local = {cls: local for local, cls in enumerate(classes)}
        parts = []
        for cls in classes:
            available = torch.where(test_labels == cls)[0]
            if len(available) < per_class:
                raise ValueError(f"audit class {cls} has fewer than {per_class} test examples")
            parts.append(available[:per_class])
        output.append(_batch(test_images, test_labels, torch.cat(parts), task_idx, class_to_local, _ID_OFFSET))
    return output


class FixedMemory:
    """A bounded, task-balanced memory that never revives discarded samples."""
    def __init__(self, replay_capacity: int, monitor_capacity: int):
        if replay_capacity < 0 or monitor_capacity < 0:
            raise ValueError("capacities must be non-negative")
        self.replay_capacity = replay_capacity
        self.monitor_capacity = monitor_capacity
        self._replay: Dict[int, Batch] = {}
        self._monitor: Dict[int, Batch] = {}
        self._refs: Dict[int, torch.Tensor] = {}

    @staticmethod
    def _quotas(capacity: int, keys: Iterable[int]) -> Dict[int, int]:
        ordered = sorted(keys)
        q, remainder = divmod(capacity, len(ordered))
        return {key: q + int(pos < remainder) for pos, key in enumerate(ordered)}

    @staticmethod
    def _trim(batch: Batch, n: int) -> Batch:
        return batch.subset(torch.arange(min(n, len(batch))))

    def _rebalance(self) -> None:
        replay_quota = self._quotas(self.replay_capacity, self._replay) if self._replay else {}
        monitor_quota = self._quotas(self.monitor_capacity, self._monitor) if self._monitor else {}
        for key, quota in replay_quota.items():
            self._replay[key] = self._trim(self._replay[key], quota)
        for key, quota in monitor_quota.items():
            old = self._monitor[key]
            keep = min(quota, len(old))
            self._monitor[key] = self._trim(old, keep)
            self._refs[key] = self._refs[key][:keep]

    def observe_task(self, task_idx: int, train: Batch, monitor: Batch,
                     monitor_losses_per_sample: torch.Tensor | Sequence[float]) -> None:
        if task_idx in self._replay or task_idx in self._monitor:
            raise ValueError("A task may be observed only once")
        if torch.any(train.task != task_idx) or torch.any(monitor.task != task_idx):
            raise ValueError("Batch task field must match task_idx")
        if set(train.ids.tolist()) & set(monitor.ids.tolist()):
            raise ValueError("train and monitor examples must be disjoint")
        refs = torch.as_tensor(monitor_losses_per_sample, dtype=torch.float32).flatten()
        if len(refs) != len(monitor):
            raise ValueError("monitor_losses_per_sample must have one value per monitor example")
        self._replay[task_idx] = train
        self._monitor[task_idx] = monitor
        self._refs[task_idx] = refs.clone()
        self._rebalance()

    def replay(self) -> Optional[Batch]:
        return Batch.concat(*(self._replay[k] for k in sorted(self._replay)))

    def monitor(self) -> Tuple[Optional[Batch], Optional[torch.Tensor]]:
        batch = Batch.concat(*(self._monitor[k] for k in sorted(self._monitor)))
        if batch is None:
            return None, None
        return batch, torch.cat([self._refs[k] for k in sorted(self._refs)])

    @property
    def retained_replay_ids(self) -> torch.Tensor:
        batch = self.replay()
        return torch.empty(0, dtype=torch.long) if batch is None else batch.ids

    @property
    def retained_monitor_ids(self) -> torch.Tensor:
        batch, _ = self.monitor()
        return torch.empty(0, dtype=torch.long) if batch is None else batch.ids

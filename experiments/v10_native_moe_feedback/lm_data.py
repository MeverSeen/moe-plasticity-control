"""Deterministic, document-isolated data utilities for v10 causal-LM runs.

This module deliberately keeps documents intact while assigning splits.  The
final test documents may be inspected by preparation code only to compute
their identifiers and split manifest; training code should receive only the
already-selected train/monitor/selection records.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import time
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import torch
import torch.nn.functional as F

try:
    import numpy as np
except ImportError:  # pragma: no cover - only relevant in reduced installations
    np = None


DOMAINS = ("A", "B")
ROLES = ("train", "monitor", "selection", "test")
DEFAULT_RATIOS = (0.7, 0.1, 0.1, 0.1)


def _normalized_text(value: str) -> str:
    """Normalize only representation noise before exact-content de-duplication."""
    return " ".join(unicodedata.normalize("NFC", value).split())


def _canonical_record(raw: Mapping[str, Any], line_number: int) -> Dict[str, str]:
    required = ("id", "source", "text", "domain")
    missing = [key for key in required if key not in raw]
    if missing:
        raise ValueError(f"JSONL line {line_number} is missing required fields: {', '.join(missing)}")
    record = {key: raw[key] for key in required}
    if not all(isinstance(record[key], str) for key in required):
        raise ValueError(f"JSONL line {line_number} must use string id/source/text/domain fields")
    if not record["id"].strip() or not record["source"].strip():
        raise ValueError(f"JSONL line {line_number} has an empty id or source")
    if record["domain"] not in DOMAINS:
        raise ValueError(f"JSONL line {line_number} has domain {record['domain']!r}; expected A or B")
    if not _normalized_text(record["text"]):
        raise ValueError(f"JSONL line {line_number} has empty normalized text")
    return record


def _validate_and_deduplicate(records: Iterable[Mapping[str, Any]]) -> List[Dict[str, str]]:
    """Validate records and retain one deterministic representative per text.

    Same-id repeats are accepted only when their semantic fields agree.  Exact
    normalized content occurring in both domains is a hard error because it
    would leak the requested A/B comparison.  Repeated content inside one
    domain is de-duplicated before source-based partitioning.
    """
    by_id: Dict[str, Tuple[str, str, str]] = {}
    by_content: Dict[str, Tuple[str, str]] = {}
    retained: List[Dict[str, str]] = []
    for line_number, raw in enumerate(records, start=1):
        record = _canonical_record(raw, line_number)
        semantic = (record["source"], _normalized_text(record["text"]), record["domain"])
        previous = by_id.get(record["id"])
        if previous is not None:
            if previous != semantic:
                raise ValueError(f"id {record['id']!r} has conflicting source, text, or domain")
            continue
        by_id[record["id"]] = semantic
        content_key = hashlib.sha256(semantic[1].encode("utf-8")).hexdigest()
        content_previous = by_content.get(content_key)
        if content_previous is not None:
            previous_domain, previous_id = content_previous
            if previous_domain != record["domain"]:
                raise ValueError(
                    f"normalized text is duplicated across domains ({previous_id!r}, {record['id']!r})"
                )
            # Keep the first record from the input.  Its source is the source
            # family that owns this content for partitioning purposes.
            continue
        by_content[content_key] = (record["domain"], record["id"])
        retained.append(record)
    if not retained:
        raise ValueError("No valid, non-duplicate documents were supplied")
    return retained


def load_documents(path: str | Path) -> List[Dict[str, str]]:
    """Read and validate a UTF-8 JSONL corpus without exposing a text manifest."""
    source = Path(path)
    rows: List[Mapping[str, Any]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"JSONL line {line_number} is blank")
            try:
                loaded = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}") from exc
            if not isinstance(loaded, dict):
                raise ValueError(f"JSONL line {line_number} must be an object")
            rows.append(loaded)
    return _validate_and_deduplicate(rows)


def _check_ratios(ratios: Sequence[float]) -> Tuple[float, float, float, float]:
    if len(ratios) != len(ROLES):
        raise ValueError("ratios must have four values for train/monitor/selection/test")
    converted = tuple(float(value) for value in ratios)
    if any(not math.isfinite(value) or value <= 0.0 for value in converted):
        raise ValueError("ratios must be finite and strictly positive")
    if not math.isclose(sum(converted), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("ratios must sum to 1.0")
    return converted  # type: ignore[return-value]


def _role_for_source(source: str, seed: int, ratios: Tuple[float, float, float, float]) -> str:
    # A stable cryptographic hash avoids Python's process-randomized hash().
    digest = hashlib.sha256(f"v10-split\0{seed}\0{source}".encode("utf-8")).digest()
    unit = int.from_bytes(digest[:8], "big") / float(1 << 64)
    cutoff = 0.0
    for role, ratio in zip(ROLES, ratios):
        cutoff += ratio
        if unit < cutoff:
            return role
    return ROLES[-1]  # Protect against any floating point edge at exactly 1.


def partition_documents(
    records: Iterable[Mapping[str, Any]], seed: int, ratios: Sequence[float] = DEFAULT_RATIOS
) -> Dict[str, Dict[str, List[Dict[str, str]]]]:
    """Partition de-duplicated records by a deterministic whole-source hash.

    A source family receives one role globally, even if it supplies documents
    to both domains.  This is stronger than splitting separately by domain and
    prevents source-family leakage across any later A/B comparison.
    """
    checked = _validate_and_deduplicate(records)
    checked_ratios = _check_ratios(ratios)
    output: Dict[str, Dict[str, List[Dict[str, str]]]] = {
        domain: {role: [] for role in ROLES} for domain in DOMAINS
    }
    for record in checked:
        role = _role_for_source(record["source"], int(seed), checked_ratios)
        output[record["domain"]][role].append(dict(record))
    for domain in DOMAINS:
        for role in ROLES:
            output[domain][role].sort(key=lambda item: item["id"])
    return output


def split_manifest(splits: Mapping[str, Mapping[str, Sequence[Mapping[str, str]]]], seed: int,
                   ratios: Sequence[float] = DEFAULT_RATIOS) -> Dict[str, Any]:
    """Return auditable IDs/digests only; never include document text."""
    checked_ratios = _check_ratios(ratios)
    groups: Dict[str, Any] = {}
    all_ids: List[str] = []
    for domain in DOMAINS:
        if domain not in splits:
            raise ValueError(f"Missing domain {domain!r} from splits")
        groups[domain] = {}
        for role in ROLES:
            rows = list(splits[domain].get(role, ()))
            ids = [str(row["id"]) for row in rows]
            sources = sorted({str(row["source"]) for row in rows})
            if len(ids) != len(set(ids)):
                raise ValueError(f"Duplicate IDs in {domain}/{role}")
            all_ids.extend(ids)
            groups[domain][role] = {
                "count": len(ids),
                "ids": ids,
                "id_digest": hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest(),
                "source_digest": hashlib.sha256("\n".join(sources).encode("utf-8")).hexdigest(),
            }
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("An ID occurs in multiple split roles")
    return {"seed": int(seed), "ratios": list(checked_ratios), "domains": groups,
            "all_ids_digest": hashlib.sha256("\n".join(sorted(all_ids)).encode("utf-8")).hexdigest()}


def prepare_documents(path: str | Path, seed: int = 0,
                      ratios: Sequence[float] = DEFAULT_RATIOS) -> Dict[str, Dict[str, List[Dict[str, str]]]]:
    """Load and partition a corpus.  Use :func:`split_manifest` separately."""
    return partition_documents(load_documents(path), seed=seed, ratios=ratios)


def _encode(tokenizer: Any, text: str) -> List[int]:
    if hasattr(tokenizer, "encode"):
        encoded = tokenizer.encode(text, add_special_tokens=False)
    else:
        encoded = tokenizer(text, add_special_tokens=False)["input_ids"]
    if isinstance(encoded, torch.Tensor):
        encoded = encoded.tolist()
    if encoded and isinstance(encoded[0], list):
        if len(encoded) != 1:
            raise ValueError("tokenizer returned more than one sequence for one document")
        encoded = encoded[0]
    if not isinstance(encoded, (list, tuple)) or not all(isinstance(token, int) for token in encoded):
        raise ValueError("tokenizer must return a sequence of integer token IDs")
    return [int(token) for token in encoded]


def tokenize_documents(records: Iterable[Mapping[str, str]], tokenizer: Any,
                       sequence_length: int) -> List[Dict[str, Any]]:
    """Tokenize each document independently, appending EOS and never joining docs."""
    if sequence_length < 2:
        raise ValueError("sequence_length must be at least 2")
    eos = getattr(tokenizer, "eos_token_id", None)
    if not isinstance(eos, int):
        raise ValueError("tokenizer must define an integer eos_token_id")
    blocks: List[Dict[str, Any]] = []
    for raw in records:
        record = _canonical_record(raw, 1)
        tokens = _encode(tokenizer, record["text"]) + [int(eos)]
        # A one-token fragment carries no next-token target and is excluded.
        for block_index, start in enumerate(range(0, len(tokens), sequence_length)):
            piece = tokens[start:start + sequence_length]
            if len(piece) < 2:
                continue
            blocks.append({"input_ids": torch.tensor(piece, dtype=torch.long),
                           "document_id": record["id"], "source": record["source"],
                           "domain": record["domain"], "block_index": block_index})
    if not blocks:
        raise ValueError("No document supplied at least two token IDs after EOS")
    return blocks


class BatchStream:
    """A deterministic replacement sampler with recoverable local RNG state.

    Contract: blocks are fixed CPU tensors with at least two tokens.  Every
    call returns padded ``input_ids``, boolean ``attention_mask``, and labels
    set to ``-100`` at padding positions.  Sampling uses only this instance's
    generator; it never advances PyTorch's global RNG.
    """

    def __init__(self, blocks: Sequence[Mapping[str, Any]], seed: int, pad_token_id: int = 0):
        if not blocks:
            raise ValueError("BatchStream needs at least one block")
        if not isinstance(pad_token_id, int):
            raise ValueError("pad_token_id must be an integer")
        self.blocks: List[Dict[str, Any]] = []
        for position, block in enumerate(blocks):
            if "input_ids" not in block:
                raise ValueError(f"block {position} has no input_ids")
            ids = torch.as_tensor(block["input_ids"], dtype=torch.long, device="cpu").flatten().clone()
            if ids.numel() < 2:
                raise ValueError(f"block {position} must contain at least two tokens")
            copied = dict(block)
            copied["input_ids"] = ids
            self.blocks.append(copied)
        self.pad_token_id = int(pad_token_id)
        self._generator = torch.Generator(device="cpu").manual_seed(int(seed))
        self._order = torch.randperm(len(self.blocks), generator=self._generator).tolist()
        self._cursor = 0

    def _blocks_digest(self) -> str:
        """Fingerprint sampling-relevant content and ordering for safe resume."""
        digest = hashlib.sha256()
        for position, block in enumerate(self.blocks):
            ids = block["input_ids"].contiguous().to(device="cpu", dtype=torch.long)
            digest.update(position.to_bytes(8, "big", signed=False))
            digest.update(ids.numel().to_bytes(8, "big", signed=False))
            digest.update(ids.numpy().tobytes())
        return digest.hexdigest()

    def _next_index(self) -> int:
        if self._cursor >= len(self._order):
            self._order = torch.randperm(len(self.blocks), generator=self._generator).tolist()
            self._cursor = 0
        value = self._order[self._cursor]
        self._cursor += 1
        return int(value)

    def next_batch(self, batch_size: int, device: torch.device | str = "cpu") -> Dict[str, torch.Tensor]:
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        selected = [self.blocks[self._next_index()]["input_ids"] for _ in range(batch_size)]
        width = max(int(ids.numel()) for ids in selected)
        input_ids = torch.full((batch_size, width), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, width), dtype=torch.bool)
        for row, ids in enumerate(selected):
            input_ids[row, :ids.numel()] = ids
            attention_mask[row, :ids.numel()] = True
        labels = input_ids.clone()
        labels[~attention_mask] = -100
        target = torch.device(device)
        return {"input_ids": input_ids.to(target), "attention_mask": attention_mask.to(target),
                "labels": labels.to(target)}

    def state_dict(self) -> Dict[str, Any]:
        return {"cursor": self._cursor, "order": list(self._order),
                "generator_state": self._generator.get_state().clone(),
                "blocks": len(self.blocks), "blocks_digest": self._blocks_digest(),
                "pad_token_id": self.pad_token_id}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if int(state.get("blocks", -1)) != len(self.blocks):
            raise ValueError("BatchStream state was created for a different number of blocks")
        if state.get("blocks_digest") != self._blocks_digest():
            raise ValueError("BatchStream state was created for different block content or ordering")
        if int(state.get("pad_token_id", self.pad_token_id)) != self.pad_token_id:
            raise ValueError("BatchStream state has a different pad_token_id")
        order = list(state.get("order", ()))
        cursor = int(state.get("cursor", -1))
        if sorted(order) != list(range(len(self.blocks))) or not 0 <= cursor <= len(order):
            raise ValueError("Invalid BatchStream order or cursor")
        generator_state = state.get("generator_state")
        if not isinstance(generator_state, torch.Tensor):
            raise ValueError("BatchStream state has no generator_state tensor")
        self._order = [int(index) for index in order]
        self._cursor = cursor
        self._generator.set_state(generator_state.cpu())


def _logits_from_output(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, Mapping) and "logits" in output:
        return output["logits"]
    logits = getattr(output, "logits", None)
    if logits is None:
        raise ValueError("model output has no logits; evaluate_lm never uses output.loss")
    return logits


def _rng_state() -> Dict[str, Any]:
    """Capture every process-global RNG used by the experiment runtime."""
    return {
        "python": copy.deepcopy(random.getstate()),
        "numpy": copy.deepcopy(np.random.get_state()) if np is not None else None,
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    if np is not None and state["numpy"] is not None:
        np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def evaluate_lm(model: Any, batches: Iterable[Mapping[str, torch.Tensor]]) -> Dict[str, float | int]:
    """Compute token-weighted next-token CE without retaining logits or using loss."""
    prior_training = bool(model.training)
    prior_rng = _rng_state()
    started = None
    timed_cuda_devices = set()
    total_loss = 0.0
    total_tokens = 0
    total_examples = 0
    try:
        model.eval()
        with torch.no_grad():
            for batch in batches:
                input_ids = batch["input_ids"]
                attention_mask = batch.get("attention_mask")
                labels = batch.get("labels", input_ids)
                if started is None:
                    if input_ids.is_cuda:
                        torch.cuda.synchronize(input_ids.device)
                        timed_cuda_devices.add(input_ids.device)
                    started = time.perf_counter()
                if input_ids.ndim != 2 or labels.shape != input_ids.shape:
                    raise ValueError("input_ids and labels must be matching [batch, time] tensors")
                if input_ids.shape[1] < 2:
                    continue
                output = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = _logits_from_output(output)
                if logits.is_cuda:
                    timed_cuda_devices.add(logits.device)
                if logits.ndim != 3 or tuple(logits.shape[:2]) != tuple(input_ids.shape):
                    raise ValueError("model logits must be [batch, time, vocabulary]")
                shifted_labels = labels[:, 1:].to(device=logits.device, dtype=torch.long).clone()
                if attention_mask is not None:
                    shifted_labels[~attention_mask[:, 1:].to(logits.device, dtype=torch.bool)] = -100
                flat_labels = shifted_labels.reshape(-1)
                token_count = int((flat_labels != -100).sum().item())
                if not token_count:
                    continue
                # Native-MoE inference commonly emits bfloat16 logits.  CE
                # accumulates a small branch difference, so do the reduction
                # explicitly in FP32 before converting the scalar to Python.
                fp32_logits = logits[:, :-1, :].float()
                loss_sum = F.cross_entropy(fp32_logits.reshape(-1, fp32_logits.shape[-1]), flat_labels,
                                           ignore_index=-100, reduction="sum")
                total_loss += float(loss_sum.item())
                total_tokens += token_count
                total_examples += int(input_ids.shape[0])
    finally:
        model.train(prior_training)
        _restore_rng(prior_rng)
    if total_tokens == 0:
        raise ValueError("evaluate_lm received no unpadded next-token labels")
    for device in timed_cuda_devices:
        torch.cuda.synchronize(device)
    if started is None:  # Kept for type checkers; the token check above raises first.
        started = time.perf_counter()
    return {"loss": total_loss / total_tokens, "tokens": total_tokens, "examples": total_examples,
            "seconds": time.perf_counter() - started}

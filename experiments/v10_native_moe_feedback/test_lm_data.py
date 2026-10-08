import json
import math
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from lm_data import (BatchStream, evaluate_lm, load_documents, partition_documents,
                     prepare_documents, split_manifest, tokenize_documents)


class TinyTokenizer:
    eos_token_id = 99
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [ord(char) % 17 + 1 for char in text]


def record(identifier, source, text, domain="A"):
    return {"id": identifier, "source": source, "text": text, "domain": domain}


class TestDocumentSplits(unittest.TestCase):
    def test_source_groups_are_disjoint_and_deduplicated_before_partition(self):
        rows = [
            record("a1", "book-a", "alpha", "A"),
            record("a2", "book-a", "beta", "A"),
            record("b1", "book-a", "gamma", "B"),
            record("a3", "book-b", "alpha", "A"),  # same-domain duplicate: discarded
            record("b2", "book-c", "delta", "B"),
        ]
        first = partition_documents(rows, seed=41)
        second = partition_documents(rows, seed=41)
        self.assertEqual(first, second)
        locations = {}
        ids = set()
        for domain, roles in first.items():
            for role, documents in roles.items():
                for document in documents:
                    self.assertNotIn(document["id"], ids)
                    ids.add(document["id"])
                    locations.setdefault(document["source"], set()).add(role)
        self.assertEqual({"a1", "a2", "b1", "b2"}, ids)
        self.assertTrue(all(len(roles) == 1 for roles in locations.values()))

    def test_conflicting_ids_and_cross_domain_content_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "conflicting"):
            partition_documents([record("same", "s", "one"), record("same", "s", "two")], seed=1)
        with self.assertRaisesRegex(ValueError, "across domains"):
            partition_documents([record("a", "s", "same", "A"), record("b", "t", "same", "B")], seed=1)

    def test_jsonl_prepare_and_manifest_do_not_publish_text(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "corpus.jsonl"
            rows = [record("a", "source-1", "not for manifest", "A"),
                    record("b", "source-2", "also private", "B")]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            loaded = load_documents(path)
            splits = prepare_documents(path, seed=7)
        self.assertEqual(rows, loaded)
        manifest = split_manifest(splits, seed=7)
        rendered = json.dumps(manifest, sort_keys=True)
        self.assertNotIn("not for manifest", rendered)
        self.assertNotIn("also private", rendered)
        self.assertIn("a", rendered)
        self.assertIn("b", rendered)


class TestTokenizationAndStream(unittest.TestCase):
    def test_document_tokenization_never_crosses_boundaries(self):
        tokenizer = TinyTokenizer()
        blocks = tokenize_documents([record("one", "s1", "abcd"), record("two", "s2", "xy")], tokenizer, 3)
        by_document = {}
        for block in blocks:
            by_document.setdefault(block["document_id"], []).append(block["input_ids"].tolist())
        self.assertEqual([[13, 14, 15], [16, 99]], by_document["one"])
        self.assertEqual([[2, 3, 99]], by_document["two"])
        self.assertTrue(all(len(block["input_ids"]) >= 2 for block in blocks))

    def test_batch_padding_and_state_restore_are_deterministic_without_global_rng(self):
        tokenizer = TinyTokenizer()
        blocks = tokenize_documents([record("one", "s1", "a"), record("two", "s2", "abcd")], tokenizer, 8)
        torch.manual_seed(123)
        before = torch.get_rng_state().clone()
        stream = BatchStream(blocks, seed=9, pad_token_id=tokenizer.pad_token_id)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        first = stream.next_batch(2)
        self.assertTrue(torch.equal(first["labels"][~first["attention_mask"]], torch.full_like(first["labels"][~first["attention_mask"]], -100)))
        state = stream.state_dict()
        expected = stream.next_batch(3)
        restored = BatchStream(blocks, seed=111, pad_token_id=tokenizer.pad_token_id)
        restored.load_state_dict(state)
        actual = restored.next_batch(3)
        for key in ("input_ids", "attention_mask", "labels"):
            self.assertTrue(torch.equal(expected[key], actual[key]))

    def test_state_refuses_same_count_but_different_block_content_or_order(self):
        tokenizer = TinyTokenizer()
        original = tokenize_documents([record("one", "s1", "ab"), record("two", "s2", "cd")], tokenizer, 8)
        changed = tokenize_documents([record("one", "s1", "az"), record("two", "s2", "cd")], tokenizer, 8)
        saved = BatchStream(original, seed=9).state_dict()
        with self.assertRaisesRegex(ValueError, "content or ordering"):
            BatchStream(changed, seed=9).load_state_dict(saved)
        with self.assertRaisesRegex(ValueError, "content or ordering"):
            BatchStream(list(reversed(original)), seed=9).load_state_dict(saved)


class Output:
    def __init__(self, logits):
        self.logits = logits
        self.loss = torch.tensor(999.0)


class UniformModel(torch.nn.Module):
    def __init__(self, vocabulary=7):
        super().__init__()
        self.vocabulary = vocabulary
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def forward(self, input_ids, attention_mask=None):
        logits = torch.zeros((*input_ids.shape, self.vocabulary), device=input_ids.device) + self.anchor
        return Output(logits)


class RNGConsumingModel(UniformModel):
    def forward(self, input_ids, attention_mask=None):
        random.random()
        np.random.random()
        torch.rand(1, device=input_ids.device)
        return super().forward(input_ids, attention_mask)


class BF16Model(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.register_buffer("stored", logits)

    def forward(self, input_ids, attention_mask=None):
        return Output(self.stored.expand(input_ids.shape[0], -1, -1).to(torch.bfloat16))


class TestEvaluation(unittest.TestCase):
    def test_evaluation_is_token_weighted_ignores_padding_and_preserves_state_rng(self):
        model = UniformModel()
        model.train()
        batch = {
            "input_ids": torch.tensor([[1, 2, 0], [3, 4, 5]]),
            "attention_mask": torch.tensor([[True, True, False], [True, True, True]]),
            "labels": torch.tensor([[1, 2, -100], [3, 4, 5]]),
        }
        torch.manual_seed(567)
        before = torch.get_rng_state().clone()
        result = evaluate_lm(model, [batch])
        self.assertTrue(model.training)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertEqual(3, result["tokens"])
        self.assertEqual(2, result["examples"])
        self.assertAlmostEqual(math.log(7), result["loss"], places=6)
        self.assertLess(result["seconds"], 10.0)

    def test_evaluation_uses_shifted_labels_not_model_auxiliary_loss(self):
        model = UniformModel(vocabulary=5)
        batch = {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.tensor([[True, True, True]]),
                 "labels": torch.tensor([[1, 2, 3]])}
        result = evaluate_lm(model, [batch])
        self.assertAlmostEqual(math.log(5), result["loss"], places=6)
        self.assertNotEqual(999.0, result["loss"])

    def test_evaluation_restores_python_numpy_and_torch_rng_consumed_by_model(self):
        model = RNGConsumingModel(vocabulary=5)
        batch = {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.tensor([[True, True, True]]),
                 "labels": torch.tensor([[1, 2, 3]])}
        random.seed(123)
        np.random.seed(456)
        torch.manual_seed(789)
        python_before, numpy_before, torch_before = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
        python_expected = random.random()
        numpy_expected = np.random.random()
        torch_expected = torch.rand(1)
        random.setstate(python_before)
        np.random.set_state(numpy_before)
        torch.set_rng_state(torch_before)
        evaluate_lm(model, [batch])
        self.assertEqual(python_expected, random.random())
        self.assertEqual(numpy_expected, np.random.random())
        self.assertTrue(torch.equal(torch_expected, torch.rand(1)))

    def test_bfloat16_logits_are_reduced_in_fp32(self):
        logits = torch.tensor([[[0.125, -1.75, 2.375, 0.5],
                                [1.625, -0.75, 0.25, 3.125],
                                [-2.5, 1.5, 0.875, -0.125]]], dtype=torch.float32)
        model = BF16Model(logits)
        batch = {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.tensor([[True, True, True]]),
                 "labels": torch.tensor([[1, 2, 3]])}
        expected_logits = logits.to(torch.bfloat16).float()[:, :-1, :].reshape(-1, 4)
        expected = F.cross_entropy(expected_logits, torch.tensor([2, 3]), reduction="mean").item()
        result = evaluate_lm(model, [batch])
        self.assertAlmostEqual(expected, result["loss"], places=7)


if __name__ == "__main__":
    unittest.main()

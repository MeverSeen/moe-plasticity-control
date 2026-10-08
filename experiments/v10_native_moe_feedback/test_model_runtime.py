"""Offline unit tests for the native OLMoE runtime."""

import copy
import random
import sys
import unittest
from pathlib import Path

import torch

try:
    from transformers import OlmoeConfig, OlmoeForCausalLM
except ImportError:  # pragma: no cover - normal unittest skip on minimal installs
    OlmoeConfig = OlmoeForCausalLM = None

sys.path.insert(0, str(Path(__file__).parent))
from model_runtime import ExpertAdamW, LearnerCheckpoint, configure_trainable_experts


@unittest.skipIf(OlmoeForCausalLM is None, "transformers is not installed")
class ModelRuntimeTests(unittest.TestCase):
    def tiny_olmoe(self, dtype=torch.float32):
        torch.manual_seed(7)
        config = OlmoeConfig(
            vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=32,
            num_experts=4, num_experts_per_tok=2,
        )
        return OlmoeForCausalLM(config).to(dtype=dtype)

    @staticmethod
    def selected(model):
        return [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]

    @staticmethod
    def loss(model):
        tokens = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
        return model(input_ids=tokens, labels=tokens).loss

    def test_configure_freezes_exact_huggingface_expert_scope(self):
        model = self.tiny_olmoe()
        report = configure_trainable_experts(model, [0, 1], [1])
        self.assertTrue(report["names"])
        self.assertTrue(all(".experts.1." in name and ("layers.0." in name or "layers.1." in name) for name in report["names"]))
        self.assertTrue(all(parameter.requires_grad == (name in report["names"]) for name, parameter in model.named_parameters()))
        self.assertEqual(report["trainable_parameters"], sum(parameter.numel() for _, parameter in self.selected(model)))
        self.assertEqual(report["total_parameters"], sum(parameter.numel() for parameter in model.parameters()))
        # Passing model.named_parameters() must still allocate state only for
        # selected experts, never a full-model FP32 mirror.
        optimizer = ExpertAdamW(model.named_parameters(), lr=1e-2)
        self.assertEqual(set(optimizer.master), set(report["names"]))
        self.assertEqual(optimizer.state, {})
        with self.assertRaises(ValueError):
            configure_trainable_experts(model, [99], [1])
        with self.assertRaises(ValueError):
            configure_trainable_experts(model, [], [1])

    def test_multiplier_zero_advances_moments_without_displacement_and_one_matches_adamw(self):
        model = self.tiny_olmoe()
        configure_trainable_experts(model, [0], [0])
        ours = ExpertAdamW(self.selected(model), lr=1e-2, weight_decay=0.1)
        before = {name: parameter.detach().clone() for name, parameter in self.selected(model)}
        self.loss(model).backward()
        diagnostics = ours.step(0.0)
        self.assertTrue(all(torch.equal(parameter, before[name]) for name, parameter in self.selected(model)))
        self.assertTrue(all(state["step"] == 1 for state in ours.state.values()))
        self.assertEqual(diagnostics["model_displacement_l2"], 0.0)
        self.assertGreater(diagnostics["gradient_l2"], 0.0)

        # A full FP32 step agrees with torch AdamW when both optimizers start
        # from the same model and moment state (the standard AdamW contract).
        direct = self.tiny_olmoe()
        reference = self.tiny_olmoe()
        configure_trainable_experts(direct, [0], [0])
        configure_trainable_experts(reference, [0], [0])
        direct_opt = ExpertAdamW(self.selected(direct), lr=1e-2, weight_decay=0.1)
        ref_opt = torch.optim.AdamW([parameter for _, parameter in self.selected(reference)], lr=1e-2, weight_decay=0.1)
        self.loss(direct).backward(); self.loss(reference).backward()
        direct_opt.step(1.0); ref_opt.step()
        for (name, parameter), (_, ref_parameter) in zip(self.selected(direct), self.selected(reference)):
            self.assertTrue(torch.allclose(parameter, ref_parameter, atol=2e-7, rtol=2e-6), name)

    def test_bfloat16_uses_fp32_masters_and_none_grad_is_untouched(self):
        model = self.tiny_olmoe(torch.bfloat16)
        configure_trainable_experts(model, [0], [0])
        optimizer = ExpertAdamW(self.selected(model), lr=1e-2)
        name, unused = self.selected(model)[0]
        before = unused.detach().clone()
        self.loss(model).backward()
        unused.grad = None
        optimizer.step(1.0)
        self.assertEqual(optimizer.master[name].dtype, torch.float32)
        self.assertTrue(torch.equal(unused, before))
        self.assertNotIn(name, optimizer.state)

    def test_checkpoint_repeats_branch_rng_weights_and_optimizer_state(self):
        model = self.tiny_olmoe()
        configure_trainable_experts(model, [0], [0])
        optimizer = ExpertAdamW(self.selected(model), lr=1e-2)
        checkpoint = LearnerCheckpoint.capture(model, optimizer, {"cursor": 3, "batches": [1, 2]})

        def branch():
            draw = (random.random(), torch.rand(()).item())
            optimizer.zero_grad(); self.loss(model).backward(); optimizer.step(0.5)
            return draw, {name: parameter.detach().clone() for name, parameter in self.selected(model)}, copy.deepcopy(optimizer.state_dict())

        first = branch()
        self.assertEqual(checkpoint.restore(model, optimizer), {"cursor": 3, "batches": [1, 2]})
        second = branch()
        self.assertEqual(first[0], second[0])
        for name in first[1]:
            self.assertTrue(torch.equal(first[1][name], second[1][name]))
        self.assertEqual(first[2]["state"].keys(), second[2]["state"].keys())
        for name in first[2]["state"]:
            self.assertTrue(torch.equal(first[2]["state"][name]["exp_avg"], second[2]["state"][name]["exp_avg"]))

    def test_checkpoint_with_existing_moments_is_immutable_across_restores(self):
        model = self.tiny_olmoe()
        configure_trainable_experts(model, [0], [0])
        optimizer = ExpertAdamW(model.named_parameters(), lr=1e-2)
        optimizer.zero_grad(); self.loss(model).backward(); optimizer.step(1.0)
        checkpoint = LearnerCheckpoint.capture(model, optimizer, {"cursor": 9})
        saved_checkpoint_state = copy.deepcopy(checkpoint.optimizer_state)

        def multi_step_branch():
            for multiplier in (0.25, 1.0):
                optimizer.zero_grad(); self.loss(model).backward(); optimizer.step(multiplier)
            return ({name: parameter.detach().clone() for name, parameter in self.selected(model)}, optimizer.state_dict())

        first_weights, first_state = multi_step_branch()
        checkpoint.restore(model, optimizer)
        second_weights, second_state = multi_step_branch()
        for name in first_weights:
            self.assertTrue(torch.equal(first_weights[name], second_weights[name]))
        for name in first_state["state"]:
            self.assertTrue(torch.equal(first_state["state"][name]["exp_avg"], second_state["state"][name]["exp_avg"]))
            self.assertTrue(torch.equal(first_state["state"][name]["exp_avg_sq"], second_state["state"][name]["exp_avg_sq"]))
        for name in saved_checkpoint_state["state"]:
            self.assertTrue(torch.equal(checkpoint.optimizer_state["state"][name]["exp_avg"], saved_checkpoint_state["state"][name]["exp_avg"]))
            self.assertTrue(torch.equal(checkpoint.optimizer_state["state"][name]["exp_avg_sq"], saved_checkpoint_state["state"][name]["exp_avg_sq"]))

    def test_checkpoint_rejects_frozen_parameter_mutation(self):
        model = self.tiny_olmoe()
        configure_trainable_experts(model, [0], [0])
        checkpoint = LearnerCheckpoint.capture(model, ExpertAdamW(self.selected(model), lr=1e-2), {})
        frozen = next(parameter for _, parameter in model.named_parameters() if not parameter.requires_grad)
        with torch.no_grad():
            frozen.add_(1)
        with self.assertRaisesRegex(RuntimeError, "frozen parameter mutated"):
            checkpoint.restore(model, ExpertAdamW(self.selected(model), lr=1e-2))

    def test_checkpoint_rejects_replaced_frozen_storage(self):
        model = self.tiny_olmoe()
        configure_trainable_experts(model, [0], [0])
        checkpoint = LearnerCheckpoint.capture(model, ExpertAdamW(model.named_parameters(), lr=1e-2), {})
        frozen = next(parameter for _, parameter in model.named_parameters() if not parameter.requires_grad)
        with torch.no_grad():
            frozen.data = frozen.detach().clone()
        with self.assertRaisesRegex(RuntimeError, "frozen parameter mutated"):
            checkpoint.restore(model, ExpertAdamW(model.named_parameters(), lr=1e-2))

    def test_gradients_reach_selected_experts_through_frozen_modules(self):
        model = self.tiny_olmoe()
        # Select all layer-1 experts: the frozen router chooses only top-k, so
        # a single selected expert is not guaranteed to receive this batch.
        configure_trainable_experts(model, [1], [0, 1, 2, 3])
        self.loss(model).backward()
        grads = [parameter.grad for _, parameter in self.selected(model)]
        self.assertTrue(all(grad is not None for grad in grads))
        self.assertTrue(any(torch.count_nonzero(grad).item() for grad in grads))


if __name__ == "__main__":
    unittest.main(verbosity=2)

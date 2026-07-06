from __future__ import annotations

import math
import random
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest

import torch

from scripts.train_grpo_addition import build_config, load_grpo_hard_cases, parse_args, sample_rollout_problems
from src.grpo import (
    build_completion_logprob_batch,
    completion_token_logprobs,
    grpo_loss,
    group_relative_advantages,
    score_addition_completion,
    score_arithmetic_completion,
)
from src.sampling import sample_completion_ids
from src.tokenizer import ArithmeticTokenizer


class GRPOTests(unittest.TestCase):
    def test_sample_completion_uses_stochastic_generation_path(self) -> None:
        tokenizer = ArithmeticTokenizer(["<pad>", *list("0123456789"), "+", "=", "\n"])

        class ForcedModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.next_ids = [tokenizer.token_to_id["2"], tokenizer.newline_id]
                self.calls = 0

            def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
                logits = torch.full((1, input_ids.size(1), tokenizer.vocab_size), -100.0)
                next_id = self.next_ids[min(self.calls, len(self.next_ids) - 1)]
                logits[0, -1, next_id] = 100.0
                self.calls += 1
                return logits

        completion = sample_completion_ids(
            model=ForcedModel(),
            tokenizer=tokenizer,
            prompt="1+1=",
            context_length=8,
            device=torch.device("cpu"),
            generator=torch.Generator().manual_seed(1),
        )

        self.assertEqual(completion.text, "2\n")
        self.assertTrue(completion.ended_with_newline)

    def test_score_addition_completion_uses_deterministic_verifier(self) -> None:
        correct = score_addition_completion(37, 48, "85\n")
        wrong = score_addition_completion(37, 48, "84\n")
        invalid = score_addition_completion(37, 48, "\n")

        self.assertEqual(correct.reward, 1.0)
        self.assertEqual(correct.expected, "85")
        self.assertEqual(wrong.reward, 0.0)
        self.assertTrue(wrong.valid)
        self.assertEqual(invalid.reward, 0.0)
        self.assertFalse(invalid.valid)

    def test_score_arithmetic_completion_uses_operation_verifier(self) -> None:
        multiplication = score_arithmetic_completion(12, 13, "*", "156\n")
        subtraction = score_arithmetic_completion(12, 30, "-", "-18\n")
        division = score_arithmetic_completion(78, 4, "/", "19R2\n")
        zero_division = score_arithmetic_completion(78, 0, "/", "ERR\n")
        invalid = score_arithmetic_completion(78, 4, "/", "19\n")

        self.assertEqual(multiplication.reward, 1.0)
        self.assertEqual(subtraction.reward, 1.0)
        self.assertEqual(division.reward, 1.0)
        self.assertEqual(zero_division.reward, 1.0)
        self.assertEqual(invalid.reward, 0.0)
        self.assertFalse(invalid.valid)

    def test_group_relative_advantages_are_zero_when_group_has_no_signal(self) -> None:
        no_signal = group_relative_advantages(torch.zeros((2, 4)))
        mixed = group_relative_advantages(torch.tensor([[0.0, 1.0, 0.0, 1.0]]))

        self.assertTrue(torch.equal(no_signal, torch.zeros_like(no_signal)))
        self.assertLess(float(mixed[0, 0]), 0.0)
        self.assertGreater(float(mixed[0, 1]), 0.0)
        self.assertAlmostEqual(float(mixed.sum().item()), 0.0, places=6)

    def test_completion_logprob_batch_masks_prompt_tokens(self) -> None:
        tokenizer = ArithmeticTokenizer()
        prompt_ids = [tokenizer.encode("1+1="), tokenizer.encode("2+2=")]
        completion_ids = [tokenizer.encode("2\n"), tokenizer.encode("4\n")]
        batch = build_completion_logprob_batch(
            tokenizer=tokenizer,
            prompt_token_ids=prompt_ids,
            completion_token_ids=completion_ids,
            device=torch.device("cpu"),
        )

        self.assertEqual(int(batch.completion_mask.sum().item()), 4)
        self.assertFalse(bool(batch.completion_mask[0, 0].item()))
        self.assertTrue(bool(batch.completion_mask[0, len(prompt_ids[0]) - 1].item()))

        class UniformModel(torch.nn.Module):
            def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
                return torch.zeros((input_ids.size(0), input_ids.size(1), tokenizer.vocab_size))

        token_logprobs = completion_token_logprobs(UniformModel(), batch)
        expected = -math.log(tokenizer.vocab_size)

        self.assertAlmostEqual(
            float(token_logprobs[batch.completion_mask][0].item()),
            expected,
            places=5,
        )
        self.assertEqual(float(token_logprobs[~batch.completion_mask].sum().item()), 0.0)

    def test_grpo_loss_returns_finite_optimization_metrics(self) -> None:
        policy_logprobs = torch.zeros((2, 3), requires_grad=True)
        old_logprobs = torch.zeros((2, 3))
        reference_logprobs = torch.zeros((2, 3))
        advantages = torch.tensor([1.0, -1.0])
        mask = torch.tensor([[True, True, False], [True, True, False]])

        result = grpo_loss(
            policy_logprobs=policy_logprobs,
            old_logprobs=old_logprobs,
            reference_logprobs=reference_logprobs,
            advantages=advantages,
            completion_mask=mask,
            beta=0.02,
            clip_epsilon=0.2,
        )

        self.assertTrue(torch.isfinite(result.loss))
        self.assertEqual(float(result.clip_fraction.item()), 0.0)
        self.assertAlmostEqual(float(result.mean_ratio.item()), 1.0, places=6)

    def test_grpo_config_can_sample_multiplication_rollouts(self) -> None:
        config = build_config(
            parse_args(
                [
                    "--operations",
                    "*",
                    "--operand-sampling",
                    "diverse",
                    "--prompts-per-step",
                    "8",
                ]
            )
        )

        problems = sample_rollout_problems(config, random.Random(1), None)

        self.assertEqual(config.operations, ("*",))
        self.assertEqual(config.operand_sampling, "diverse")
        self.assertEqual(len(problems), 8)
        self.assertTrue(all(operation == "*" for _, operation, _ in problems))

    def test_grpo_hard_cases_can_include_operations(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "hard_cases.json"
            path.write_text(
                '{"errors":[{"a":62,"operation":"*","b":88},{"a":37,"b":48}]}',
                encoding="utf-8",
            )

            hard_cases = load_grpo_hard_cases(path)

        self.assertEqual(hard_cases, [(62, "*", 88), (37, "+", 48)])

    def test_grpo_addition_trainer_can_show_help(self) -> None:
        result = subprocess.run(
            [sys.executable, "scripts/train_grpo_addition.py", "--help"],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0)
        self.assertIn("Run GRPO on the current 2-digit arithmetic", result.stdout)
        self.assertIn("--operations", result.stdout)
        self.assertIn("--operand-sampling", result.stdout)


if __name__ == "__main__":
    unittest.main()

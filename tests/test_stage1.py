from __future__ import annotations

import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from main import (
    ExperimentConfig,
    build_config,
    default_run_dir,
    load_initial_checkpoint,
    parse_args,
    save_checkpoint,
)
from scripts.compare_runs import summarize_run, to_markdown
from scripts.evaluate_checkpoint import evaluate_exhaustive
from src.data import (
    canonical_answer,
    format_addition,
    format_arithmetic,
    format_operation_answer,
    format_prompt,
    load_hard_cases,
    make_batch,
    max_addition_sequence_length,
    mixed_addition_pairs,
    mixed_arithmetic_problems,
    parse_operation_weights,
    parse_operations,
    parse_formatted_answer,
)
from src.generate import generate_text
from src.infer import parse_addition_query, parse_arithmetic_query
from src.model import AdditionTransformer, ModelConfig
from src.tokenizer import ArithmeticTokenizer
from src.train import next_token_loss, token_loss_weights


class Stage1Tests(unittest.TestCase):
    def test_tokenizer_round_trips_arithmetic_text(self) -> None:
        tokenizer = ArithmeticTokenizer()
        text = "7+8=15\n"

        token_ids = tokenizer.encode(text)

        self.assertEqual(tokenizer.vocab_size, 19)
        self.assertEqual(tokenizer.decode(token_ids), text)

    def test_tokenizer_rejects_unsupported_characters(self) -> None:
        tokenizer = ArithmeticTokenizer()

        with self.assertRaises(ValueError):
            tokenizer.encode("1.2=3\n")

    def test_batch_shapes_match_context_length(self) -> None:
        tokenizer = ArithmeticTokenizer()
        context_length = 16

        inputs, targets = make_batch(
            tokenizer=tokenizer,
            batch_size=4,
            context_length=context_length,
            digit_length=1,
            device=torch.device("cpu"),
            rng=random.Random(1),
        )

        self.assertEqual(inputs.shape, (4, context_length))
        self.assertEqual(targets.shape, (4, context_length))
        self.assertLessEqual(max_addition_sequence_length(1), context_length + 1)

    def test_model_forward_returns_next_token_logits(self) -> None:
        tokenizer = ArithmeticTokenizer()
        config = ModelConfig(
            vocab_size=tokenizer.vocab_size,
            d_model=32,
            num_heads=2,
            num_layers=1,
            ffn_hidden=128,
            context_length=16,
        )
        model = AdditionTransformer(config)
        input_ids = torch.zeros((2, 16), dtype=torch.long)

        logits = model(input_ids)

        self.assertEqual(logits.shape, (2, 16, tokenizer.vocab_size))

    def test_generation_stops_on_newline(self) -> None:
        tokenizer = ArithmeticTokenizer()

        class FakeModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.next_ids = [tokenizer.token_to_id["2"], tokenizer.newline_id]
                self.calls = 0

            def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
                logits = torch.zeros((1, input_ids.size(1), tokenizer.vocab_size))
                next_id = self.next_ids[min(self.calls, len(self.next_ids) - 1)]
                logits[0, -1, next_id] = 1.0
                self.calls += 1
                return logits

        generated = generate_text(
            FakeModel(),
            tokenizer,
            prompt="1+1=",
            context_length=8,
            device=torch.device("cpu"),
        )

        self.assertEqual(generated, "1+1=2\n")

    def test_runner_accepts_experiment_control_flags(self) -> None:
        args = parse_args(
            [
                "--steps",
                "3",
                "--eval-interval",
                "1",
                "--accuracy-examples",
                "2",
                "--val-batches",
                "1",
                "--samples",
                "1",
                "--hard-case-file",
                "final/eval_exhaustive_2digit.json",
                "--hard-case-ratio",
                "0.25",
                "--init-checkpoint",
                "final/checkpoint_best_val.pt",
                "--allow-vocab-expansion",
                "--number-format",
                "fixed_reversed",
                "--loss-prompt-weight",
                "0.2",
                "--operations",
                "+,-,*,/",
                "--operation-weights",
                "+:1,-:2,*:3,/:4",
                "--operand-sampling",
                "diverse",
            ]
        )

        config = build_config(args)

        self.assertEqual(config.steps, 3)
        self.assertEqual(config.eval_interval, 1)
        self.assertEqual(config.accuracy_examples, 2)
        self.assertEqual(config.val_batches, 1)
        self.assertEqual(config.samples, 1)
        self.assertEqual(config.hard_case_file, "final/eval_exhaustive_2digit.json")
        self.assertEqual(config.hard_case_ratio, 0.25)
        self.assertEqual(config.init_checkpoint, "final/checkpoint_best_val.pt")
        self.assertTrue(config.allow_vocab_expansion)
        self.assertEqual(config.number_format, "fixed_reversed")
        self.assertEqual(config.loss_prompt_weight, 0.2)
        self.assertEqual(config.operations, ("+", "-", "*", "/"))
        self.assertEqual(config.operation_weights, (1.0, 2.0, 3.0, 4.0))
        self.assertEqual(config.operand_sampling, "diverse")

    def test_runner_defaults_to_stage3_tuning_config(self) -> None:
        config = build_config(parse_args([]))

        self.assertEqual(config.stage, "stage3")
        self.assertEqual(config.d_model, 128)
        self.assertEqual(config.num_heads, 4)
        self.assertEqual(config.num_layers, 4)
        self.assertEqual(config.ffn_hidden, 512)
        self.assertEqual(config.digit_length, 2)
        self.assertEqual(config.steps, 500)
        self.assertEqual(config.eval_interval, 100)
        self.assertEqual(default_run_dir(config.stage).parts[:2], ("runs", "stage3"))

    def test_number_formats_make_carry_local_representations(self) -> None:
        self.assertEqual(format_addition(37, 48, 2, "normal"), "37+48=85\n")
        self.assertEqual(format_addition(37, 48, 2, "fixed"), "37+48=085\n")
        self.assertEqual(format_addition(37, 48, 2, "reversed"), "73+84=58\n")
        self.assertEqual(format_addition(37, 48, 2, "fixed_reversed"), "73+84=580\n")
        self.assertEqual(format_prompt(7, 5, 2, "fixed_reversed"), "70+50=")
        self.assertEqual(parse_formatted_answer("580", "fixed_reversed"), "85")
        self.assertEqual(parse_formatted_answer("58", "reversed"), "85")

    def test_operation_formatting_supports_all_bounded_ops(self) -> None:
        self.assertEqual(format_arithmetic(80, "+", 26, 2, "fixed_reversed"), "08+62=601\n")
        self.assertEqual(format_arithmetic(91, "-", 47, 2, "fixed_reversed"), "19-74=44\n")
        self.assertEqual(format_arithmetic(3, "-", 7, 2, "fixed_reversed"), "30-70=-40\n")
        self.assertEqual(format_arithmetic(12, "*", 8, 2, "fixed_reversed"), "21*80=6900\n")
        self.assertEqual(format_arithmetic(80, "/", 7, 2, "fixed_reversed"), "08/70=11R30\n")
        self.assertEqual(format_operation_answer(8, 0, 2, "fixed_reversed", "/"), "ERR")
        self.assertEqual(parse_formatted_answer("11R30", "fixed_reversed", "/"), "11R3")
        self.assertEqual(parse_formatted_answer("-40", "fixed_reversed", "-"), "-4")
        self.assertEqual(canonical_answer(80, 7, "/"), "11R3")
        self.assertEqual(parse_operations("+,-,*,/"), ("+", "-", "*", "/"))
        self.assertEqual(parse_operation_weights("1,2,3,4", ("+", "-", "*", "/")), (1.0, 2.0, 3.0, 4.0))
        self.assertEqual(
            parse_operation_weights("+:1,-:2,*:3,/:4", ("+", "-", "*", "/")),
            (1.0, 2.0, 3.0, 4.0),
        )

        with self.assertRaisesRegex(ValueError, "at least one operation weight"):
            parse_operation_weights("+:0,-:0", ("+", "-"))

        with self.assertRaisesRegex(ValueError, "inactive operation"):
            parse_operation_weights("*:1", ("+", "-"))

    def test_weighted_arithmetic_problem_generation_can_focus_one_operation(self) -> None:
        problems = mixed_arithmetic_problems(
            batch_size=12,
            digit_length=2,
            rng=random.Random(1),
            operations=("+", "-", "*", "/"),
            operation_weights=(0.0, 0.0, 1.0, 0.0),
        )

        self.assertEqual({operation for _, operation, _ in problems}, {"*"})

    def test_diverse_operand_sampling_adds_operation_specific_patterns(self) -> None:
        multiplication = mixed_arithmetic_problems(
            batch_size=40,
            digit_length=2,
            rng=random.Random(2),
            operations=("*",),
            operand_sampling="diverse",
        )
        division = mixed_arithmetic_problems(
            batch_size=80,
            digit_length=2,
            rng=random.Random(3),
            operations=("/",),
            operand_sampling="diverse",
        )

        self.assertTrue(any(min(a, b) <= 9 for a, _, b in multiplication))
        self.assertTrue(any(a % 10 == 0 or b % 10 == 0 for a, _, b in multiplication))
        self.assertTrue(any(b == 0 for _, _, b in division))
        self.assertTrue(any(b != 0 and a % b == 0 for a, _, b in division))

    def test_answer_weighted_loss_uses_equal_sign_boundary(self) -> None:
        tokenizer = ArithmeticTokenizer()
        inputs, targets = make_batch(
            tokenizer=tokenizer,
            batch_size=1,
            context_length=8,
            digit_length=1,
            device=torch.device("cpu"),
            rng=random.Random(1),
        )

        weights = token_loss_weights(inputs, targets, tokenizer.token_to_id["="], tokenizer.pad_id, 0.2)
        decoded_input = tokenizer.decode(inputs[0].tolist())
        equal_index = decoded_input.index("=")

        self.assertTrue(torch.all(weights[0, :equal_index] == 0.2))
        self.assertEqual(weights[0, equal_index].item(), 1.0)
        self.assertEqual(weights[targets == tokenizer.pad_id].sum().item(), 0.0)

        logits = torch.zeros((1, targets.size(1), tokenizer.vocab_size))
        loss = next_token_loss(logits, targets, tokenizer.pad_id, weights)
        self.assertGreater(float(loss.item()), 0.0)

    def test_compare_run_summarizes_saved_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "runs" / "stage2" / "example"
            run_dir.mkdir(parents=True)
            (run_dir / "config.json").write_text(
                """
{
  "device": "cpu",
  "experiment": {"stage": "stage2", "digit_length": 1},
  "model": {
    "d_model": 128,
    "num_heads": 4,
    "num_layers": 4,
    "ffn_hidden": 512,
    "context_length": 16,
    "vocab_size": 14
  },
  "parameter_count": 801422
}
""".strip()
                + "\n",
                encoding="utf-8",
            )
            (run_dir / "metrics.csv").write_text(
                "step,train_loss,val_loss,exact_accuracy,elapsed_s\n"
                "1,2.0,2.1,0.0,0.5\n"
                "10,1.0,1.2,0.4,1.5\n"
                "20,0.8,0.9,0.6,2.5\n",
                encoding="utf-8",
            )
            (run_dir / "summary.json").write_text(
                """
{
  "device": "cpu",
  "elapsed_s": 2.5,
  "final_exact_accuracy": 0.6,
  "final_train_loss": 0.8,
  "final_val_loss": 0.9,
  "parameter_count": 801422
}
""".strip()
                + "\n",
                encoding="utf-8",
            )

            summary = summarize_run(run_dir)
            rendered = to_markdown([summary])

        self.assertEqual(summary["stage"], "stage2")
        self.assertEqual(summary["steps"], 20)
        self.assertEqual(summary["best_val_loss"], 0.9)
        self.assertEqual(summary["best_exact_accuracy"], 0.6)
        self.assertIn("final_val_loss", rendered)

    def test_checkpoint_save_contains_reloadable_training_state(self) -> None:
        tokenizer = ArithmeticTokenizer()
        model_config = ModelConfig(
            vocab_size=tokenizer.vocab_size,
            d_model=32,
            num_heads=2,
            num_layers=1,
            ffn_hidden=128,
            context_length=16,
        )
        model = AdditionTransformer(model_config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        metrics = {
            "step": 1.0,
            "train_loss": 2.0,
            "val_loss": 1.5,
            "exact_accuracy": 0.25,
            "elapsed_s": 0.5,
        }

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "checkpoint.pt"
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                ExperimentConfig(),
                model_config,
                tokenizer,
                step=1,
                metrics_row=metrics,
            )
            checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))

        self.assertEqual(checkpoint["step"], 1)
        self.assertEqual(checkpoint["metrics"], metrics)
        self.assertEqual(checkpoint["model_config"]["d_model"], 32)
        self.assertIn("model_state_dict", checkpoint)
        self.assertIn("optimizer_state_dict", checkpoint)

    def test_load_initial_checkpoint_restores_model_weights(self) -> None:
        tokenizer = ArithmeticTokenizer()
        model_config = ModelConfig(
            vocab_size=tokenizer.vocab_size,
            d_model=32,
            num_heads=2,
            num_layers=1,
            ffn_hidden=128,
            context_length=16,
        )
        source_model = AdditionTransformer(model_config)
        optimizer = torch.optim.AdamW(source_model.parameters(), lr=0.001)
        metrics = {
            "step": 3.0,
            "train_loss": 2.0,
            "val_loss": 1.5,
            "exact_accuracy": 0.25,
            "elapsed_s": 0.5,
        }

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "checkpoint.pt"
            save_checkpoint(
                checkpoint_path,
                source_model,
                optimizer,
                ExperimentConfig(),
                model_config,
                tokenizer,
                step=3,
                metrics_row=metrics,
            )

            restored_model = AdditionTransformer(model_config)
            loaded_step = load_initial_checkpoint(
                checkpoint_path,
                restored_model,
                model_config,
                tokenizer,
                torch.device("cpu"),
            )

        self.assertEqual(loaded_step, 3)
        for source, restored in zip(source_model.parameters(), restored_model.parameters(), strict=True):
            self.assertTrue(torch.equal(source, restored))

    def test_load_initial_checkpoint_can_expand_vocab_by_token_name(self) -> None:
        source_tokenizer = ArithmeticTokenizer(list(ArithmeticTokenizer.BASE_TOKENS))
        target_tokenizer = ArithmeticTokenizer()
        source_config = ModelConfig(
            vocab_size=source_tokenizer.vocab_size,
            d_model=32,
            num_heads=2,
            num_layers=1,
            ffn_hidden=128,
            context_length=16,
        )
        target_config = ModelConfig(
            vocab_size=target_tokenizer.vocab_size,
            d_model=32,
            num_heads=2,
            num_layers=1,
            ffn_hidden=128,
            context_length=16,
        )
        source_model = AdditionTransformer(source_config)
        optimizer = torch.optim.AdamW(source_model.parameters(), lr=0.001)
        with torch.no_grad():
            source_model.token_embedding.weight[source_tokenizer.token_to_id["="]].fill_(3.0)
            source_model.token_embedding.weight[source_tokenizer.newline_id].fill_(5.0)
            source_model.lm_head.weight[source_tokenizer.token_to_id["="]].fill_(7.0)
            source_model.lm_head.weight[source_tokenizer.newline_id].fill_(11.0)

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "checkpoint.pt"
            save_checkpoint(
                checkpoint_path,
                source_model,
                optimizer,
                ExperimentConfig(),
                source_config,
                source_tokenizer,
                step=4,
                metrics_row={
                    "step": 4.0,
                    "train_loss": 1.0,
                    "val_loss": 1.0,
                    "exact_accuracy": 0.0,
                    "elapsed_s": 0.1,
                },
            )

            target_model = AdditionTransformer(target_config)
            with self.assertRaisesRegex(ValueError, "does not match requested"):
                load_initial_checkpoint(
                    checkpoint_path,
                    target_model,
                    target_config,
                    target_tokenizer,
                    torch.device("cpu"),
                )

            loaded_step = load_initial_checkpoint(
                checkpoint_path,
                target_model,
                target_config,
                target_tokenizer,
                torch.device("cpu"),
                allow_vocab_expansion=True,
            )

        self.assertEqual(loaded_step, 4)
        self.assertTrue(
            torch.all(target_model.token_embedding.weight[target_tokenizer.token_to_id["="]] == 3.0)
        )
        self.assertTrue(torch.all(target_model.token_embedding.weight[target_tokenizer.newline_id] == 5.0))
        self.assertTrue(torch.all(target_model.lm_head.weight[target_tokenizer.token_to_id["="]] == 7.0))
        self.assertTrue(torch.all(target_model.lm_head.weight[target_tokenizer.newline_id] == 11.0))
        self.assertTrue(
            torch.equal(
                source_model.blocks[0].attn.q_proj.weight,
                target_model.blocks[0].attn.q_proj.weight,
            )
        )

    def test_exhaustive_eval_counts_all_pairs(self) -> None:
        tokenizer = ArithmeticTokenizer()

        class OracleAdditionModel(torch.nn.Module):
            def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
                logits = torch.zeros((input_ids.size(0), input_ids.size(1), tokenizer.vocab_size))
                for row in range(input_ids.size(0)):
                    prompt = tokenizer.decode(input_ids[row].tolist())
                    if "=" not in prompt:
                        next_char = "="
                    else:
                        expression, suffix = prompt.split("=", maxsplit=1)
                        a_text, b_text = expression.split("+", maxsplit=1)
                        answer = str(int(a_text) + int(b_text)) + "\n"
                        next_char = answer[len(suffix)] if len(suffix) < len(answer) else "\n"
                    logits[row, -1, tokenizer.token_to_id[next_char]] = 1.0
                return logits

        result = evaluate_exhaustive(
            model=OracleAdditionModel(),
            tokenizer=tokenizer,
            device=torch.device("cpu"),
            digit_length=1,
            context_length=8,
            max_errors=5,
            batch_size=7,
            progress_every=0,
        )

        self.assertEqual(result["total"], 100)
        self.assertEqual(result["correct"], 100)
        self.assertEqual(result["incorrect"], 0)
        self.assertEqual(result["accuracy"], 1.0)

    def test_exhaustive_eval_supports_fixed_reversed_format(self) -> None:
        tokenizer = ArithmeticTokenizer()

        class FixedReversedOracle(torch.nn.Module):
            def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
                logits = torch.zeros((input_ids.size(0), input_ids.size(1), tokenizer.vocab_size))
                for row in range(input_ids.size(0)):
                    prompt = tokenizer.decode(input_ids[row].tolist())
                    if "=" not in prompt:
                        next_char = "="
                    else:
                        expression, suffix = prompt.split("=", maxsplit=1)
                        a_text, b_text = expression.split("+", maxsplit=1)
                        a = int(a_text[::-1])
                        b = int(b_text[::-1])
                        answer = f"{a + b:02d}"[::-1] + "\n"
                        next_char = answer[len(suffix)] if len(suffix) < len(answer) else "\n"
                    logits[row, -1, tokenizer.token_to_id[next_char]] = 1.0
                return logits

        result = evaluate_exhaustive(
            model=FixedReversedOracle(),
            tokenizer=tokenizer,
            device=torch.device("cpu"),
            digit_length=1,
            context_length=8,
            max_errors=5,
            batch_size=7,
            progress_every=0,
            number_format="fixed_reversed",
        )

        self.assertEqual(result["correct"], 100)
        self.assertEqual(result["number_format"], "fixed_reversed")

    def test_checkpoint_evaluator_can_run_as_script(self) -> None:
        result = subprocess.run(
            [sys.executable, "scripts/evaluate_checkpoint.py", "--help"],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0)
        self.assertIn("Evaluate a saved addition-transformer checkpoint", result.stdout)

    def test_chat_addition_script_can_show_help(self) -> None:
        result = subprocess.run(
            [sys.executable, "scripts/chat_addition.py", "--help"],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0)
        self.assertIn("Chat with a trained addition-transformer checkpoint", result.stdout)

    def test_parse_addition_query_accepts_chatty_input(self) -> None:
        self.assertEqual(parse_addition_query("37+48", 2), (37, 48))
        self.assertEqual(parse_addition_query("what is 7 + 9?", 2), (7, 9))
        self.assertEqual(parse_arithmetic_query("91 - 47", 2), (91, "-", 47))
        self.assertEqual(parse_arithmetic_query("12*8", 2), (12, "*", 8))
        self.assertEqual(parse_arithmetic_query("80/7", 2), (80, "/", 7))

        with self.assertRaisesRegex(ValueError, "supports 2-digit operands only"):
            parse_addition_query("100+1", 2)

        with self.assertRaisesRegex(ValueError, "enter only one"):
            parse_arithmetic_query("1+2 and 3+4", 2)

    def test_load_hard_cases_from_exhaustive_eval_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            hard_case_path = Path(tmp) / "eval.json"
            hard_case_path.write_text(
                """
{
  "errors": [
    {"a": 0, "b": 8, "expected": "8"},
    {"a": 92, "b": 41, "expected": "133"}
  ]
}
""".strip()
                + "\n",
                encoding="utf-8",
            )

            hard_cases = load_hard_cases(hard_case_path)

        self.assertEqual(hard_cases, [(0, 8), (92, 41)])

    def test_mixed_addition_pairs_uses_requested_hard_case_ratio(self) -> None:
        hard_cases = [(0, 8), (92, 41)]
        pairs = mixed_addition_pairs(
            batch_size=8,
            digit_length=2,
            rng=random.Random(1),
            hard_cases=hard_cases,
            hard_case_ratio=0.25,
        )

        hard_count = sum(pair in hard_cases for pair in pairs)

        self.assertEqual(len(pairs), 8)
        self.assertEqual(hard_count, 2)


if __name__ == "__main__":
    unittest.main()

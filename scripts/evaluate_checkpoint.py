from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.generate import generate_text
from src.model import AdditionTransformer, ModelConfig
from src.data import (
    NUMBER_FORMATS,
    Operation,
    NumberFormat,
    canonical_answer,
    format_prompt,
    parse_formatted_answer,
    parse_operations,
)
from src.tokenizer import ArithmeticTokenizer
from src.train import select_device


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    return torch.load(path, map_location=device)


def build_model(checkpoint: dict[str, Any], device: torch.device) -> AdditionTransformer:
    model_config = ModelConfig(**checkpoint["model_config"])
    model = AdditionTransformer(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def parse_answer(generated: str, prompt: str) -> str:
    if not generated.startswith(prompt):
        return ""
    return generated[len(prompt) :].split("\n", maxsplit=1)[0]


@torch.no_grad()
def generate_batch(
    model: AdditionTransformer,
    tokenizer: ArithmeticTokenizer,
    device: torch.device,
    prompts: list[str],
    context_length: int,
) -> list[str]:
    if not prompts:
        return []

    encoded = [tokenizer.encode(prompt) for prompt in prompts]
    lengths = {len(token_ids) for token_ids in encoded}
    if len(lengths) != 1:
        raise ValueError("batched generation requires prompts with equal token length")

    prompt_length = lengths.pop()
    if prompt_length > context_length:
        raise ValueError(f"prompt length {prompt_length} exceeds context_length={context_length}")

    sequences = [token_ids[:] for token_ids in encoded]
    done = [False] * len(sequences)
    max_new_tokens = context_length - prompt_length

    for _ in range(max_new_tokens):
        inputs = torch.tensor(sequences, dtype=torch.long, device=device)
        logits = model(inputs)
        next_ids = torch.argmax(logits[:, -1, :], dim=-1).tolist()

        for index, next_id in enumerate(next_ids):
            if done[index]:
                sequences[index].append(tokenizer.pad_id)
                continue
            sequences[index].append(int(next_id))
            if int(next_id) == tokenizer.newline_id:
                done[index] = True

        if all(done):
            break

    return [tokenizer.decode(sequence) for sequence in sequences]


def build_examples(
    digit_length: int,
    number_format: NumberFormat = "normal",
    operations: tuple[Operation, ...] = ("+",),
) -> list[dict[str, Any]]:
    upper = (10**digit_length) - 1
    return [
        {
            "a": a,
            "b": b,
            "operation": operation,
            "prompt": format_prompt(a, b, digit_length, number_format, operation),
            "expected": canonical_answer(a, b, operation),
        }
        for operation in operations
        for a in range(upper + 1)
        for b in range(upper + 1)
    ]


@torch.no_grad()
def evaluate_exhaustive(
    model: AdditionTransformer,
    tokenizer: ArithmeticTokenizer,
    device: torch.device,
    digit_length: int,
    context_length: int,
    max_errors: int,
    batch_size: int = 256,
    progress_every: int = 1000,
    number_format: NumberFormat = "normal",
    operations: tuple[Operation, ...] = ("+",),
) -> dict[str, Any]:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    examples = build_examples(digit_length, number_format, operations)
    total = len(examples)
    correct = 0
    processed = 0
    errors: list[dict[str, Any]] = []
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    started = time.perf_counter()

    for example in examples:
        grouped[len(tokenizer.encode(example["prompt"]))].append(example)

    for prompt_length in sorted(grouped):
        group = grouped[prompt_length]
        for start in range(0, len(group), batch_size):
            batch = group[start : start + batch_size]
            generated_texts = generate_batch(
                model=model,
                tokenizer=tokenizer,
                device=device,
                prompts=[example["prompt"] for example in batch],
                context_length=context_length,
            )
            for example, generated in zip(batch, generated_texts, strict=True):
                answer = parse_formatted_answer(
                    parse_answer(generated, example["prompt"]),
                    number_format,
                    example["operation"],
                )
                is_correct = answer == example["expected"]
                correct += int(is_correct)
                if not is_correct and len(errors) < max_errors:
                    errors.append(
                        {
                            "a": example["a"],
                            "b": example["b"],
                            "operation": example["operation"],
                            "prompt": example["prompt"],
                            "expected": example["expected"],
                            "answer": answer,
                            "generated": generated,
                        }
                    )

            processed += len(batch)
            if progress_every and (processed == total or processed % progress_every < len(batch)):
                elapsed = time.perf_counter() - started
                print(
                    f"progress={processed}/{total} "
                    f"accuracy_so_far={correct / processed:.6f} "
                    f"elapsed_s={elapsed:.1f}",
                    file=sys.stderr,
                    flush=True,
                )

    accuracy = correct / total if total else 0.0
    upper = (10**digit_length) - 1
    return {
        "digit_length": digit_length,
        "range": [0, upper],
        "total": total,
        "correct": correct,
        "incorrect": total - correct,
        "accuracy": accuracy,
        "number_format": number_format,
        "operations": operations,
        "batch_size": batch_size,
        "errors": errors,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved addition-transformer checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--digit-length", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-errors", type=int, default=50)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--number-format", choices=NUMBER_FORMATS, default="normal")
    parser.add_argument("--operations", type=str, default="+")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, device)
    tokenizer = ArithmeticTokenizer(checkpoint.get("vocab"))

    model = build_model(checkpoint, device)
    model_config = checkpoint["model_config"]
    result = evaluate_exhaustive(
        model=model,
        tokenizer=tokenizer,
        device=device,
        digit_length=args.digit_length,
        context_length=int(model_config["context_length"]),
        max_errors=args.max_errors,
        batch_size=args.batch_size,
        progress_every=args.progress_every,
        number_format=args.number_format,
        operations=parse_operations(args.operations),
    )
    result["checkpoint"] = str(args.checkpoint)
    result["checkpoint_step"] = checkpoint.get("step")
    result["checkpoint_metrics"] = checkpoint.get("metrics")
    result["model_config"] = model_config
    result["parameter_count"] = checkpoint.get("parameter_count")

    write_json(args.out, result)
    print(
        f"accuracy={result['accuracy']:.6f} "
        f"correct={result['correct']} "
        f"total={result['total']} "
        f"incorrect={result['incorrect']}"
    )
    print(f"out={args.out}")


if __name__ == "__main__":
    main()

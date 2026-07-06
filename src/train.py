from __future__ import annotations

import random
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from src.data import (
    Operation,
    NumberFormat,
    OperandSampling,
    OperationWeights,
    canonical_answer,
    format_prompt,
    make_batch,
    parse_formatted_answer,
    random_arithmetic_problem,
)
from src.generate import generate_text
from src.tokenizer import ArithmeticTokenizer


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int
    steps: int
    eval_interval: int
    val_batches: int
    accuracy_examples: int
    learning_rate: float
    weight_decay: float
    grad_clip: float
    digit_length: int
    context_length: int
    seed: int
    number_format: NumberFormat = "normal"
    loss_prompt_weight: float = 1.0
    operations: tuple[Operation, ...] = ("+",)
    operation_weights: OperationWeights | None = None
    operand_sampling: OperandSampling = "uniform"


def select_device(preferred: str = "auto") -> torch.device:
    if preferred == "cpu":
        return torch.device("cpu")
    if preferred == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        raise RuntimeError("MPS was requested but is not available")
    if preferred == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise RuntimeError("CUDA was requested but is not available")
    if preferred != "auto":
        raise ValueError(f"unknown device preference: {preferred}")

    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def token_loss_weights(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    equals_id: int,
    pad_id: int,
    prompt_weight: float,
) -> torch.Tensor:
    if prompt_weight < 0.0:
        raise ValueError("prompt_weight must be non-negative")

    answer_region = torch.cumsum((inputs == equals_id).to(torch.int64), dim=1) > 0
    weights = torch.where(
        answer_region,
        torch.ones_like(targets, dtype=torch.float32),
        torch.full_like(targets, float(prompt_weight), dtype=torch.float32),
    )
    return weights.masked_fill(targets == pad_id, 0.0)


def next_token_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pad_id: int,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if weights is None:
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=pad_id,
        )

    losses = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=pad_id,
        reduction="none",
    )
    flat_weights = weights.reshape(-1).to(losses.dtype)
    return (losses * flat_weights).sum() / flat_weights.sum().clamp_min(1.0)


@torch.no_grad()
def evaluate_loss(
    model: torch.nn.Module,
    tokenizer: ArithmeticTokenizer,
    device: torch.device,
    batch_size: int,
    context_length: int,
    digit_length: int,
    batches: int,
    rng: random.Random,
    number_format: NumberFormat = "normal",
    loss_prompt_weight: float = 1.0,
    operations: tuple[Operation, ...] = ("+",),
    operation_weights: OperationWeights | None = None,
    operand_sampling: OperandSampling = "uniform",
) -> float:
    model.eval()
    losses: list[float] = []
    for _ in range(batches):
        inputs, targets = make_batch(
            tokenizer,
            batch_size,
            context_length,
            digit_length,
            device,
            rng,
            number_format=number_format,
            operations=operations,
            operation_weights=operation_weights,
            operand_sampling=operand_sampling,
        )
        weights = token_loss_weights(
            inputs,
            targets,
            tokenizer.token_to_id["="],
            tokenizer.pad_id,
            loss_prompt_weight,
        )
        losses.append(float(next_token_loss(model(inputs), targets, tokenizer.pad_id, weights).item()))
    return sum(losses) / len(losses)


@torch.no_grad()
def exact_answer_accuracy(
    model: torch.nn.Module,
    tokenizer: ArithmeticTokenizer,
    device: torch.device,
    context_length: int,
    digit_length: int,
    examples: int,
    rng: random.Random,
    number_format: NumberFormat = "normal",
    operations: tuple[Operation, ...] = ("+",),
    operation_weights: OperationWeights | None = None,
    operand_sampling: OperandSampling = "uniform",
) -> float:
    model.eval()
    correct = 0
    for _ in range(examples):
        a, operation, b = random_arithmetic_problem(
            digit_length,
            rng,
            operations,
            operation_weights,
            operand_sampling,
        )
        prompt = format_prompt(a, b, digit_length, number_format, operation)
        generated = generate_text(model, tokenizer, prompt, context_length, device)
        answer = parse_formatted_answer(
            generated[len(prompt) :].split("\n", maxsplit=1)[0],
            number_format,
            operation,
        )
        if answer == canonical_answer(a, b, operation):
            correct += 1
    return correct / examples

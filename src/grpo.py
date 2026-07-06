from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from src.data import NumberFormat, Operation, canonical_answer, parse_formatted_answer
from src.tokenizer import ArithmeticTokenizer


@dataclass(frozen=True)
class RewardResult:
    raw_answer: str
    parsed_answer: str
    expected: str
    reward: float
    valid: bool


@dataclass(frozen=True)
class CompletionLogProbBatch:
    input_ids: torch.Tensor
    target_ids: torch.Tensor
    completion_mask: torch.Tensor


@dataclass(frozen=True)
class GRPOLossResult:
    loss: torch.Tensor
    policy_loss: torch.Tensor
    kl_loss: torch.Tensor
    clip_fraction: torch.Tensor
    mean_ratio: torch.Tensor


def score_arithmetic_completion(
    a: int,
    b: int,
    operation: Operation,
    completion_text: str,
    number_format: NumberFormat = "normal",
) -> RewardResult:
    raw_answer = completion_text.split("\n", maxsplit=1)[0]
    parsed_answer = parse_formatted_answer(raw_answer, number_format, operation)
    expected = canonical_answer(a, b, operation)
    reward = 1.0 if parsed_answer == expected else 0.0
    return RewardResult(
        raw_answer=raw_answer,
        parsed_answer=parsed_answer,
        expected=expected,
        reward=reward,
        valid=bool(parsed_answer),
    )


def score_addition_completion(
    a: int,
    b: int,
    completion_text: str,
    number_format: NumberFormat = "normal",
) -> RewardResult:
    return score_arithmetic_completion(a, b, "+", completion_text, number_format)


def group_relative_advantages(rewards: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if rewards.ndim != 2:
        raise ValueError("rewards must have shape [num_prompts, group_size]")

    means = rewards.mean(dim=1, keepdim=True)
    stds = rewards.std(dim=1, keepdim=True, unbiased=False)
    return (rewards - means) / stds.clamp_min(eps)


def build_completion_logprob_batch(
    tokenizer: ArithmeticTokenizer,
    prompt_token_ids: list[list[int]],
    completion_token_ids: list[list[int]],
    device: torch.device,
) -> CompletionLogProbBatch:
    if len(prompt_token_ids) != len(completion_token_ids):
        raise ValueError("prompt_token_ids and completion_token_ids must have the same length")
    if not prompt_token_ids:
        raise ValueError("at least one sequence is required")
    if any(not completion for completion in completion_token_ids):
        raise ValueError("each completion must contain at least one token")

    sequences = [prompt + completion for prompt, completion in zip(prompt_token_ids, completion_token_ids)]
    max_input_length = max(len(sequence) - 1 for sequence in sequences)

    inputs = torch.full(
        (len(sequences), max_input_length),
        tokenizer.pad_id,
        dtype=torch.long,
        device=device,
    )
    targets = torch.full_like(inputs, tokenizer.pad_id)
    completion_mask = torch.zeros_like(inputs, dtype=torch.bool)

    for row, (prompt, completion, sequence) in enumerate(zip(prompt_token_ids, completion_token_ids, sequences)):
        input_sequence = sequence[:-1]
        target_sequence = sequence[1:]
        inputs[row, : len(input_sequence)] = torch.tensor(input_sequence, dtype=torch.long, device=device)
        targets[row, : len(target_sequence)] = torch.tensor(target_sequence, dtype=torch.long, device=device)

        completion_start = len(prompt) - 1
        completion_end = completion_start + len(completion)
        completion_mask[row, completion_start:completion_end] = True

    return CompletionLogProbBatch(
        input_ids=inputs,
        target_ids=targets,
        completion_mask=completion_mask,
    )


def completion_token_logprobs(
    model: torch.nn.Module,
    batch: CompletionLogProbBatch,
) -> torch.Tensor:
    logits = model(batch.input_ids)
    logprobs = F.log_softmax(logits, dim=-1)
    token_logprobs = torch.gather(logprobs, dim=-1, index=batch.target_ids.unsqueeze(-1)).squeeze(-1)
    return token_logprobs.masked_fill(~batch.completion_mask, 0.0)


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def grpo_loss(
    policy_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    reference_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    completion_mask: torch.Tensor,
    beta: float,
    clip_epsilon: float,
) -> GRPOLossResult:
    if beta < 0.0:
        raise ValueError("beta must be non-negative")
    if not 0.0 <= clip_epsilon < 1.0:
        raise ValueError("clip_epsilon must be in [0, 1)")

    token_advantages = advantages.to(policy_logprobs.dtype).unsqueeze(1)
    log_ratio = policy_logprobs - old_logprobs
    ratio = torch.exp(log_ratio)
    clipped_ratio = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon)
    surrogate = torch.minimum(ratio * token_advantages, clipped_ratio * token_advantages)

    ref_delta = reference_logprobs - policy_logprobs
    kl = torch.exp(ref_delta) - ref_delta - 1.0

    policy_loss = -masked_mean(surrogate, completion_mask)
    kl_loss = masked_mean(kl, completion_mask)
    loss = policy_loss + (beta * kl_loss)
    clip_fraction = masked_mean((ratio != clipped_ratio).to(policy_logprobs.dtype), completion_mask)
    mean_ratio = masked_mean(ratio, completion_mask)

    return GRPOLossResult(
        loss=loss,
        policy_loss=policy_loss,
        kl_loss=kl_loss,
        clip_fraction=clip_fraction,
        mean_ratio=mean_ratio,
    )

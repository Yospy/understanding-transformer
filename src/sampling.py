from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from src.tokenizer import ArithmeticTokenizer


@dataclass(frozen=True)
class SampledCompletion:
    token_ids: list[int]
    text: str
    ended_with_newline: bool


@torch.no_grad()
def sample_completion_ids(
    model: torch.nn.Module,
    tokenizer: ArithmeticTokenizer,
    prompt: str,
    context_length: int,
    device: torch.device,
    max_new_tokens: int | None = None,
    temperature: float = 1.0,
    generator: torch.Generator | None = None,
) -> SampledCompletion:
    if temperature <= 0.0:
        raise ValueError("temperature must be positive for sampled generation")

    model.eval()
    ids = tokenizer.encode(prompt)
    if len(ids) > context_length:
        raise ValueError(f"prompt length {len(ids)} exceeds context_length={context_length}")

    budget = max_new_tokens if max_new_tokens is not None else context_length - len(ids)
    completion_ids: list[int] = []

    for _ in range(max(0, budget)):
        window = ids[-context_length:]
        inputs = torch.tensor([window], dtype=torch.long, device=device)
        logits = model(inputs)[0, -1, :].float()
        probs = F.softmax(logits / temperature, dim=-1)
        next_id = int(torch.multinomial(probs.cpu(), num_samples=1, generator=generator).item())

        ids.append(next_id)
        completion_ids.append(next_id)
        if next_id == tokenizer.newline_id:
            break

    return SampledCompletion(
        token_ids=completion_ids,
        text=tokenizer.decode(completion_ids),
        ended_with_newline=bool(completion_ids and completion_ids[-1] == tokenizer.newline_id),
    )


def sample_completions(
    model: torch.nn.Module,
    tokenizer: ArithmeticTokenizer,
    prompt: str,
    context_length: int,
    device: torch.device,
    count: int,
    max_new_tokens: int | None = None,
    temperature: float = 1.0,
    generator: torch.Generator | None = None,
) -> list[SampledCompletion]:
    if count < 1:
        raise ValueError("count must be >= 1")

    return [
        sample_completion_ids(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            context_length=context_length,
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            generator=generator,
        )
        for _ in range(count)
    ]


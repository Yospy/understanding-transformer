from __future__ import annotations

import torch

from src.tokenizer import ArithmeticTokenizer


@torch.no_grad()
def generate_ids(
    model: torch.nn.Module,
    tokenizer: ArithmeticTokenizer,
    prompt: str,
    context_length: int,
    device: torch.device,
    max_new_tokens: int | None = None,
) -> list[int]:
    model.eval()
    ids = tokenizer.encode(prompt)
    if len(ids) > context_length:
        raise ValueError(f"prompt length {len(ids)} exceeds context_length={context_length}")

    budget = max_new_tokens if max_new_tokens is not None else context_length - len(ids)
    for _ in range(max(0, budget)):
        window = ids[-context_length:]
        inputs = torch.tensor([window], dtype=torch.long, device=device)
        logits = model(inputs)
        next_id = int(torch.argmax(logits[0, -1]).item())
        ids.append(next_id)
        if next_id == tokenizer.newline_id:
            break

    return ids


def generate_text(
    model: torch.nn.Module,
    tokenizer: ArithmeticTokenizer,
    prompt: str,
    context_length: int,
    device: torch.device,
    max_new_tokens: int | None = None,
) -> str:
    return tokenizer.decode(generate_ids(model, tokenizer, prompt, context_length, device, max_new_tokens))

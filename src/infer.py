from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from src.data import (
    ArithmeticProblem,
    Operation,
    NumberFormat,
    canonical_answer,
    format_prompt,
    parse_formatted_answer,
)
from src.generate import generate_text
from src.model import AdditionTransformer, ModelConfig
from src.tokenizer import ArithmeticTokenizer
from src.train import select_device


ARITHMETIC_PATTERN = re.compile(r"(?<!\d)(\d+)\s*([+\-*/])\s*(\d+)(?!\d)")


@dataclass(frozen=True)
class AdditionPrediction:
    a: int
    operation: Operation
    b: int
    answer: str
    expected: str
    correct: bool
    prompt: str
    generated: str
    raw_answer: str


def parse_addition_query(text: str, digit_length: int) -> tuple[int, int]:
    a, operation, b = parse_arithmetic_query(text, digit_length)
    if operation != "+":
        raise ValueError("enter one addition expression like 37+48")
    return a, b


def parse_arithmetic_query(text: str, digit_length: int) -> ArithmeticProblem:
    match = ARITHMETIC_PATTERN.search(text)
    if not match:
        raise ValueError("enter one expression like 37+48, 37-18, 12*9, or 80/7")

    if ARITHMETIC_PATTERN.search(text, match.end()):
        raise ValueError("enter only one arithmetic expression")

    a = int(match.group(1))
    operation = match.group(2)
    b = int(match.group(3))
    upper = (10**digit_length) - 1
    if a > upper or b > upper:
        raise ValueError(
            f"this checkpoint supports {digit_length}-digit operands only; enter numbers between 0 and {upper}"
        )
    return a, operation, b  # type: ignore[return-value]


def load_checkpoint_model(
    checkpoint_path: Path,
    device_preference: str = "auto",
) -> tuple[AdditionTransformer, ArithmeticTokenizer, dict[str, Any], torch.device]:
    device = select_device(device_preference)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    tokenizer = ArithmeticTokenizer(checkpoint.get("vocab"))

    model_config = ModelConfig(**checkpoint["model_config"])
    model = AdditionTransformer(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, tokenizer, checkpoint, device


@torch.no_grad()
def predict_arithmetic(
    model: torch.nn.Module,
    tokenizer: ArithmeticTokenizer,
    device: torch.device,
    a: int,
    operation: Operation,
    b: int,
    digit_length: int,
    context_length: int,
    number_format: NumberFormat,
) -> AdditionPrediction:
    prompt = format_prompt(a, b, digit_length, number_format, operation)
    generated = generate_text(model, tokenizer, prompt, context_length, device)
    raw_answer = generated[len(prompt) :].split("\n", maxsplit=1)[0]
    answer = parse_formatted_answer(raw_answer, number_format, operation)
    expected = canonical_answer(a, b, operation)
    return AdditionPrediction(
        a=a,
        operation=operation,
        b=b,
        answer=answer,
        expected=expected,
        correct=answer == expected,
        prompt=prompt,
        generated=generated,
        raw_answer=raw_answer,
    )


def predict_addition(
    model: torch.nn.Module,
    tokenizer: ArithmeticTokenizer,
    device: torch.device,
    a: int,
    b: int,
    digit_length: int,
    context_length: int,
    number_format: NumberFormat,
) -> AdditionPrediction:
    return predict_arithmetic(model, tokenizer, device, a, "+", b, digit_length, context_length, number_format)


class AdditionChatModel:
    def __init__(
        self,
        checkpoint_path: Path,
        digit_length: int = 2,
        number_format: NumberFormat = "fixed_reversed",
        device_preference: str = "auto",
    ) -> None:
        self.model, self.tokenizer, self.checkpoint, self.device = load_checkpoint_model(
            checkpoint_path,
            device_preference,
        )
        self.digit_length = digit_length
        self.number_format = number_format
        self.context_length = int(self.checkpoint["model_config"]["context_length"])

    def ask(self, text: str) -> AdditionPrediction:
        a, operation, b = parse_arithmetic_query(text, self.digit_length)
        return predict_arithmetic(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            a=a,
            operation=operation,
            b=b,
            digit_length=self.digit_length,
            context_length=self.context_length,
            number_format=self.number_format,
        )

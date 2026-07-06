from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Literal

import torch

from src.tokenizer import ArithmeticTokenizer


HardCase = tuple[int, int]
NumberFormat = Literal["normal", "fixed", "reversed", "fixed_reversed"]
NUMBER_FORMATS: tuple[NumberFormat, ...] = ("normal", "fixed", "reversed", "fixed_reversed")
Operation = Literal["+", "-", "*", "/"]
OPERATIONS: tuple[Operation, ...] = ("+", "-", "*", "/")
OperandSampling = Literal["uniform", "diverse"]
OPERAND_SAMPLING_MODES: tuple[OperandSampling, ...] = ("uniform", "diverse")
ArithmeticProblem = tuple[int, Operation, int]
OperationWeights = tuple[float, ...]


def max_addition_sequence_length(digit_length: int) -> int:
    if digit_length < 1:
        raise ValueError("digit_length must be >= 1")
    return (digit_length * 2) + (digit_length + 1) + 3


def max_arithmetic_sequence_length(digit_length: int) -> int:
    if digit_length < 1:
        raise ValueError("digit_length must be >= 1")

    prompt_length = (digit_length * 2) + 2
    answer_length = max(
        digit_length + 1,
        digit_length + 1,
        digit_length * 2,
        (digit_length * 2) + 1,
        3,
    )
    return prompt_length + answer_length + 1


def validate_context_length(digit_length: int, context_length: int) -> None:
    required = max_arithmetic_sequence_length(digit_length) - 1
    if context_length < required:
        raise ValueError(
            f"context_length={context_length} is too small for digit_length={digit_length}; "
            f"need at least {required}"
        )


def random_addition_pair(digit_length: int, rng: random.Random) -> tuple[int, int]:
    if digit_length < 1:
        raise ValueError("digit_length must be >= 1")
    upper = (10**digit_length) - 1
    return rng.randint(0, upper), rng.randint(0, upper)


def random_addition_diverse_pair(digit_length: int, rng: random.Random) -> tuple[int, int]:
    upper = (10**digit_length) - 1
    if digit_length != 2:
        return random_addition_pair(digit_length, rng)

    pattern = rng.random()
    if pattern < 0.35:
        a = (rng.randint(0, 9) * 10) + rng.randint(1, 9)
        b_ones = rng.randint(10 - (a % 10), 9)
        b = rng.randint(0, 9) * 10 + b_ones
        return a, min(b, upper)
    if pattern < 0.55:
        a = rng.randint(0, upper)
        b_ones = rng.randint(0, max(0, 9 - (a % 10)))
        b = rng.randint(0, 9) * 10 + b_ones
        return a, min(b, upper)
    return random_addition_pair(digit_length, rng)


def random_subtraction_diverse_pair(digit_length: int, rng: random.Random) -> tuple[int, int]:
    upper = (10**digit_length) - 1
    pattern = rng.random()
    if pattern < 0.45:
        a = rng.randint(0, upper)
        return a, rng.randint(0, a)
    if pattern < 0.9:
        b = rng.randint(0, upper)
        return rng.randint(0, b), b
    return random_addition_pair(digit_length, rng)


def random_multiplication_diverse_pair(digit_length: int, rng: random.Random) -> tuple[int, int]:
    upper = (10**digit_length) - 1
    small_upper = min(9, upper)
    pattern = rng.random()
    if pattern < 0.25:
        return rng.randint(0, small_upper), rng.randint(0, small_upper)
    if pattern < 0.6:
        small = rng.randint(0, small_upper)
        full = rng.randint(0, upper)
        return (small, full) if rng.random() < 0.5 else (full, small)
    if pattern < 0.8 and upper >= 10:
        round_values = [value for value in range(0, upper + 1, 10)]
        return rng.choice(round_values), rng.randint(0, upper)
    return random_addition_pair(digit_length, rng)


def random_division_diverse_pair(digit_length: int, rng: random.Random) -> tuple[int, int]:
    upper = (10**digit_length) - 1
    pattern = rng.random()
    if pattern < 0.08:
        return rng.randint(0, upper), 0
    if pattern < 0.48:
        divisor = rng.randint(1, upper)
        quotient = rng.randint(0, upper // divisor)
        return divisor * quotient, divisor
    if pattern < 0.78:
        divisor = rng.randint(1, upper)
        quotient = rng.randint(0, upper // divisor)
        remainder = rng.randint(0, divisor - 1)
        return min((divisor * quotient) + remainder, upper), divisor
    return rng.randint(0, upper), rng.randint(1, upper)


def random_arithmetic_pair(
    digit_length: int,
    operation: Operation,
    rng: random.Random,
    operand_sampling: OperandSampling = "uniform",
) -> tuple[int, int]:
    if operand_sampling == "uniform":
        return random_addition_pair(digit_length, rng)
    if operand_sampling != "diverse":
        raise ValueError(f"unknown operand_sampling: {operand_sampling}")
    if operation == "+":
        return random_addition_diverse_pair(digit_length, rng)
    if operation == "-":
        return random_subtraction_diverse_pair(digit_length, rng)
    if operation == "*":
        return random_multiplication_diverse_pair(digit_length, rng)
    if operation == "/":
        return random_division_diverse_pair(digit_length, rng)
    raise ValueError(f"unknown operation: {operation}")


def parse_operations(operations_text: str) -> tuple[Operation, ...]:
    operations: list[Operation] = []
    for raw in operations_text.split(","):
        op = raw.strip()
        if not op:
            continue
        if op not in OPERATIONS:
            raise ValueError(f"unknown operation: {op}")
        operations.append(op)  # type: ignore[arg-type]
    if not operations:
        raise ValueError("at least one operation is required")
    return tuple(dict.fromkeys(operations))


def parse_operation_weights(
    weights_text: str | None,
    operations: tuple[Operation, ...],
) -> OperationWeights | None:
    if weights_text is None or not weights_text.strip():
        return None

    raw_parts = [part.strip() for part in weights_text.split(",") if part.strip()]
    if not raw_parts:
        return None

    if any(":" in part for part in raw_parts):
        weights_by_operation: dict[str, float] = {}
        for part in raw_parts:
            if ":" not in part:
                raise ValueError("operation weights must be all positional or all op:value pairs")
            raw_operation, raw_weight = part.split(":", maxsplit=1)
            operation = raw_operation.strip()
            if operation not in operations:
                raise ValueError(f"operation weight provided for inactive operation: {operation}")
            weights_by_operation[operation] = float(raw_weight)
        weights = tuple(weights_by_operation.get(operation, 0.0) for operation in operations)
    else:
        if len(raw_parts) != len(operations):
            raise ValueError(
                f"positional operation weights must match operations count: "
                f"{len(raw_parts)} weights for {len(operations)} operations"
            )
        weights = tuple(float(part) for part in raw_parts)

    if any(weight < 0.0 for weight in weights):
        raise ValueError("operation weights must be non-negative")
    if sum(weights) <= 0.0:
        raise ValueError("at least one operation weight must be positive")
    return weights


def choose_operation(
    rng: random.Random,
    operations: tuple[Operation, ...],
    operation_weights: OperationWeights | None = None,
) -> Operation:
    if operation_weights is None:
        return rng.choice(operations)
    if len(operation_weights) != len(operations):
        raise ValueError("operation_weights must match operations length")
    return rng.choices(operations, weights=operation_weights, k=1)[0]


def random_arithmetic_problem(
    digit_length: int,
    rng: random.Random,
    operations: tuple[Operation, ...] = ("+",),
    operation_weights: OperationWeights | None = None,
    operand_sampling: OperandSampling = "uniform",
) -> ArithmeticProblem:
    operation = choose_operation(rng, operations, operation_weights)
    a, b = random_arithmetic_pair(digit_length, operation, rng, operand_sampling)
    return a, operation, b


def format_number(value: int, width: int, number_format: NumberFormat) -> str:
    if value < 0:
        raise ValueError("value must be non-negative")
    if number_format not in NUMBER_FORMATS:
        raise ValueError(f"unknown number_format: {number_format}")

    fixed = number_format in {"fixed", "fixed_reversed"}
    reversed_digits = number_format in {"reversed", "fixed_reversed"}
    text = f"{value:0{width}d}" if fixed else str(value)
    return text[::-1] if reversed_digits else text


def format_answer(value: int, digit_length: int, number_format: NumberFormat = "normal") -> str:
    return format_number(value, digit_length + 1, number_format)


def format_signed_number(value: int, width: int, number_format: NumberFormat) -> str:
    if value < 0:
        return "-" + format_number(abs(value), width, number_format)
    return format_number(value, width, number_format)


def format_prompt(
    a: int,
    b: int,
    digit_length: int,
    number_format: NumberFormat = "normal",
    operation: Operation = "+",
) -> str:
    left = format_number(a, digit_length, number_format)
    right = format_number(b, digit_length, number_format)
    return f"{left}{operation}{right}="


def canonical_answer(a: int, b: int, operation: Operation) -> str:
    if operation == "+":
        return str(a + b)
    if operation == "-":
        return str(a - b)
    if operation == "*":
        return str(a * b)
    if operation == "/":
        if b == 0:
            return "ERR"
        return f"{a // b}R{a % b}"
    raise ValueError(f"unknown operation: {operation}")


def format_operation_answer(
    a: int,
    b: int,
    digit_length: int,
    number_format: NumberFormat = "normal",
    operation: Operation = "+",
) -> str:
    if operation == "+":
        return format_number(a + b, digit_length + 1, number_format)
    if operation == "-":
        return format_signed_number(a - b, digit_length, number_format)
    if operation == "*":
        return format_number(a * b, digit_length * 2, number_format)
    if operation == "/":
        if b == 0:
            return "ERR"
        quotient = format_number(a // b, digit_length, number_format)
        remainder = format_number(a % b, digit_length, number_format)
        return f"{quotient}R{remainder}"
    raise ValueError(f"unknown operation: {operation}")


def format_addition(
    a: int,
    b: int,
    digit_length: int,
    number_format: NumberFormat = "normal",
) -> str:
    return format_arithmetic(a, "+", b, digit_length, number_format)


def format_arithmetic(
    a: int,
    operation: Operation,
    b: int,
    digit_length: int,
    number_format: NumberFormat = "normal",
) -> str:
    return (
        f"{format_prompt(a, b, digit_length, number_format, operation)}"
        f"{format_operation_answer(a, b, digit_length, number_format, operation)}\n"
    )


def parse_formatted_number(answer: str, number_format: NumberFormat = "normal") -> str:
    if number_format not in NUMBER_FORMATS:
        raise ValueError(f"unknown number_format: {number_format}")
    if not answer or not answer.isdigit():
        return ""

    canonical = answer[::-1] if number_format in {"reversed", "fixed_reversed"} else answer
    return str(int(canonical))


def parse_formatted_answer(
    answer: str,
    number_format: NumberFormat = "normal",
    operation: Operation = "+",
) -> str:
    if answer == "ERR":
        return "ERR"
    if operation == "/":
        if "R" not in answer:
            return ""
        quotient, remainder = answer.split("R", maxsplit=1)
        parsed_quotient = parse_formatted_number(quotient, number_format)
        parsed_remainder = parse_formatted_number(remainder, number_format)
        if not parsed_quotient or not parsed_remainder:
            return ""
        return f"{parsed_quotient}R{parsed_remainder}"
    if operation == "-":
        sign = "-" if answer.startswith("-") else ""
        unsigned = answer[1:] if sign else answer
        parsed = parse_formatted_number(unsigned, number_format)
        if not parsed:
            return ""
        return str(int(f"{sign}{parsed}"))
    return parse_formatted_number(answer, number_format)


def load_hard_cases(path: Path) -> list[HardCase]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    errors = payload.get("errors")
    if not isinstance(errors, list):
        raise ValueError(f"hard-case file must contain an 'errors' list: {path}")

    hard_cases: list[HardCase] = []
    for index, error in enumerate(errors):
        if not isinstance(error, dict) or "a" not in error or "b" not in error:
            raise ValueError(f"hard-case entry {index} must contain 'a' and 'b'")
        a = int(error["a"])
        b = int(error["b"])
        hard_cases.append((a, b))
    if not hard_cases:
        raise ValueError(f"hard-case file contains no errors: {path}")
    return hard_cases


def mixed_addition_pairs(
    batch_size: int,
    digit_length: int,
    rng: random.Random,
    hard_cases: list[HardCase] | None = None,
    hard_case_ratio: float = 0.0,
) -> list[HardCase]:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if not 0.0 <= hard_case_ratio <= 1.0:
        raise ValueError("hard_case_ratio must be between 0 and 1")

    hard_count = 0
    if hard_cases:
        hard_count = round(batch_size * hard_case_ratio)
    random_count = batch_size - hard_count

    pairs = [rng.choice(hard_cases) for _ in range(hard_count)] if hard_count else []
    pairs.extend(random_addition_pair(digit_length, rng) for _ in range(random_count))
    rng.shuffle(pairs)
    return pairs


def mixed_arithmetic_problems(
    batch_size: int,
    digit_length: int,
    rng: random.Random,
    operations: tuple[Operation, ...] = ("+",),
    operation_weights: OperationWeights | None = None,
    operand_sampling: OperandSampling = "uniform",
    hard_cases: list[HardCase] | None = None,
    hard_case_ratio: float = 0.0,
) -> list[ArithmeticProblem]:
    hard_count = round(batch_size * hard_case_ratio) if hard_cases else 0
    problems: list[ArithmeticProblem] = []
    if hard_count:
        problems.extend((a, "+", b) for a, b in mixed_addition_pairs(hard_count, digit_length, rng, hard_cases, 1.0))
    problems.extend(
        random_arithmetic_problem(digit_length, rng, operations, operation_weights, operand_sampling)
        for _ in range(batch_size - hard_count)
    )
    rng.shuffle(problems)
    return problems


def encode_padded_example(
    tokenizer: ArithmeticTokenizer,
    text: str,
    context_length: int,
) -> tuple[list[int], list[int]]:
    token_ids = tokenizer.encode(text)
    total_length = context_length + 1
    if len(token_ids) > total_length:
        raise ValueError(f"encoded example length {len(token_ids)} exceeds limit {total_length}: {text!r}")

    padded = token_ids + [tokenizer.pad_id] * (total_length - len(token_ids))
    return padded[:-1], padded[1:]


def make_batch(
    tokenizer: ArithmeticTokenizer,
    batch_size: int,
    context_length: int,
    digit_length: int,
    device: torch.device,
    rng: random.Random,
    hard_cases: list[HardCase] | None = None,
    hard_case_ratio: float = 0.0,
    number_format: NumberFormat = "normal",
    operations: tuple[Operation, ...] = ("+",),
    operation_weights: OperationWeights | None = None,
    operand_sampling: OperandSampling = "uniform",
) -> tuple[torch.Tensor, torch.Tensor]:
    validate_context_length(digit_length, context_length)
    inputs: list[list[int]] = []
    targets: list[list[int]] = []

    for a, operation, b in mixed_arithmetic_problems(
        batch_size,
        digit_length,
        rng,
        operations,
        operation_weights,
        operand_sampling,
        hard_cases,
        hard_case_ratio,
    ):
        x, y = encode_padded_example(
            tokenizer,
            format_arithmetic(a, operation, b, digit_length, number_format),
            context_length,
        )
        inputs.append(x)
        targets.append(y)

    return (
        torch.tensor(inputs, dtype=torch.long, device=device),
        torch.tensor(targets, dtype=torch.long, device=device),
    )

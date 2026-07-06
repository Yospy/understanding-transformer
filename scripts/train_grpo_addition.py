from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import (
    NUMBER_FORMATS,
    OPERAND_SAMPLING_MODES,
    OPERATIONS,
    NumberFormat,
    OperandSampling,
    Operation,
    OperationWeights,
    format_prompt,
    mixed_arithmetic_problems,
    parse_operation_weights,
    parse_operations,
)
from src.grpo import (
    build_completion_logprob_batch,
    completion_token_logprobs,
    grpo_loss,
    group_relative_advantages,
    score_arithmetic_completion,
)
from src.model import AdditionTransformer, ModelConfig, count_parameters
from src.sampling import sample_completions
from src.tokenizer import ArithmeticTokenizer
from src.train import exact_answer_accuracy, select_device


@dataclass(frozen=True)
class GRPOTrainConfig:
    checkpoint: str
    steps: int = 100
    prompts_per_step: int = 16
    group_size: int = 8
    optimization_epochs: int = 1
    learning_rate: float = 1e-5
    weight_decay: float = 0.0
    beta: float = 0.02
    clip_epsilon: float = 0.2
    grad_clip: float = 1.0
    temperature: float = 1.0
    max_new_tokens: int | None = None
    digit_length: int = 2
    number_format: NumberFormat = "normal"
    operations: tuple[Operation, ...] = ("+",)
    operation_weights: OperationWeights | None = None
    operand_sampling: OperandSampling = "uniform"
    hard_case_file: str | None = None
    hard_case_ratio: float = 0.0
    eval_interval: int = 10
    accuracy_examples: int = 200
    save_interval: int = 50
    seed: int = 2026
    device: str = "auto"
    log_sample_groups: int = 2


GRPOHardCase = tuple[int, Operation, int]
OPERATION_METRIC_LABELS: dict[Operation, str] = {
    "+": "plus",
    "-": "minus",
    "*": "mul",
    "/": "div",
}
OPERATION_METRICS = ("reward_mean", "pass_at_k", "active_group_rate", "valid_rate")


METRIC_FIELDNAMES = [
    "step",
    "loss",
    "policy_loss",
    "kl_loss",
    "clip_fraction",
    "mean_ratio",
    "grad_norm",
    "reward_mean",
    "reward_std",
    "reward_min",
    "reward_max",
    "pass_at_k",
    "all_zero_group_rate",
    "active_group_rate",
    "valid_rate",
    "invalid_rate",
    "avg_completion_tokens",
    "advantage_abs_mean",
    "greedy_accuracy",
    "elapsed_s",
    *[
        f"{metric}_{label}"
        for label in OPERATION_METRIC_LABELS.values()
        for metric in OPERATION_METRICS
    ],
]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run GRPO on the current 2-digit arithmetic transformer capability."
    )
    parser.add_argument("--checkpoint", type=Path, default=Path("final/checkpoint_best_val.pt"))
    parser.add_argument("--steps", type=int, default=GRPOTrainConfig.steps)
    parser.add_argument("--prompts-per-step", type=int, default=GRPOTrainConfig.prompts_per_step)
    parser.add_argument("--group-size", type=int, default=GRPOTrainConfig.group_size)
    parser.add_argument("--optimization-epochs", type=int, default=GRPOTrainConfig.optimization_epochs)
    parser.add_argument("--learning-rate", type=float, default=GRPOTrainConfig.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=GRPOTrainConfig.weight_decay)
    parser.add_argument("--beta", type=float, default=GRPOTrainConfig.beta)
    parser.add_argument("--clip-epsilon", type=float, default=GRPOTrainConfig.clip_epsilon)
    parser.add_argument("--grad-clip", type=float, default=GRPOTrainConfig.grad_clip)
    parser.add_argument("--temperature", type=float, default=GRPOTrainConfig.temperature)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--digit-length", type=int, default=GRPOTrainConfig.digit_length)
    parser.add_argument("--number-format", choices=NUMBER_FORMATS, default=GRPOTrainConfig.number_format)
    parser.add_argument("--operations", type=str, default=",".join(GRPOTrainConfig.operations))
    parser.add_argument("--operation-weights", type=str, default=None)
    parser.add_argument(
        "--operand-sampling",
        choices=OPERAND_SAMPLING_MODES,
        default=GRPOTrainConfig.operand_sampling,
    )
    parser.add_argument("--hard-case-file", type=Path, default=None)
    parser.add_argument("--hard-case-ratio", type=float, default=GRPOTrainConfig.hard_case_ratio)
    parser.add_argument("--eval-interval", type=int, default=GRPOTrainConfig.eval_interval)
    parser.add_argument("--accuracy-examples", type=int, default=GRPOTrainConfig.accuracy_examples)
    parser.add_argument("--save-interval", type=int, default=GRPOTrainConfig.save_interval)
    parser.add_argument("--seed", type=int, default=GRPOTrainConfig.seed)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default=GRPOTrainConfig.device)
    parser.add_argument("--log-sample-groups", type=int, default=GRPOTrainConfig.log_sample_groups)
    parser.add_argument("--run-dir", type=Path, default=None)
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> GRPOTrainConfig:
    if args.steps < 1:
        raise ValueError("steps must be >= 1")
    if args.prompts_per_step < 1:
        raise ValueError("prompts_per_step must be >= 1")
    if args.group_size < 2:
        raise ValueError("group_size must be >= 2 for group-relative advantages")
    if args.optimization_epochs < 1:
        raise ValueError("optimization_epochs must be >= 1")
    if args.temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if not 0.0 <= args.hard_case_ratio <= 1.0:
        raise ValueError("hard_case_ratio must be between 0 and 1")

    operations = parse_operations(args.operations)

    return GRPOTrainConfig(
        checkpoint=str(args.checkpoint),
        steps=args.steps,
        prompts_per_step=args.prompts_per_step,
        group_size=args.group_size,
        optimization_epochs=args.optimization_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        beta=args.beta,
        clip_epsilon=args.clip_epsilon,
        grad_clip=args.grad_clip,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        digit_length=args.digit_length,
        number_format=args.number_format,
        operations=operations,
        operation_weights=parse_operation_weights(args.operation_weights, operations),
        operand_sampling=args.operand_sampling,
        hard_case_file=str(args.hard_case_file) if args.hard_case_file else None,
        hard_case_ratio=args.hard_case_ratio,
        eval_interval=args.eval_interval,
        accuracy_examples=args.accuracy_examples,
        save_interval=args.save_interval,
        seed=args.seed,
        device=args.device,
        log_sample_groups=args.log_sample_groups,
    )


def default_run_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path("runs") / "stage8-multiop-grpo" / timestamp


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_metrics(path: Path, rows: list[dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=METRIC_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def append_sample_groups(path: Path, step: int, groups: list[dict[str, Any]], limit: int) -> None:
    if limit <= 0:
        return
    with path.open("a", encoding="utf-8") as file:
        for group in groups[:limit]:
            file.write(json.dumps({"step": step, **group}, sort_keys=True) + "\n")


def load_policy_and_reference(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[AdditionTransformer, AdditionTransformer, ArithmeticTokenizer, dict[str, Any], ModelConfig]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    tokenizer = ArithmeticTokenizer(checkpoint.get("vocab"))
    model_config = ModelConfig(**checkpoint["model_config"])

    policy_model = AdditionTransformer(model_config).to(device)
    reference_model = AdditionTransformer(model_config).to(device)
    policy_model.load_state_dict(checkpoint["model_state_dict"])
    reference_model.load_state_dict(checkpoint["model_state_dict"])

    reference_model.eval()
    for parameter in reference_model.parameters():
        parameter.requires_grad_(False)

    return policy_model, reference_model, tokenizer, checkpoint, model_config


def load_grpo_hard_cases(path: Path) -> list[GRPOHardCase]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    errors = payload.get("errors")
    if not isinstance(errors, list):
        raise ValueError(f"hard-case file must contain an 'errors' list: {path}")

    hard_cases: list[GRPOHardCase] = []
    for index, error in enumerate(errors):
        if not isinstance(error, dict) or "a" not in error or "b" not in error:
            raise ValueError(f"hard-case entry {index} must contain 'a' and 'b'")
        operation = str(error.get("operation", "+"))
        if operation not in OPERATIONS:
            raise ValueError(f"hard-case entry {index} has unknown operation: {operation}")
        hard_cases.append((int(error["a"]), operation, int(error["b"])))  # type: ignore[arg-type]

    if not hard_cases:
        raise ValueError(f"hard-case file contains no errors: {path}")
    return hard_cases


def sample_rollout_problems(
    config: GRPOTrainConfig,
    rng: random.Random,
    hard_cases: list[GRPOHardCase] | None,
) -> list[GRPOHardCase]:
    eligible_hard_cases = [case for case in hard_cases or [] if case[1] in config.operations]
    if hard_cases and not eligible_hard_cases:
        raise ValueError("hard-case file contains no cases for the active operations")

    hard_count = round(config.prompts_per_step * config.hard_case_ratio) if eligible_hard_cases else 0
    random_count = config.prompts_per_step - hard_count

    problems = [rng.choice(eligible_hard_cases) for _ in range(hard_count)]
    problems.extend(
        mixed_arithmetic_problems(
            batch_size=random_count,
            digit_length=config.digit_length,
            rng=rng,
            operations=config.operations,
            operation_weights=config.operation_weights,
            operand_sampling=config.operand_sampling,
        )
    )
    rng.shuffle(problems)
    return problems


def collect_rollouts(
    policy_model: AdditionTransformer,
    tokenizer: ArithmeticTokenizer,
    device: torch.device,
    config: GRPOTrainConfig,
    rng: random.Random,
    generator: torch.Generator,
    hard_cases: list[GRPOHardCase] | None,
) -> tuple[list[list[int]], list[list[int]], torch.Tensor, torch.Tensor, list[int], list[dict[str, Any]]]:
    problems = sample_rollout_problems(config, rng, hard_cases)

    prompt_token_ids: list[list[int]] = []
    completion_token_ids: list[list[int]] = []
    reward_groups: list[list[float]] = []
    valid_groups: list[list[float]] = []
    completion_lengths: list[int] = []
    sample_groups: list[dict[str, Any]] = []

    for a, operation, b in problems:
        prompt = format_prompt(a, b, config.digit_length, config.number_format, operation)
        prompt_ids = tokenizer.encode(prompt)
        completions = sample_completions(
            model=policy_model,
            tokenizer=tokenizer,
            prompt=prompt,
            context_length=policy_model.config.context_length,
            device=device,
            count=config.group_size,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            generator=generator,
        )

        group_rewards: list[float] = []
        group_valid: list[float] = []
        samples: list[dict[str, Any]] = []
        expected = ""

        for completion in completions:
            reward = score_arithmetic_completion(a, b, operation, completion.text, config.number_format)
            expected = reward.expected
            prompt_token_ids.append(prompt_ids)
            completion_token_ids.append(completion.token_ids)
            completion_lengths.append(len(completion.token_ids))
            group_rewards.append(reward.reward)
            group_valid.append(1.0 if reward.valid else 0.0)
            samples.append(
                {
                    "text": completion.text,
                    "raw_answer": reward.raw_answer,
                    "parsed_answer": reward.parsed_answer,
                    "reward": reward.reward,
                    "valid": reward.valid,
                    "ended_with_newline": completion.ended_with_newline,
                    "tokens": completion.token_ids,
                }
            )

        reward_groups.append(group_rewards)
        valid_groups.append(group_valid)
        sample_groups.append(
            {
                "a": a,
                "b": b,
                "operation": operation,
                "prompt": prompt,
                "expected": expected,
                "samples": samples,
            }
        )

    return (
        prompt_token_ids,
        completion_token_ids,
        torch.tensor(reward_groups, dtype=torch.float32, device=device),
        torch.tensor(valid_groups, dtype=torch.float32, device=device),
        completion_lengths,
        sample_groups,
    )


def checkpoint_payload(
    model: AdditionTransformer,
    optimizer: torch.optim.Optimizer,
    config: GRPOTrainConfig,
    model_config: ModelConfig,
    tokenizer: ArithmeticTokenizer,
    step: int,
    metrics_row: dict[str, float],
    source_checkpoint: dict[str, Any],
) -> dict[str, Any]:
    return {
        "step": step,
        "metrics": metrics_row,
        "grpo_config": asdict(config),
        "source_checkpoint_step": source_checkpoint.get("step"),
        "source_checkpoint_metrics": source_checkpoint.get("metrics"),
        "model_config": asdict(model_config),
        "parameter_count": count_parameters(model),
        "vocab": tokenizer.tokens,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }


def save_checkpoint(
    path: Path,
    model: AdditionTransformer,
    optimizer: torch.optim.Optimizer,
    config: GRPOTrainConfig,
    model_config: ModelConfig,
    tokenizer: ArithmeticTokenizer,
    step: int,
    metrics_row: dict[str, float],
    source_checkpoint: dict[str, Any],
) -> None:
    torch.save(
        checkpoint_payload(
            model=model,
            optimizer=optimizer,
            config=config,
            model_config=model_config,
            tokenizer=tokenizer,
            step=step,
            metrics_row=metrics_row,
            source_checkpoint=source_checkpoint,
        ),
        path,
    )


def clean_for_log(text: str) -> str:
    return text.replace("\n", "\\n") if text else "<empty>"


def print_sample_groups(step: int, groups: list[dict[str, Any]], limit: int) -> None:
    for group in groups[: max(0, limit)]:
        rendered_samples = ", ".join(
            (
                f"{clean_for_log(sample['raw_answer'])}"
                f"|parsed={clean_for_log(sample['parsed_answer'])}"
                f"|r={sample['reward']:.0f}"
            )
            for sample in group["samples"]
        )
        print(
            f"sample_group step={step:04d} op={group['operation']} prompt={group['prompt']} "
            f"expected={group['expected']} samples=[{rendered_samples}]"
        )


def per_operation_metrics(
    rewards: torch.Tensor,
    valid: torch.Tensor,
    sample_groups: list[dict[str, Any]],
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for operation, label in OPERATION_METRIC_LABELS.items():
        indexes = [index for index, group in enumerate(sample_groups) if group["operation"] == operation]
        if not indexes:
            for metric in OPERATION_METRICS:
                metrics[f"{metric}_{label}"] = float("nan")
            continue

        index_tensor = torch.tensor(indexes, dtype=torch.long, device=rewards.device)
        op_rewards = rewards.index_select(0, index_tensor)
        op_valid = valid.index_select(0, index_tensor)
        op_active = op_rewards.std(dim=1, unbiased=False) > 0
        op_pass_at_k = (op_rewards.max(dim=1).values > 0).float().mean()

        metrics[f"reward_mean_{label}"] = float(op_rewards.mean().item())
        metrics[f"pass_at_k_{label}"] = float(op_pass_at_k.item())
        metrics[f"active_group_rate_{label}"] = float(op_active.float().mean().item())
        metrics[f"valid_rate_{label}"] = float(op_valid.mean().item())
    return metrics


def main() -> None:
    args = parse_args()
    config = build_config(args)

    torch.manual_seed(config.seed)
    random.seed(config.seed)

    run_dir = args.run_dir or default_run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)

    device = select_device(config.device)
    policy_model, reference_model, tokenizer, source_checkpoint, model_config = load_policy_and_reference(
        Path(config.checkpoint),
        device,
    )
    hard_cases = load_grpo_hard_cases(Path(config.hard_case_file)) if config.hard_case_file else None

    optimizer = torch.optim.AdamW(
        policy_model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    rollout_rng = random.Random(config.seed)
    eval_rng = random.Random(config.seed + 1)
    torch_generator = torch.Generator().manual_seed(config.seed)

    write_json(
        run_dir / "config.json",
        {
            "grpo": asdict(config),
            "model": asdict(model_config),
            "parameter_count": count_parameters(policy_model),
            "device": str(device),
            "vocab": tokenizer.tokens,
            "source_checkpoint": str(config.checkpoint),
            "source_checkpoint_step": source_checkpoint.get("step"),
            "source_checkpoint_metrics": source_checkpoint.get("metrics"),
        },
    )

    print(f"run_dir={run_dir}")
    print(f"device={device}")
    print(f"checkpoint={config.checkpoint}")
    print(
        "grpo_settings="
        f"steps={config.steps} prompts_per_step={config.prompts_per_step} "
        f"group_size={config.group_size} temperature={config.temperature} "
        f"lr={config.learning_rate} beta={config.beta} clip_epsilon={config.clip_epsilon} "
        f"operations={','.join(config.operations)} operand_sampling={config.operand_sampling} "
        f"operation_weights={config.operation_weights}"
    )
    if hard_cases:
        print(f"hard_cases={len(hard_cases)} hard_case_ratio={config.hard_case_ratio}")

    history: list[dict[str, float]] = []
    best_greedy_accuracy = float("-inf")
    best_checkpoint_path = run_dir / "checkpoint_best_greedy.pt"
    final_checkpoint_path = run_dir / "checkpoint_final.pt"
    sample_groups_path = run_dir / "sample_groups.jsonl"
    start = time.perf_counter()

    for step in range(1, config.steps + 1):
        (
            prompt_token_ids,
            completion_token_ids,
            rewards,
            valid,
            completion_lengths,
            sample_groups,
        ) = collect_rollouts(
            policy_model=policy_model,
            tokenizer=tokenizer,
            device=device,
            config=config,
            rng=rollout_rng,
            generator=torch_generator,
            hard_cases=hard_cases,
        )

        advantages = group_relative_advantages(rewards).reshape(-1)
        batch = build_completion_logprob_batch(
            tokenizer=tokenizer,
            prompt_token_ids=prompt_token_ids,
            completion_token_ids=completion_token_ids,
            device=device,
        )

        with torch.no_grad():
            old_logprobs = completion_token_logprobs(policy_model, batch).detach()
            reference_logprobs = completion_token_logprobs(reference_model, batch).detach()

        last_loss_result = None
        grad_norm = torch.tensor(0.0)
        for _ in range(config.optimization_epochs):
            policy_model.train()
            policy_logprobs = completion_token_logprobs(policy_model, batch)
            loss_result = grpo_loss(
                policy_logprobs=policy_logprobs,
                old_logprobs=old_logprobs,
                reference_logprobs=reference_logprobs,
                advantages=advantages,
                completion_mask=batch.completion_mask,
                beta=config.beta,
                clip_epsilon=config.clip_epsilon,
            )

            optimizer.zero_grad(set_to_none=True)
            loss_result.loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(policy_model.parameters(), config.grad_clip)
            optimizer.step()
            last_loss_result = loss_result

        if last_loss_result is None:
            raise RuntimeError("optimization did not run")

        reward_std_by_group = rewards.std(dim=1, unbiased=False)
        active_groups = reward_std_by_group > 0
        pass_at_k = (rewards.max(dim=1).values > 0).float().mean()
        greedy_accuracy = float("nan")
        if config.accuracy_examples > 0 and (
            step == 1 or step % config.eval_interval == 0 or step == config.steps
        ):
            greedy_accuracy = exact_answer_accuracy(
                model=policy_model,
                tokenizer=tokenizer,
                device=device,
                context_length=model_config.context_length,
                digit_length=config.digit_length,
                examples=config.accuracy_examples,
                rng=eval_rng,
                number_format=config.number_format,
                operations=config.operations,
                operation_weights=config.operation_weights,
                operand_sampling=config.operand_sampling,
            )

        row = {
            "step": float(step),
            "loss": float(last_loss_result.loss.detach().item()),
            "policy_loss": float(last_loss_result.policy_loss.detach().item()),
            "kl_loss": float(last_loss_result.kl_loss.detach().item()),
            "clip_fraction": float(last_loss_result.clip_fraction.detach().item()),
            "mean_ratio": float(last_loss_result.mean_ratio.detach().item()),
            "grad_norm": float(grad_norm.detach().item()),
            "reward_mean": float(rewards.mean().item()),
            "reward_std": float(rewards.std(unbiased=False).item()),
            "reward_min": float(rewards.min().item()),
            "reward_max": float(rewards.max().item()),
            "pass_at_k": float(pass_at_k.item()),
            "all_zero_group_rate": float((rewards.sum(dim=1) == 0).float().mean().item()),
            "active_group_rate": float(active_groups.float().mean().item()),
            "valid_rate": float(valid.mean().item()),
            "invalid_rate": float(1.0 - valid.mean().item()),
            "avg_completion_tokens": float(sum(completion_lengths) / len(completion_lengths)),
            "advantage_abs_mean": float(advantages.abs().mean().item()),
            "greedy_accuracy": greedy_accuracy,
            "elapsed_s": float(time.perf_counter() - start),
        }
        row.update(per_operation_metrics(rewards, valid, sample_groups))
        history.append(row)
        write_metrics(run_dir / "metrics.csv", history)

        should_log = step == 1 or step % config.eval_interval == 0 or step == config.steps
        if should_log:
            print(
                f"step={step:04d} "
                f"loss={row['loss']:.6f} policy_loss={row['policy_loss']:.6f} "
                f"kl={row['kl_loss']:.6f} clip_frac={row['clip_fraction']:.3f} "
                f"ratio={row['mean_ratio']:.3f} grad_norm={row['grad_norm']:.3f} "
                f"reward_mean={row['reward_mean']:.3f} reward_std={row['reward_std']:.3f} "
                f"reward_range=[{row['reward_min']:.0f},{row['reward_max']:.0f}] "
                f"pass@{config.group_size}={row['pass_at_k']:.3f} "
                f"active_groups={row['active_group_rate']:.3f} "
                f"all_zero_groups={row['all_zero_group_rate']:.3f} "
                f"valid={row['valid_rate']:.3f} invalid={row['invalid_rate']:.3f} "
                f"avg_tokens={row['avg_completion_tokens']:.2f} "
                f"adv_abs={row['advantage_abs_mean']:.3f} "
                f"greedy_acc={row['greedy_accuracy']:.3f}"
            )
            print_sample_groups(step, sample_groups, config.log_sample_groups)
            append_sample_groups(sample_groups_path, step, sample_groups, config.log_sample_groups)

        if step % config.save_interval == 0:
            save_checkpoint(
                run_dir / f"checkpoint_step_{step:04d}.pt",
                policy_model,
                optimizer,
                config,
                model_config,
                tokenizer,
                step,
                row,
                source_checkpoint,
            )

        if row["greedy_accuracy"] == row["greedy_accuracy"] and row["greedy_accuracy"] >= best_greedy_accuracy:
            best_greedy_accuracy = row["greedy_accuracy"]
            save_checkpoint(
                best_checkpoint_path,
                policy_model,
                optimizer,
                config,
                model_config,
                tokenizer,
                step,
                row,
                source_checkpoint,
            )

    final = history[-1]
    save_checkpoint(
        final_checkpoint_path,
        policy_model,
        optimizer,
        config,
        model_config,
        tokenizer,
        int(final["step"]),
        final,
        source_checkpoint,
    )
    write_json(
        run_dir / "summary.json",
        {
            "run_dir": str(run_dir),
            "metrics_path": str(run_dir / "metrics.csv"),
            "sample_groups_path": str(sample_groups_path),
            "checkpoint_best_greedy_path": str(best_checkpoint_path),
            "checkpoint_final_path": str(final_checkpoint_path),
            "best_greedy_accuracy": best_greedy_accuracy,
            "final_metrics": final,
        },
    )

    print(f"metrics={run_dir / 'metrics.csv'}")
    print(f"sample_groups={sample_groups_path}")
    print(f"checkpoint_best_greedy={best_checkpoint_path}")
    print(f"checkpoint_final={final_checkpoint_path}")


if __name__ == "__main__":
    main()

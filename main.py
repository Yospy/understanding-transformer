from __future__ import annotations

import argparse
import csv
import json
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import torch

from src.data import (
    NUMBER_FORMATS,
    OPERAND_SAMPLING_MODES,
    HardCase,
    NumberFormat,
    Operation,
    OperandSampling,
    OperationWeights,
    format_prompt,
    load_hard_cases,
    make_batch,
    parse_operation_weights,
    parse_operations,
    random_arithmetic_problem,
    validate_context_length,
)
from src.generate import generate_text
from src.model import AdditionTransformer, ModelConfig, count_parameters
from src.tokenizer import ArithmeticTokenizer
from src.train import (
    TrainConfig,
    exact_answer_accuracy,
    evaluate_loss,
    next_token_loss,
    select_device,
    token_loss_weights,
)


@dataclass(frozen=True)
class ExperimentConfig:
    stage: str = "stage3"
    d_model: int = 128
    num_heads: int = 4
    num_layers: int = 4
    ffn_hidden: int = 512
    context_length: int = 16
    digit_length: int = 2
    batch_size: int = 64
    steps: int = 500
    eval_interval: int = 100
    val_batches: int = 8
    accuracy_examples: int = 25
    learning_rate: float = 3e-3
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    seed: int = 1337
    device: str = "auto"
    samples: int = 8
    hard_case_file: str | None = None
    hard_case_ratio: float = 0.0
    init_checkpoint: str | None = None
    allow_vocab_expansion: bool = False
    number_format: NumberFormat = "normal"
    loss_prompt_weight: float = 1.0
    operations: tuple[Operation, ...] = ("+",)
    operation_weights: OperationWeights | None = None
    operand_sampling: OperandSampling = "uniform"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an addition-transformer experiment.")
    parser.add_argument("--stage", type=str, default=ExperimentConfig.stage)
    parser.add_argument("--d-model", type=int, default=ExperimentConfig.d_model)
    parser.add_argument("--num-heads", type=int, default=ExperimentConfig.num_heads)
    parser.add_argument("--num-layers", type=int, default=ExperimentConfig.num_layers)
    parser.add_argument("--ffn-hidden", type=int, default=ExperimentConfig.ffn_hidden)
    parser.add_argument("--steps", type=int, default=ExperimentConfig.steps)
    parser.add_argument("--batch-size", type=int, default=ExperimentConfig.batch_size)
    parser.add_argument("--eval-interval", type=int, default=ExperimentConfig.eval_interval)
    parser.add_argument("--val-batches", type=int, default=ExperimentConfig.val_batches)
    parser.add_argument("--accuracy-examples", type=int, default=ExperimentConfig.accuracy_examples)
    parser.add_argument("--learning-rate", type=float, default=ExperimentConfig.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=ExperimentConfig.weight_decay)
    parser.add_argument("--grad-clip", type=float, default=ExperimentConfig.grad_clip)
    parser.add_argument("--digit-length", type=int, default=ExperimentConfig.digit_length)
    parser.add_argument("--context-length", type=int, default=ExperimentConfig.context_length)
    parser.add_argument("--seed", type=int, default=ExperimentConfig.seed)
    parser.add_argument("--samples", type=int, default=ExperimentConfig.samples)
    parser.add_argument("--hard-case-file", type=Path, default=None)
    parser.add_argument("--hard-case-ratio", type=float, default=ExperimentConfig.hard_case_ratio)
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--allow-vocab-expansion",
        action="store_true",
        help="load matching non-vocab weights and remap shared token rows from a smaller checkpoint vocab",
    )
    parser.add_argument("--number-format", choices=NUMBER_FORMATS, default=ExperimentConfig.number_format)
    parser.add_argument("--loss-prompt-weight", type=float, default=ExperimentConfig.loss_prompt_weight)
    parser.add_argument("--operations", type=str, default=",".join(ExperimentConfig.operations))
    parser.add_argument(
        "--operation-weights",
        type=str,
        default=None,
        help="optional operation sampling weights, e.g. '+:1,-:3,*:4,/:2' or '1,3,4,2'",
    )
    parser.add_argument(
        "--operand-sampling",
        choices=OPERAND_SAMPLING_MODES,
        default=ExperimentConfig.operand_sampling,
        help="operand distribution for generated arithmetic data",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default=ExperimentConfig.device)
    parser.add_argument("--run-dir", type=Path, default=None)
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> ExperimentConfig:
    operations = parse_operations(args.operations)
    return ExperimentConfig(
        stage=args.stage,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        ffn_hidden=args.ffn_hidden,
        steps=args.steps,
        batch_size=args.batch_size,
        eval_interval=args.eval_interval,
        val_batches=args.val_batches,
        accuracy_examples=args.accuracy_examples,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        digit_length=args.digit_length,
        context_length=args.context_length,
        seed=args.seed,
        device=args.device,
        samples=args.samples,
        hard_case_file=str(args.hard_case_file) if args.hard_case_file else None,
        hard_case_ratio=args.hard_case_ratio,
        init_checkpoint=str(args.init_checkpoint) if args.init_checkpoint else None,
        allow_vocab_expansion=args.allow_vocab_expansion,
        number_format=args.number_format,
        loss_prompt_weight=args.loss_prompt_weight,
        operations=operations,
        operation_weights=parse_operation_weights(args.operation_weights, operations),
        operand_sampling=args.operand_sampling,
    )


def default_run_dir(stage: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    normalized_stage = stage.strip().lower().replace(" ", "-")
    return Path("runs") / normalized_stage / timestamp


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_metrics(path: Path, rows: list[dict[str, float]]) -> None:
    fieldnames = ["step", "train_loss", "val_loss", "exact_accuracy", "elapsed_s"]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_samples(path: Path, samples: list[str]) -> None:
    path.write_text("\n".join(samples) + "\n", encoding="utf-8")


def checkpoint_payload(
    model: AdditionTransformer,
    optimizer: torch.optim.Optimizer,
    config: ExperimentConfig,
    model_config: ModelConfig,
    tokenizer: ArithmeticTokenizer,
    step: int,
    metrics_row: dict[str, float],
) -> dict[str, Any]:
    return {
        "step": step,
        "metrics": metrics_row,
        "experiment_config": asdict(config),
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
    config: ExperimentConfig,
    model_config: ModelConfig,
    tokenizer: ArithmeticTokenizer,
    step: int,
    metrics_row: dict[str, float],
) -> None:
    torch.save(
        checkpoint_payload(model, optimizer, config, model_config, tokenizer, step, metrics_row),
        path,
    )


def load_initial_checkpoint(
    path: Path,
    model: AdditionTransformer,
    model_config: ModelConfig,
    tokenizer: ArithmeticTokenizer,
    device: torch.device,
    allow_vocab_expansion: bool = False,
) -> int:
    checkpoint = torch.load(path, map_location=device)
    checkpoint_model_config = checkpoint.get("model_config")
    requested_model_config = asdict(model_config)
    if checkpoint_model_config == requested_model_config:
        if checkpoint.get("vocab") and checkpoint["vocab"] != tokenizer.tokens:
            raise ValueError("checkpoint vocab does not match ArithmeticTokenizer")
        model.load_state_dict(checkpoint["model_state_dict"])
        return int(checkpoint.get("step", 0))

    if not allow_vocab_expansion:
        raise ValueError(
            f"checkpoint model config does not match requested model config: {checkpoint_model_config}"
        )

    load_vocab_expanded_checkpoint(checkpoint, model, requested_model_config, tokenizer)
    return int(checkpoint.get("step", 0))


def load_vocab_expanded_checkpoint(
    checkpoint: dict[str, Any],
    model: AdditionTransformer,
    requested_model_config: dict[str, Any],
    tokenizer: ArithmeticTokenizer,
) -> None:
    checkpoint_model_config = checkpoint.get("model_config")
    if not isinstance(checkpoint_model_config, dict):
        raise ValueError("checkpoint missing model_config")

    mismatched = {
        key: (checkpoint_model_config.get(key), requested_value)
        for key, requested_value in requested_model_config.items()
        if key != "vocab_size" and checkpoint_model_config.get(key) != requested_value
    }
    if mismatched:
        raise ValueError(
            "checkpoint model config cannot be vocab-expanded because non-vocab fields differ: "
            f"{mismatched}"
        )

    checkpoint_vocab = checkpoint.get("vocab")
    if not isinstance(checkpoint_vocab, list) or not checkpoint_vocab:
        raise ValueError("checkpoint vocab is required for vocab expansion")

    old_token_to_id = {str(token): index for index, token in enumerate(checkpoint_vocab)}
    missing_tokens = [token for token in old_token_to_id if token not in tokenizer.token_to_id]
    if missing_tokens:
        raise ValueError(f"checkpoint vocab tokens are missing from target tokenizer: {missing_tokens}")

    source_state = checkpoint["model_state_dict"]
    target_state = model.state_dict()
    row_remapped_parameters = {"token_embedding.weight", "lm_head.weight"}

    for name, target_tensor in list(target_state.items()):
        source_tensor = source_state.get(name)
        if source_tensor is None:
            raise ValueError(f"checkpoint is missing parameter: {name}")

        if name in row_remapped_parameters:
            if source_tensor.ndim != 2 or target_tensor.ndim != 2:
                raise ValueError(f"{name} must be a rank-2 vocab parameter")
            if source_tensor.size(1) != target_tensor.size(1):
                raise ValueError(
                    f"{name} hidden dimension mismatch: checkpoint={tuple(source_tensor.shape)} "
                    f"target={tuple(target_tensor.shape)}"
                )

            remapped = target_tensor.clone()
            for token, old_id in old_token_to_id.items():
                remapped[tokenizer.token_to_id[token]] = source_tensor[old_id]
            target_state[name] = remapped
            continue

        if source_tensor.shape != target_tensor.shape:
            raise ValueError(
                f"checkpoint parameter shape mismatch for {name}: "
                f"checkpoint={tuple(source_tensor.shape)} target={tuple(target_tensor.shape)}"
            )
        target_state[name] = source_tensor

    model.load_state_dict(target_state)


def main() -> None:
    args = parse_args()
    config = build_config(args)
    validate_context_length(config.digit_length, config.context_length)
    if not 0.0 <= config.hard_case_ratio <= 1.0:
        raise ValueError("hard_case_ratio must be between 0 and 1")
    if config.loss_prompt_weight < 0.0:
        raise ValueError("loss_prompt_weight must be non-negative")

    torch.manual_seed(config.seed)
    random.seed(config.seed)

    run_dir = args.run_dir or default_run_dir(config.stage)
    run_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = ArithmeticTokenizer()
    device = select_device(config.device)
    hard_cases: list[HardCase] | None = None
    if config.hard_case_file:
        hard_cases = load_hard_cases(Path(config.hard_case_file))
        print(f"hard_cases={len(hard_cases)} hard_case_ratio={config.hard_case_ratio}")
    model_config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=config.d_model,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        ffn_hidden=config.ffn_hidden,
        context_length=config.context_length,
    )
    model = AdditionTransformer(model_config).to(device)
    init_step = 0
    if config.init_checkpoint:
        init_step = load_initial_checkpoint(
            Path(config.init_checkpoint),
            model,
            model_config,
            tokenizer,
            device,
            allow_vocab_expansion=config.allow_vocab_expansion,
        )
        print(
            f"init_checkpoint={config.init_checkpoint} init_step={init_step} "
            f"allow_vocab_expansion={config.allow_vocab_expansion}"
        )
    train_config = TrainConfig(
        batch_size=config.batch_size,
        steps=config.steps,
        eval_interval=config.eval_interval,
        val_batches=config.val_batches,
        accuracy_examples=config.accuracy_examples,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        grad_clip=config.grad_clip,
        digit_length=config.digit_length,
        context_length=config.context_length,
        seed=config.seed,
        number_format=config.number_format,
        loss_prompt_weight=config.loss_prompt_weight,
        operations=config.operations,
        operation_weights=config.operation_weights,
        operand_sampling=config.operand_sampling,
    )

    config_payload = {
        "experiment": asdict(config),
        "model": asdict(model_config),
        "parameter_count": count_parameters(model),
        "device": str(device),
        "vocab": tokenizer.tokens,
        "init_step": init_step,
    }
    write_json(run_dir / "config.json", config_payload)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )
    train_rng = random.Random(config.seed)
    eval_rng = random.Random(config.seed + 1)
    generation_rng = random.Random(config.seed + 2)

    history: list[dict[str, float]] = []
    checkpoint_step_paths: list[str] = []
    best_val_loss = float("inf")
    best_checkpoint_path = run_dir / "checkpoint_best_val.pt"
    final_checkpoint_path = run_dir / "checkpoint_final.pt"
    start = time.perf_counter()

    for step in range(1, train_config.steps + 1):
        inputs, targets = make_batch(
            tokenizer=tokenizer,
            batch_size=train_config.batch_size,
            context_length=train_config.context_length,
            digit_length=train_config.digit_length,
            device=device,
            rng=train_rng,
            hard_cases=hard_cases,
            hard_case_ratio=config.hard_case_ratio,
            number_format=config.number_format,
            operations=config.operations,
            operation_weights=config.operation_weights,
            operand_sampling=config.operand_sampling,
        )
        model.train()
        logits = model(inputs)
        loss_weights = token_loss_weights(
            inputs,
            targets,
            tokenizer.token_to_id["="],
            tokenizer.pad_id,
            config.loss_prompt_weight,
        )
        loss = next_token_loss(logits, targets, tokenizer.pad_id, loss_weights)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
        optimizer.step()

        if step == 1 or step % train_config.eval_interval == 0 or step == train_config.steps:
            val_loss = evaluate_loss(
                model=model,
                tokenizer=tokenizer,
                device=device,
                batch_size=train_config.batch_size,
                context_length=train_config.context_length,
                digit_length=train_config.digit_length,
                batches=train_config.val_batches,
                rng=eval_rng,
                number_format=train_config.number_format,
                loss_prompt_weight=train_config.loss_prompt_weight,
                operations=train_config.operations,
                operation_weights=train_config.operation_weights,
                operand_sampling=train_config.operand_sampling,
            )
            accuracy = exact_answer_accuracy(
                model=model,
                tokenizer=tokenizer,
                device=device,
                context_length=train_config.context_length,
                digit_length=train_config.digit_length,
                examples=train_config.accuracy_examples,
                rng=eval_rng,
                number_format=train_config.number_format,
                operations=train_config.operations,
                operation_weights=train_config.operation_weights,
                operand_sampling=train_config.operand_sampling,
            )
            row = {
                "step": float(step),
                "train_loss": float(loss.item()),
                "val_loss": float(val_loss),
                "exact_accuracy": float(accuracy),
                "elapsed_s": float(time.perf_counter() - start),
            }
            history.append(row)
            step_checkpoint_path = run_dir / f"checkpoint_step_{step:04d}.pt"
            save_checkpoint(
                step_checkpoint_path,
                model,
                optimizer,
                config,
                model_config,
                tokenizer,
                step,
                row,
            )
            checkpoint_step_paths.append(str(step_checkpoint_path))
            if row["val_loss"] < best_val_loss:
                best_val_loss = row["val_loss"]
                save_checkpoint(
                    best_checkpoint_path,
                    model,
                    optimizer,
                    config,
                    model_config,
                    tokenizer,
                    step,
                    row,
                )
            print(
                f"step={step:04d} "
                f"train_loss={row['train_loss']:.4f} "
                f"val_loss={row['val_loss']:.4f} "
                f"exact_accuracy={row['exact_accuracy']:.3f}"
            )

    sample_outputs: list[str] = []
    for _ in range(config.samples):
        a, operation, b = random_arithmetic_problem(
            config.digit_length,
            generation_rng,
            config.operations,
            config.operation_weights,
            config.operand_sampling,
        )
        prompt = format_prompt(a, b, config.digit_length, config.number_format, operation)
        sample_outputs.append(generate_text(model, tokenizer, prompt, config.context_length, device))

    write_metrics(run_dir / "metrics.csv", history)
    write_samples(run_dir / "samples.txt", sample_outputs)

    final = history[-1]
    save_checkpoint(
        final_checkpoint_path,
        model,
        optimizer,
        config,
        model_config,
        tokenizer,
        int(final["step"]),
        final,
    )
    summary = {
        "run_dir": str(run_dir),
        "metrics_path": str(run_dir / "metrics.csv"),
        "samples_path": str(run_dir / "samples.txt"),
        "config_path": str(run_dir / "config.json"),
        "checkpoint_best_val_path": str(best_checkpoint_path),
        "checkpoint_final_path": str(final_checkpoint_path),
        "checkpoint_step_paths": checkpoint_step_paths,
        "parameter_count": count_parameters(model),
        "device": str(device),
        "init_step": init_step,
        "best_val_loss": best_val_loss,
        "final_train_loss": final["train_loss"],
        "final_val_loss": final["val_loss"],
        "final_exact_accuracy": final["exact_accuracy"],
        "elapsed_s": final["elapsed_s"],
    }
    write_json(run_dir / "summary.json", summary)

    print(f"run_dir={run_dir}")
    print(f"metrics={run_dir / 'metrics.csv'}")
    print(f"samples={run_dir / 'samples.txt'}")
    print(f"checkpoint_best_val={best_checkpoint_path}")
    print(f"checkpoint_final={final_checkpoint_path}")


if __name__ == "__main__":
    main()

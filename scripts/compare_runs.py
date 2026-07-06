from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


SUMMARY_FIELDS = [
    "run_dir",
    "stage",
    "steps",
    "params",
    "d_model",
    "num_heads",
    "num_layers",
    "ffn_hidden",
    "context_length",
    "digit_length",
    "final_val_loss",
    "best_val_loss",
    "final_exact_accuracy",
    "best_exact_accuracy",
    "elapsed_s",
    "device",
]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_metrics(path: Path) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        return [{key: float(value) for key, value in row.items()} for row in reader]


def summarize_run(run_dir: Path) -> dict[str, Any]:
    config_path = run_dir / "config.json"
    metrics_path = run_dir / "metrics.csv"
    summary_path = run_dir / "summary.json"

    missing = [str(path) for path in (config_path, metrics_path, summary_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing run artifact(s): {', '.join(missing)}")

    config = load_json(config_path)
    summary = load_json(summary_path)
    metrics = load_metrics(metrics_path)
    if not metrics:
        raise ValueError(f"metrics file is empty: {metrics_path}")

    experiment = config.get("experiment", {})
    model = config.get("model", {})
    final = metrics[-1]

    return {
        "run_dir": str(run_dir),
        "stage": experiment.get("stage", run_dir.parent.name),
        "steps": int(max(row["step"] for row in metrics)),
        "params": int(summary.get("parameter_count", config.get("parameter_count", 0))),
        "d_model": int(model.get("d_model", experiment.get("d_model", 0))),
        "num_heads": int(model.get("num_heads", experiment.get("num_heads", 0))),
        "num_layers": int(model.get("num_layers", experiment.get("num_layers", 0))),
        "ffn_hidden": int(model.get("ffn_hidden", experiment.get("ffn_hidden", 0))),
        "context_length": int(model.get("context_length", experiment.get("context_length", 0))),
        "digit_length": int(experiment.get("digit_length", 0)),
        "final_val_loss": final["val_loss"],
        "best_val_loss": min(row["val_loss"] for row in metrics),
        "final_exact_accuracy": final["exact_accuracy"],
        "best_exact_accuracy": max(row["exact_accuracy"] for row in metrics),
        "elapsed_s": float(summary.get("elapsed_s", final["elapsed_s"])),
        "device": str(summary.get("device", config.get("device", ""))),
    }


def format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def to_markdown(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "No runs to compare."

    header = "| " + " | ".join(SUMMARY_FIELDS) + " |"
    separator = "| " + " | ".join("---" for _ in SUMMARY_FIELDS) + " |"
    body = [
        "| " + " | ".join(format_value(row[field]) for field in SUMMARY_FIELDS) + " |"
        for row in rows
    ]
    return "\n".join([header, separator, *body])


def to_csv(rows: list[dict[str, Any]]) -> str:
    lines = [",".join(SUMMARY_FIELDS)]
    for row in rows:
        lines.append(",".join(format_value(row[field]) for field in SUMMARY_FIELDS))
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare saved addition-transformer experiment runs.")
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--format", choices=["markdown", "csv"], default="markdown")
    parser.add_argument(
        "--sort",
        choices=["none", "final_val_loss", "best_val_loss", "best_exact_accuracy", "elapsed_s", "params"],
        default="final_val_loss",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = [summarize_run(run_dir) for run_dir in args.run_dirs]
    if args.sort != "none":
        reverse = args.sort == "best_exact_accuracy"
        rows.sort(key=lambda row: row[args.sort], reverse=reverse)

    if args.format == "csv":
        print(to_csv(rows))
    else:
        print(to_markdown(rows))


if __name__ == "__main__":
    main()

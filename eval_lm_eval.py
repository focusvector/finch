from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


DEFAULT_TASKS = ["hellaswag", "winogrande", "mmlu", "ifeval"]
PREFERRED_METRICS = {
    "hellaswag": ["acc_norm", "acc"],
    "winogrande": ["acc"],
    "mmlu": ["acc"],
    "ifeval": ["prompt_level_strict_acc", "inst_level_strict_acc"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare a base model and a FINCH QLoRA adapter with lm-eval.")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--adapter-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--tasks", type=str, default=None, help="Comma-separated lm-eval task names.")
    parser.add_argument("--batch-size", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--limit", type=str, default=None, help="Optional lm-eval --limit for smoke tests.")
    parser.add_argument("--num-fewshot", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_config(path: str | None) -> dict[str, Any]:
    if path is None:
        return {}
    config_path = Path(path)
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def get_nested(cfg: dict[str, Any], keys: tuple[str, ...], default: Any) -> Any:
    current: Any = cfg
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def choose(value: Any, fallback: Any) -> Any:
    return fallback if value is None else value


def parse_tasks(value: str | list[str] | tuple[str, ...] | None) -> list[str]:
    if value is None:
        return DEFAULT_TASKS
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value).split(",") if item.strip()]


def cli_bool(value: Any) -> str:
    return "True" if bool(value) else "False"


def build_model_args(model_name: str, adapter_dir: str | None, cfg: dict[str, Any]) -> str:
    eval_model_args = get_nested(cfg, ("eval", "model_args"), {}) or {}
    parts = [f"pretrained={model_name}"]
    if adapter_dir:
        parts.append(f"peft={adapter_dir}")
    if "dtype" in eval_model_args:
        parts.append(f"dtype={eval_model_args['dtype']}")
    if "load_in_4bit" in eval_model_args:
        parts.append(f"load_in_4bit={cli_bool(eval_model_args['load_in_4bit'])}")
    if "trust_remote_code" in eval_model_args:
        parts.append(f"trust_remote_code={cli_bool(eval_model_args['trust_remote_code'])}")
    if "parallelize" in eval_model_args:
        parts.append(f"parallelize={cli_bool(eval_model_args['parallelize'])}")
    if "device_map" in eval_model_args:
        parts.append(f"device_map={eval_model_args['device_map']}")
    if "max_memory_per_gpu" in eval_model_args:
        parts.append(f"max_memory_per_gpu={eval_model_args['max_memory_per_gpu']}")
    return ",".join(parts)


def base_command(cli_style: str) -> list[str]:
    if cli_style == "run":
        return [sys.executable, "-m", "lm_eval", "run"]
    return [sys.executable, "-m", "lm_eval"]


def build_eval_command(
    label: str,
    model_name: str,
    adapter_dir: str | None,
    tasks: list[str],
    output_dir: Path,
    cfg: dict[str, Any],
    batch_size: str,
    device: str,
    limit: str | None,
    num_fewshot: int | None,
    cli_style: str,
) -> list[str]:
    command = base_command(cli_style)
    command.extend(
        [
            "--model",
            "hf",
            "--model_args",
            build_model_args(model_name, adapter_dir, cfg),
            "--tasks",
            ",".join(tasks),
            "--device",
            device,
            "--batch_size",
            batch_size,
            "--output_path",
            str(output_dir / label),
        ]
    )
    if limit is not None:
        command.extend(["--limit", str(limit)])
    if num_fewshot is not None:
        command.extend(["--num_fewshot", str(num_fewshot)])
    return command


def command_line_shape_error(stderr: str) -> bool:
    lowered = stderr.lower()
    return "unrecognized arguments" in lowered or "invalid choice" in lowered or "usage:" in lowered


def run_eval_command(command: list[str], output_dir: Path, dry_run: bool) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "command": command,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    if dry_run:
        metadata["dry_run"] = True
        return metadata
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    (output_dir / "stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (output_dir / "stderr.txt").write_text(completed.stderr, encoding="utf-8")
    metadata.update(
        {
            "returncode": completed.returncode,
            "finished_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    if completed.returncode != 0:
        metadata["stderr_tail"] = completed.stderr[-4000:]
    return metadata


def find_results_json(output_dir: Path) -> Path | None:
    candidates = []
    for path in output_dir.rglob("*.json"):
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict) and "results" in data:
            candidates.append(path)
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def metric_value(task_metrics: dict[str, Any], metric_name: str) -> float | None:
    for key, value in task_metrics.items():
        if key == metric_name or key.startswith(f"{metric_name},"):
            if isinstance(value, (int, float)):
                return float(value)
    return None


def average_subtask_metric(results: dict[str, Any], task: str, metric_name: str) -> float | None:
    values = []
    for name, metrics in results.items():
        if not name.startswith(f"{task}_") or not isinstance(metrics, dict):
            continue
        value = metric_value(metrics, metric_name)
        if value is not None:
            values.append(value)
    if not values:
        return None
    return sum(values) / len(values)


def extract_metrics(results_payload: dict[str, Any], tasks: list[str]) -> dict[str, dict[str, float]]:
    results = results_payload.get("results", {})
    extracted: dict[str, dict[str, float]] = {}
    if not isinstance(results, dict):
        return extracted
    for task in tasks:
        extracted[task] = {}
        preferred = PREFERRED_METRICS.get(task, ["acc"])
        task_metrics = results.get(task)
        for metric_name in preferred:
            value = None
            if isinstance(task_metrics, dict):
                value = metric_value(task_metrics, metric_name)
            if value is None:
                value = average_subtask_metric(results, task, metric_name)
            if value is not None:
                extracted[task][metric_name] = value
    return extracted


def load_metrics(output_dir: Path, tasks: list[str]) -> tuple[dict[str, dict[str, float]], str | None]:
    results_path = find_results_json(output_dir)
    if results_path is None:
        return {}, None
    with results_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return extract_metrics(payload, tasks), str(results_path)


def compare_metrics(base: dict[str, dict[str, float]], finch: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    comparison: dict[str, dict[str, float]] = {}
    for task, base_metrics in base.items():
        comparison[task] = {}
        finch_metrics = finch.get(task, {})
        for metric_name, base_value in base_metrics.items():
            if metric_name in finch_metrics:
                comparison[task][f"{metric_name}_delta"] = finch_metrics[metric_name] - base_value
    return comparison


def main() -> None:
    args = parse_args()
    cfg = read_config(args.config)
    model_name = choose(args.model_name, get_nested(cfg, ("model", "name"), None))
    adapter_dir = choose(args.adapter_dir, get_nested(cfg, ("training", "save_dir"), None))
    output_dir = Path(str(choose(args.output_dir, get_nested(cfg, ("eval", "output_dir"), "eval_results"))))
    tasks = parse_tasks(choose(args.tasks, get_nested(cfg, ("eval", "tasks"), DEFAULT_TASKS)))
    batch_size = str(choose(args.batch_size, get_nested(cfg, ("eval", "batch_size"), "auto")))
    device = str(choose(args.device, get_nested(cfg, ("eval", "device"), "cuda:0")))
    limit = choose(args.limit, get_nested(cfg, ("eval", "limit"), None))
    num_fewshot = choose(args.num_fewshot, get_nested(cfg, ("eval", "num_fewshot"), None))
    cli_style = str(get_nested(cfg, ("eval", "cli_style"), "auto"))

    if model_name is None:
        raise ValueError("Provide --model-name or set model.name in the config.")
    if adapter_dir is None:
        raise ValueError("Provide --adapter-dir or set training.save_dir in the config.")
    if importlib.util.find_spec("lm_eval") is None:
        raise RuntimeError("lm-eval is not installed. Install it with `pip install lm_eval[hf]`.")

    output_dir.mkdir(parents=True, exist_ok=True)
    styles_to_try = ["legacy", "run"] if cli_style == "auto" else [cli_style]
    run_metadata: dict[str, Any] = {"tasks": tasks, "runs": {}}
    for label, adapter in [("base", None), ("finch_qlora", str(adapter_dir))]:
        last_metadata: dict[str, Any] | None = None
        for style in styles_to_try:
            command = build_eval_command(
                label=label,
                model_name=str(model_name),
                adapter_dir=adapter,
                tasks=tasks,
                output_dir=output_dir,
                cfg=cfg,
                batch_size=batch_size,
                device=device,
                limit=limit,
                num_fewshot=int(num_fewshot) if num_fewshot is not None else None,
                cli_style=style,
            )
            metadata = run_eval_command(command, output_dir / label, args.dry_run)
            metadata["cli_style"] = style
            last_metadata = metadata
            if args.dry_run or metadata.get("returncode") == 0:
                break
            if style == styles_to_try[-1] or not command_line_shape_error(str(metadata.get("stderr_tail", ""))):
                break
        run_metadata["runs"][label] = last_metadata
        if not args.dry_run and last_metadata and last_metadata.get("returncode") != 0:
            raise RuntimeError(f"lm-eval failed for {label}. See {output_dir / label / 'stderr.txt'}.")

    base_metrics, base_results_path = load_metrics(output_dir / "base", tasks)
    finch_metrics, finch_results_path = load_metrics(output_dir / "finch_qlora", tasks)
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_name": model_name,
        "adapter_dir": adapter_dir,
        "tasks": tasks,
        "base_results_path": base_results_path,
        "finch_results_path": finch_results_path,
        "base": base_metrics,
        "finch_qlora": finch_metrics,
        "delta": compare_metrics(base_metrics, finch_metrics),
        "metadata": run_metadata,
    }
    (output_dir / "comparison.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

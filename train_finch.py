from __future__ import annotations

import argparse
import json
import math
import os
import random
from functools import partial
from pathlib import Path
from typing import Any

import torch
import yaml
from datasets import Dataset, load_dataset, load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--train-path", type=str, default=None)
    parser.add_argument("--dataset-name", type=str, default=None)
    parser.add_argument("--dataset-config", type=str, default=None)
    parser.add_argument("--dataset-split", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--eta-max", type=float, default=None)
    parser.add_argument("--ema-alpha", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--max-grad-norm", type=float, default=None)
    parser.add_argument("--torch-dtype", type=str, default=None)
    parser.add_argument("--lora-r", type=int, default=None)
    parser.add_argument("--lora-alpha", type=int, default=None)
    parser.add_argument("--lora-dropout", type=float, default=None)
    parser.add_argument("--lora-target-modules", type=str, default=None)
    parser.add_argument("--bnb-4bit-quant-type", type=str, default=None)
    parser.add_argument("--bnb-4bit-use-double-quant", action="store_true")
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--save-final", action="store_true", default=None)
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


def torch_dtype_from_name(name: str) -> torch.dtype:
    if name == "auto":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def str_list(value: str | list[str] | tuple[str, ...] | None) -> list[str] | str:
    if value is None:
        return "all-linear"
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    value = str(value).strip()
    if value.lower() in {"", "auto", "all-linear", "all_linear"}:
        return "all-linear"
    return [item.strip() for item in value.split(",") if item.strip()]


def build_qlora_model(
    model_name: str,
    dtype: torch.dtype,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_target_modules: list[str] | str,
    quant_type: str,
    use_double_quant: bool,
    gradient_checkpointing: bool,
) -> torch.nn.Module:
    try:
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    except ImportError as exc:
        raise ImportError("QLoRA training requires peft. Install it with `pip install peft`.") from exc

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=quant_type,
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=use_double_quant,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map="auto",
        dtype=dtype,
    )
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=gradient_checkpointing,
    )
    if gradient_checkpointing:
        model.config.use_cache = False
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=lora_target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def build_prompt(tokenizer: AutoTokenizer, system: str, question: str, enable_thinking: bool) -> str:
    messages = [
        {"role": "system", "content": str(system or "")},
        {"role": "user", "content": str(question or "")},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except ValueError:
            pass
    parts = []
    if str(system or "").strip():
        parts.append(f"System: {str(system).strip()}")
    parts.append(f"User: {str(question or '').strip()}")
    parts.append("Assistant:")
    return "\n\n".join(parts)


def collate_sft(batch: list[dict[str, Any]], tokenizer: AutoTokenizer, max_length: int, enable_thinking: bool) -> dict[str, torch.Tensor]:
    prompts = []
    full_texts = []
    for item in batch:
        prompt = build_prompt(tokenizer, item.get("system", ""), item["prompt"], enable_thinking)
        answer = str(item["answer"])
        text = prompt + answer
        if tokenizer.eos_token:
            text += tokenizer.eos_token
        prompts.append(prompt)
        full_texts.append(text)
    model_inputs = tokenizer(
        full_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )
    labels = model_inputs["input_ids"].clone()
    labels[model_inputs["attention_mask"] == 0] = -100
    for i, prompt in enumerate(prompts):
        prompt_len = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        seq_len = int(model_inputs["attention_mask"][i].sum().item())
        labels[i, : min(prompt_len, seq_len)] = -100
    model_inputs["labels"] = labels
    return model_inputs


def normalize_sft_example(example: dict[str, Any]) -> dict[str, str]:
    if "prompt" in example and "answer" in example:
        return {
            "system": str(example.get("system", "") or ""),
            "prompt": str(example.get("prompt", "") or ""),
            "answer": str(example.get("answer", "") or ""),
        }
    if "question" in example and "answer" in example:
        return {
            "system": str(example.get("system", "") or ""),
            "prompt": str(example.get("question", "") or ""),
            "answer": str(example.get("answer", "") or ""),
        }
    if "instruction" in example and "output" in example:
        prompt = str(example.get("instruction", "") or "").strip()
        input_text = str(example.get("input", "") or "").strip()
        if input_text:
            prompt = f"{prompt}\n\nInput: {input_text}"
        return {
            "system": str(example.get("system", "") or ""),
            "prompt": prompt,
            "answer": str(example.get("output", "") or ""),
        }
    if "messages" in example and isinstance(example["messages"], list):
        system = ""
        prompt_turns: list[str] = []
        answer = ""
        for message in example["messages"]:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role") or message.get("from") or "").lower()
            content = str(message.get("content") or message.get("value") or message.get("text") or "")
            if role == "system" and not system:
                system = content
            elif role in {"user", "human"}:
                prompt_turns.append(content)
            elif role in {"assistant", "gpt"}:
                answer = content
        if prompt_turns and answer:
            return {"system": system, "prompt": "\n\n".join(prompt_turns), "answer": answer}
    raise ValueError(
        "Unsupported dataset schema. Expected prompt/answer, question/answer, instruction/output, or messages."
    )


def maybe_limit_dataset(dataset: Dataset, max_samples: int | None, seed: int) -> Dataset:
    if max_samples is None or max_samples <= 0 or max_samples >= len(dataset):
        return dataset
    return dataset.shuffle(seed=seed).select(range(max_samples))


def load_training_dataset(
    train_path: str | None,
    dataset_name: str | None,
    dataset_config: str | None,
    dataset_split: str,
    max_samples: int | None,
    seed: int,
) -> tuple[Dataset, dict[str, Any]]:
    if train_path:
        path = Path(train_path)
        if not path.exists():
            if dataset_name is None:
                raise FileNotFoundError(
                    f"Directory {train_path} not found. Set dataset.name to load from Hugging Face, "
                    "or set dataset.train_path to an existing saved dataset."
                )
        else:
            raw_dataset = load_from_disk(str(path))
            if not isinstance(raw_dataset, Dataset):
                raise ValueError("Saved training data must be a single Dataset, not a DatasetDict.")
            raw_dataset = maybe_limit_dataset(raw_dataset, max_samples, seed)
            mapped_dataset = raw_dataset.map(normalize_sft_example, remove_columns=raw_dataset.column_names)
            return mapped_dataset, {
                "source": "disk",
                "train_path": str(path),
                "examples": len(mapped_dataset),
                "max_samples": max_samples,
            }

    if dataset_name is None:
        raise ValueError("Provide dataset.train_path or dataset.name in the config.")

    if dataset_config:
        raw_dataset = load_dataset(dataset_name, dataset_config, split=dataset_split)
    else:
        raw_dataset = load_dataset(dataset_name, split=dataset_split)
    if not isinstance(raw_dataset, Dataset):
        raise ValueError("Expected a single dataset split for training.")
    raw_dataset = maybe_limit_dataset(raw_dataset, max_samples, seed)
    mapped_dataset = raw_dataset.map(normalize_sft_example, remove_columns=raw_dataset.column_names)
    return mapped_dataset, {
        "source": "huggingface",
        "dataset_name": dataset_name,
        "dataset_config": dataset_config,
        "dataset_split": dataset_split,
        "examples": len(mapped_dataset),
        "max_samples": max_samples,
    }


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def inverse_lr(base_lr: float, loss_value: float, eta_max: float) -> float:
    return min(base_lr / math.sqrt(max(loss_value, 1.0e-12)), eta_max)


def compute_training_loss(model: torch.nn.Module, dataloader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            tokens = int((batch["labels"] != -100).sum().item())
            total_loss += float(outputs.loss.detach().float().item()) * tokens
            total_tokens += tokens
    model.train()
    if total_tokens == 0:
        raise ValueError("Training set has no target tokens.")
    return total_loss / total_tokens


def save_run_summary(path: Path, summary: dict[str, Any]) -> None:
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    args = parse_args()
    cfg = read_config(args.config)
    seed = int(choose(args.seed, get_nested(cfg, ("project", "seed"), 42)))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model_name = choose(args.model_name, get_nested(cfg, ("model", "name"), None))
    train_path = choose(args.train_path, get_nested(cfg, ("dataset", "train_path"), None))
    dataset_name = choose(args.dataset_name, get_nested(cfg, ("dataset", "name"), None))
    dataset_config = choose(args.dataset_config, get_nested(cfg, ("dataset", "config"), None))
    dataset_split = str(choose(args.dataset_split, get_nested(cfg, ("dataset", "split"), "train")))
    max_samples = choose(args.max_samples, get_nested(cfg, ("dataset", "max_samples"), None))
    if model_name is None:
        raise ValueError("Provide --model-name or set model.name in the config.")
    dtype_name = str(choose(args.torch_dtype, get_nested(cfg, ("model", "torch_dtype"), "bfloat16")))
    dtype = torch_dtype_from_name(dtype_name)
    output_dir = str(choose(args.output_dir, get_nested(cfg, ("training", "save_dir"), "tofu/checkpoints/simple_inv_lr_sft")))
    epochs = int(choose(args.epochs, get_nested(cfg, ("training", "epochs"), 1)))
    batch_size = int(choose(args.batch_size, get_nested(cfg, ("training", "batch_size"), 1)))
    max_length = int(choose(args.max_length, get_nested(cfg, ("training", "max_length"), 1024)))
    base_lr = float(choose(args.learning_rate, get_nested(cfg, ("training", "learning_rate"), 2.0e-5)))
    eta_max = float(choose(args.eta_max, get_nested(cfg, ("training", "eta_max"), 5.0e-5)))
    ema_alpha = float(choose(args.ema_alpha, get_nested(cfg, ("training", "ema_alpha"), 0.9)))
    weight_decay = float(choose(args.weight_decay, get_nested(cfg, ("training", "weight_decay"), 0.0)))
    max_grad_norm = float(choose(args.max_grad_norm, get_nested(cfg, ("training", "max_grad_norm"), 1.0)))
    save_final = bool(choose(args.save_final, get_nested(cfg, ("training", "save_final"), True)))
    lora_r = int(choose(args.lora_r, get_nested(cfg, ("qlora", "r"), 16)))
    lora_alpha = int(choose(args.lora_alpha, get_nested(cfg, ("qlora", "alpha"), 32)))
    lora_dropout = float(choose(args.lora_dropout, get_nested(cfg, ("qlora", "dropout"), 0.05)))
    lora_target_modules = str_list(choose(args.lora_target_modules, get_nested(cfg, ("qlora", "target_modules"), None)))
    quant_type = str(choose(args.bnb_4bit_quant_type, get_nested(cfg, ("qlora", "bnb_4bit_quant_type"), "nf4")))
    use_double_quant = bool(args.bnb_4bit_use_double_quant or get_nested(cfg, ("qlora", "bnb_4bit_use_double_quant"), True))
    gradient_checkpointing = not bool(args.no_gradient_checkpointing or get_nested(cfg, ("qlora", "disable_gradient_checkpointing"), False))
    enable_thinking = bool(args.enable_thinking or get_nested(cfg, ("prompts", "enable_thinking"), False))
    if not (0.0 < ema_alpha < 1.0):
        raise ValueError("--ema-alpha must be in (0, 1).")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_qlora_model(
        model_name=model_name,
        dtype=dtype,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_target_modules=lora_target_modules,
        quant_type=quant_type,
        use_double_quant=use_double_quant,
        gradient_checkpointing=gradient_checkpointing,
    )
    if not torch.cuda.is_available():
        model.to(device)
    model.train()
    dataset, dataset_metadata = load_training_dataset(
        train_path=train_path,
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        dataset_split=dataset_split,
        max_samples=int(max_samples) if max_samples is not None else None,
        seed=seed,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=partial(collate_sft, tokenizer=tokenizer, max_length=max_length, enable_thinking=enable_thinking),
    )
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable_parameters, lr=base_lr, weight_decay=weight_decay)
    set_lr(optimizer, base_lr)
    ema_loss = compute_training_loss(model, dataloader, device)
    print(f"initial_ema_loss={ema_loss:.6f}", flush=True)
    global_step = 0
    train_history: list[dict[str, float]] = []
    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        for batch in dataloader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            batch_loss = float(loss.detach().float().item())
            ema_loss = ema_alpha * ema_loss + (1.0 - ema_alpha) * batch_loss
            lr = inverse_lr(base_lr, ema_loss, eta_max)
            set_lr(optimizer, lr)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            train_history.append(
                {
                    "epoch": float(epoch),
                    "step": float(global_step),
                    "loss": batch_loss,
                    "ema_loss": ema_loss,
                    "lr": lr,
                }
            )
            print(f"epoch={epoch} step={global_step} loss={batch_loss:.6f} ema_loss={ema_loss:.6f} lr={lr:.8g}", flush=True)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    run_summary = {
        "method": "finch_qlora",
        "model_name": model_name,
        "dtype": dtype_name,
        "dataset": dataset_metadata,
        "training": {
            "epochs": epochs,
            "batch_size": batch_size,
            "max_length": max_length,
            "learning_rate": base_lr,
            "eta_max": eta_max,
            "ema_alpha": ema_alpha,
            "weight_decay": weight_decay,
            "max_grad_norm": max_grad_norm,
            "steps": global_step,
        },
        "qlora": {
            "r": lora_r,
            "alpha": lora_alpha,
            "dropout": lora_dropout,
            "target_modules": lora_target_modules,
            "bnb_4bit_quant_type": quant_type,
            "bnb_4bit_use_double_quant": use_double_quant,
            "gradient_checkpointing": gradient_checkpointing,
        },
        "final": {
            "train_loss": train_history[-1]["loss"] if train_history else None,
            "ema_loss": train_history[-1]["ema_loss"] if train_history else ema_loss,
            "lr": train_history[-1]["lr"] if train_history else base_lr,
        },
        "train_history": train_history,
    }
    save_run_summary(output_path / "run_summary.json", run_summary)
    if save_final:
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()

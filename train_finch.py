from __future__ import annotations

import argparse
import math
import os
import random
from functools import partial
from pathlib import Path
from typing import Any

import torch
import yaml
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--train-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--eta-max", type=float, default=5.0e-5)
    parser.add_argument("--ema-alpha", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--max-grad-norm", type=float, default=None)
    parser.add_argument("--torch-dtype", type=str, default=None)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--save-final", action="store_true")
    return parser.parse_args()


def read_config(path: str | None) -> dict[str, Any]:
    if path is None:
        return {}
    with open(path, "r", encoding="utf-8") as handle:
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
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


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
    if model_name is None or train_path is None:
        raise ValueError("Provide --model-name and --train-path, or a config with model.name and dataset.train_path.")
    dtype_name = str(choose(args.torch_dtype, get_nested(cfg, ("model", "torch_dtype"), "bfloat16")))
    dtype = torch_dtype_from_name(dtype_name)
    output_dir = str(choose(args.output_dir, get_nested(cfg, ("training", "save_dir"), "tofu/checkpoints/simple_inv_lr_sft")))
    epochs = int(choose(args.epochs, get_nested(cfg, ("training", "epochs"), 1)))
    batch_size = int(choose(args.batch_size, get_nested(cfg, ("training", "batch_size"), 1)))
    max_length = int(choose(args.max_length, get_nested(cfg, ("training", "max_length"), 1024)))
    base_lr = float(choose(args.learning_rate, get_nested(cfg, ("training", "learning_rate"), 2.0e-5)))
    weight_decay = float(choose(args.weight_decay, get_nested(cfg, ("training", "weight_decay"), 0.0)))
    max_grad_norm = float(choose(args.max_grad_norm, get_nested(cfg, ("training", "max_grad_norm"), 1.0)))
    enable_thinking = bool(args.enable_thinking or get_nested(cfg, ("prompts", "enable_thinking"), False))
    if not (0.0 < args.ema_alpha < 1.0):
        raise ValueError("--ema-alpha must be in (0, 1).")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    dataset = load_from_disk(train_path)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=partial(collate_sft, tokenizer=tokenizer, max_length=max_length, enable_thinking=enable_thinking),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
    set_lr(optimizer, base_lr)
    ema_loss = compute_training_loss(model, dataloader, device)
    print(f"initial_ema_loss={ema_loss:.6f}", flush=True)
    global_step = 0
    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        for batch in dataloader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            batch_loss = float(loss.detach().float().item())
            ema_loss = args.ema_alpha * ema_loss + (1.0 - args.ema_alpha) * batch_loss
            lr = inverse_lr(base_lr, ema_loss, args.eta_max)
            set_lr(optimizer, lr)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            print(f"epoch={epoch} step={global_step} loss={batch_loss:.6f} ema_loss={ema_loss:.6f} lr={lr:.8g}", flush=True)
    if args.save_final:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()


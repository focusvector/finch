Requirements:

- `torch`
- `transformers`
- `datasets`
- `pyyaml`
- `bitsandbytes`
- `peft`
- `lm_eval[hf]`

This implementation keeps the FINCH loss-adaptive learning-rate schedule intact, but trains a QLoRA
adapter instead of updating the full model. The base model is loaded in 4-bit NF4 quantization and
only LoRA parameters are optimized.

Run with a config:

```bash
python -m train_finch --config your_config.yaml
```

Or run with explicit arguments:

```bash
python -m train_finch \
  --model-name your_model \
  --train-path your_data \
  --epochs 1 \
  --batch-size 32 \
  --learning-rate 2e-5
```

For an 8GB GPU, start from the included `config.yaml`. By default it trains on
`locuslab/TOFU`, config `full`, split `train`. You can also point `dataset.train_path` at a saved
Hugging Face dataset.

Supported training schemas are:

- `prompt` and `answer`
- `question` and `answer`
- `instruction` and `output`
- chat `messages`

```bash
python train_finch.py --config config.yaml
```

The trained FINCH QLoRA adapter is saved to `training.save_dir`.

Compare the base model against the FINCH QLoRA adapter with lm-eval:

```bash
python eval_lm_eval.py --config config.yaml
```

By default this runs:

- `hellaswag`
- `winogrande`
- `mmlu`
- `ifeval`

Raw lm-eval outputs are stored under `eval.output_dir/base` and `eval.output_dir/finch_qlora`.
A compact summary with base metrics, FINCH QLoRA metrics, and deltas is written to
`eval.output_dir/comparison.json`.

Use `eval.limit` or `--limit` for a quick smoke test before launching full evaluations:

```bash
python eval_lm_eval.py --config config.yaml --limit 20
```

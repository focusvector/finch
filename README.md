Requirements:

- `torch`
- `transformers`
- `datasets`
- `pyyaml`

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


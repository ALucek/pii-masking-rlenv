# PII Masking RL Environment

A PII masking RL environment made with [verifiers.](https://github.com/PrimeIntellect-ai/verifiers)

See this environment on the Prime Intellect Environments Hub Here: https://app.primeintellect.ai/dashboard/environments/adamlucek/pii-masking

### Overview
- **Environment ID**: `pii-masking`
- **Short description**: Evaluates models' ability to identify and mask personally identifiable information (PII) in text by replacing it with `[PII]` tags.
- **Tags**: `pii`, `privacy`, `redaction`, `single-turn`

### Datasets
- **Primary dataset**: [AdamLucek/open-pii-masking-en-us-30k](https://huggingface.co/datasets/AdamLucek/open-pii-masking-en-us-30k)
- **Source links**: A filtered and transformed subset of [ai4privacy/open-pii-masking-500k-ai4privacy](https://huggingface.co/datasets/ai4privacy/open-pii-masking-500k-ai4privacy) filtered for only US English examples where all unique PII labels have been removed and replaced with a [PII] mask.
- **Split sizes**: Default 80% train, 20% eval (configurable via `num_eval_examples`)

### Task
- **Type**: `single-turn`
- **Parser**: `XMLParser` with `masked_output` field
- **Rubric overview**: Exact match (100%), PII count (50%), format compliance (10%)

### Quickstart

Install the prime CLI and verifiers [by following this Quick Start guide.](https://github.com/PrimeIntellect-ai/verifiers?tab=readme-ov-file#quick-start)

Run an evaluation with default settings:
```bash
uv run vf-eval pii-masking
```

Configure model and sampling:
```bash
uv run vf-eval pii-masking \
  -m gpt-4o-mini \
  -n 20 -r 3 -t 1024 -T 0.7 \
  -a '{"num_train_examples": 1000, "num_eval_examples": 200}'  # env-specific args as JSON
```

**Notes**:
- Use `-a` / `--env-args` to pass environment-specific configuration as a JSON object.

### Environment Arguments

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `num_train_examples` | int | `-1` | Number of training examples to use. Use `-1` for all available examples. |
| `num_eval_examples` | int | `-1` | Number of evaluation examples. If `-1`, uses 20% of the training dataset. Otherwise, uses the specified count. |
| `random_seed` | int | `42` | Random seed for dataset splitting. Ensures reproducible train/eval splits. |

### Metrics

| Metric | Meaning |
| ------ | ------- |
| `reward` | Main scalar reward (weighted sum of exact match, PII count, format compliance) |
| `exact_match_reward` | Binary reward (1.0 if perfect match, 0.0 otherwise). Compares parsed masked output character-by-character with expected answer. |
| `pii_count_reward` | Binary reward (1.0 if correct, 0.0 otherwise). Checks if number of `[PII]` tags matches expected count. |
| `format_reward` | Parser-generated format reward. Ensures output is properly formatted with valid XML tags (`<masked_output>...</masked_output>`). |

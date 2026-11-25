# Independent validation of a stock-prediction model

I built a stock-prediction research pipeline and a live paper-trading system with
AI assistance (Claude Code). This repository is my independent re-implementation and
validation of it. I wrote all the code here myself, working only from a written
model specification. The original code stays private and is called the
*developer implementation* throughout.

The goal is to reproduce the developer implementation's results from the spec
alone, reconcile every difference, and challenge the model's conceptual soundness,
following the developer/validator split in US bank model-risk guidance (SR 26-2).

## Contents

| Path | What it is |
|---|---|
| [`docs/MODEL_SPEC.md`](docs/MODEL_SPEC.md) | Developer's model documentation: data, features, label, model, walk-forward design, metrics, costs, and which choices were arbitrary or chosen after seeing results |
| [`docs/PREREGISTRATION.md`](docs/PREREGISTRATION.md) | Append-only log of what was fixed before the data that will test it existed |
| `src/` | Validation harness |
| `tests/` | Tests for the harness, including leakage checks |

## Status

Work starts the week of 2026-11-30.

## Running the tests

```bash
pytest
```

Price data comes from yfinance and is downloaded locally into `data/`, which is
not committed.

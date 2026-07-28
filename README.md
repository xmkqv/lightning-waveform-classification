# Lightning waveform classification

This project compares compact neural architectures for classifying 1,000-sample
lightning waveforms. It trains each model with the same data split and metrics,
records single-waveform inference time, and exports the comparison as a static
HTML report.

The repository is an experimental research codebase. It is useful for comparing
model design choices, but it is not a production classifier or a published
benchmark.

## What is included

- Seven PyTorch Lightning models: MLP, FCN, bottleneck, DCT, DPPV, random
  projection, and wavelet variants
- Stratified train, validation, and test splits with training-only scaling and
  class weighting
- Repeated-seed experiment support with CSV logs
- A Jinja and Plotly report that compares metrics, timing, architecture, and
  training history

The generated report and experiment logs are not tracked. Rebuild them from your
own data so that the code and the reported evidence stay in sync.

## Architecture

```text
NumPy waveforms
    -> LightningDataModule
    -> shared BaseClassifier metrics and training loop
    -> model architecture selected through the CLI
    -> CSV experiment logs
    -> static HTML comparison report
```

`waveform_classification/data` owns loading and splitting.
`waveform_classification/nets` contains the model variants and their shared
training behavior. `waveform_classification/report` reads completed runs and
renders the report. The scripts in `scripts` configure training and multi-model
runs.

## Setup

Python 3.13 and [uv](https://docs.astral.sh/uv/) are required.

```bash
uv sync --locked --dev
```

The environment file is optional. `RESULTS_DIR` defaults to `results` and
`LOG_DIR` defaults to `.cache/logs`. Copy `.env.example` to `.env` if you use
direnv and want to override either path.

Place the dataset under `datasets/waveform/lightning`. The loader expects ten
NumPy arrays named `01.npy` through `09.npy`, plus `010.npy`. Each array must have
shape `(n_samples, 1000)`. Dataset files are intentionally excluded from Git.
See [DATA.md](DATA.md) for the data policy and provenance checklist.

## Run an experiment

Inspect the model-specific options before starting a run:

```bash
uv run python scripts/run.py --help
```

Run one model:

```bash
uv run python scripts/run.py mlp
```

Run the complete comparison and write `results/report.html`:

```bash
uv run python scripts/run-complete.py
```

The full comparison trains seven models across five seeds. Use
`uv run python scripts/run-complete.py --minimal` for a short pipeline check.

## Validate

```bash
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pytest
uv build
```

## Limits and status

- The source dataset is not distributed with this repository
- Performance depends on the supplied dataset and hardware
- The timing metric measures a batch of one on the selected accelerator
- No package release or stable API is promised at version 0.1.0

## Reuse

Released under the [MIT License](LICENSE). Dataset rights are separate and are
described in [DATA.md](DATA.md).

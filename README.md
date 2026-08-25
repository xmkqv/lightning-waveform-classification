# lightning waveform classification

Compares seven compact neural architectures for classifying 1,000-sample
lightning waveforms. Every model trains on one shared stratified split with the
same metrics, records single-waveform inference time, and exports the comparison
as a static HTML report.

[![CI](https://github.com/xmkqv/lightning-waveform-classification/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/xmkqv/lightning-waveform-classification/actions/workflows/ci.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![python 3.13+](https://img.shields.io/badge/python-3.13%2B-blue.svg)](pyproject.toml)

This is an experimental research codebase, useful for comparing model design
choices. It is not a production classifier or a published benchmark. No results
ship with it: the source dataset is not redistributable, so every number has to
be regenerated from your own data.

## quickstart

Python 3.13 and [uv](https://docs.astral.sh/uv/) are required.

```bash
uv sync --locked --dev
```

These two run on a fresh clone, with no dataset:

```bash
uv run pytest
uv run python scripts/run.py --help
```

Training needs data. Place ten NumPy arrays under `datasets/waveform/lightning`,
named `01.npy` through `09.npy` plus `010.npy`, one per class, each of shape
`(n_samples, 1000)`. Class order is defined by `Data.classnames` in
`waveform_classification/data/lightning.py`. Dataset files are excluded from
Git; see [DATA.md](DATA.md) for the data policy and provenance checklist.

Train one model:

```bash
uv run python scripts/run.py mlp
```

Run the full comparison and write `results/report.html`:

```bash
uv run python scripts/run-complete.py
```

The full comparison trains seven models across five seeds. Use
`uv run python scripts/run-complete.py --minimal` for a short pipeline check:
one seed, two epochs.

`RESULTS_DIR` defaults to `results` and `LOG_DIR` defaults to `.cache/logs`.
Copy `.env.example` to `.env` to override either path.

## what is included

- Seven PyTorch Lightning models: MLP, FCN, bottleneck, DCT, DPPV, random
  projection, and wavelet variants
- Stratified train, validation, and test splits with training-only scaling and
  class weighting
- Repeated-seed experiment support with CSV logs
- A Jinja and Plotly report that compares metrics, timing, architecture, and
  training history

The generated report and experiment logs are not tracked. Rebuild them from your
own data so that the code and the reported evidence stay in sync.

## architecture

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

## validate

The same chain runs in CI on every push:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pytest
uv build
```

## limits and status

- The source dataset is not distributed with this repository
- Performance depends on the supplied dataset and hardware
- The timing metric measures a batch of one on the selected accelerator
- No package release or stable API is promised at version 0.1.0

## license

Released under the [MIT License](LICENSE). Dataset rights are separate and are
described in [DATA.md](DATA.md).

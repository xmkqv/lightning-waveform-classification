"""Data extraction from training logs and model checkpoints."""

from __future__ import annotations

import re
import warnings
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import attrs
import numpy as np
import pandas as pd
import yaml


@attrs.frozen
class Stat:
    """Aggregated statistic with mean and standard deviation."""

    mean: float
    std: float
    n: int

    def __str__(self) -> str:
        if self.n == 1:
            return f"{self.mean:.3f}"
        return f"{self.mean:.3f}±{self.std:.3f}"


@attrs.frozen
class RunMetrics:
    """Metrics from a single training run."""

    model_name: str
    group: int
    seed: int
    metrics: Mapping[str, float]
    hparams: Mapping[str, Any]

    @property
    def params(self) -> int:
        """Get parameter count from hparams."""
        return int(self.hparams.get("n_weight", 0))


@attrs.frozen
class AggregatedRun:
    """Aggregated metrics across seeds in a group."""

    model_name: str
    group: int
    n_seeds: int
    metrics: Mapping[str, Stat]
    hparams: Mapping[str, Any]

    @property
    def params(self) -> int:
        """Get parameter count from hparams."""
        return int(self.hparams.get("n_weight", 0))


@attrs.frozen
class TrainingHistory:
    """Training metrics over epochs."""

    epochs: tuple[int, ...]
    train_loss: tuple[float, ...]
    val_loss: tuple[float, ...]
    train_acc: tuple[float, ...] = ()
    val_acc: tuple[float, ...] = ()


class MetricSource(Protocol):
    """Protocol for metric extraction sources."""

    def latest_run(self, model: str) -> AggregatedRun:
        """Get aggregated metrics from latest group of a model."""
        ...

    def all_runs(self, model: str) -> Sequence[RunMetrics]:
        """Get all individual runs for a model."""
        ...

    def training_history(self, model: str, group: int, seed: int) -> TrainingHistory:
        """Get training history for a specific run."""
        ...


class CSVSource:
    """Extract metrics from CSVLogger output files.

    Directory naming convention: g{group:03d}_s{seed}
    Example: mlp/g001_s42/ for model "mlp", group 1, seed 42
    """

    # Pattern: g001_s42 (group 1, seed 42)
    RUN_PATTERN = re.compile(r"^g(\d+)_s(\d+)$")

    def __init__(self, log_dir: Path) -> None:
        self.log_dir = log_dir
        self._cache: dict[str, pd.DataFrame] = {}

    def _get_runs(self, model: str) -> dict[int, list[tuple[int, str]]]:
        """Get all runs grouped by group ID.

        Returns: {group_id: [(seed, dir_name), ...]}
        """
        model_dir = self.log_dir / model
        if not model_dir.exists():
            return {}

        groups: dict[int, list[tuple[int, str]]] = defaultdict(list)
        for p in model_dir.iterdir():
            if not p.is_dir():
                continue
            match = self.RUN_PATTERN.match(p.name)
            if match:
                group_id = int(match.group(1))
                seed = int(match.group(2))
                groups[group_id].append((seed, p.name))

        return dict(groups)

    def _latest_group(self, model: str) -> int:
        """Get the latest (highest) group ID for a model."""
        groups = self._get_runs(model)
        if not groups:
            raise ValueError(f"No runs found for model: {model}")
        return max(groups.keys())

    def _load_metrics_df(self, model: str, run_dir: str) -> pd.DataFrame:
        """Load or retrieve cached metrics DataFrame."""
        key = f"{model}/{run_dir}"
        if key not in self._cache:
            path = self.log_dir / model / run_dir / "metrics.csv"
            if not path.exists():
                raise FileNotFoundError(f"Metrics file not found: {path}")
            self._cache[key] = pd.read_csv(path)
        return self._cache[key]

    def _load_hparams(self, model: str, run_dir: str) -> dict[str, Any]:
        """Load hyperparameters from hparams.yaml."""
        path = self.log_dir / model / run_dir / "hparams.yaml"
        if path.exists():
            with path.open() as f:
                return yaml.full_load(f) or {}
        return {}

    def _extract_scalars(self, df: pd.DataFrame) -> dict[str, float]:
        """Extract final scalar values from metrics DataFrame."""
        metrics: dict[str, float] = {}
        for col in df.columns:
            if col in ("step", "epoch"):
                continue
            valid = df[col].dropna()
            if len(valid) > 0:
                metrics[col] = float(valid.iloc[-1])
        return metrics

    def _extract_history(self, df: pd.DataFrame, tag: str) -> tuple[tuple[int, ...], tuple[float, ...]]:
        """Extract full history for a metric."""
        if tag not in df.columns:
            return (), ()

        mask = df[tag].notna()
        if not bool(mask.to_numpy().any()):
            return (), ()

        subset = df[mask]
        steps = tuple(int(s) for s in subset["step"].tolist())
        values = tuple(float(v) for v in subset[tag].tolist())
        return steps, values

    def _aggregate_metrics(self, runs: list[RunMetrics]) -> Mapping[str, Stat]:
        """Aggregate metrics across runs."""
        # Collect all metric keys
        all_keys: set[str] = set()
        for run in runs:
            all_keys.update(run.metrics.keys())

        # Aggregate each metric
        aggregated: dict[str, Stat] = {}
        for key in all_keys:
            values = [r.metrics[key] for r in runs if key in r.metrics]
            if values:
                aggregated[key] = Stat(
                    mean=float(np.mean(values)),
                    std=float(np.std(values)),
                    n=len(values),
                )
        return aggregated

    def get_run(self, model: str, group: int, seed: int) -> RunMetrics:
        """Get metrics from a specific run."""
        run_dir = f"g{group:03d}_s{seed}"
        df = self._load_metrics_df(model, run_dir)
        metrics = self._extract_scalars(df)
        hparams = self._load_hparams(model, run_dir)

        return RunMetrics(
            model_name=model,
            group=group,
            seed=seed,
            metrics=metrics,
            hparams=hparams,
        )

    def group_runs(self, model: str, group: int) -> list[RunMetrics]:
        """Get all runs in a group."""
        groups = self._get_runs(model)
        if group not in groups:
            raise ValueError(f"Group {group} not found for model: {model}")

        runs: list[RunMetrics] = []
        for seed, run_dir in groups[group]:
            df = self._load_metrics_df(model, run_dir)
            metrics = self._extract_scalars(df)
            hparams = self._load_hparams(model, run_dir)
            runs.append(
                RunMetrics(
                    model_name=model,
                    group=group,
                    seed=seed,
                    metrics=metrics,
                    hparams=hparams,
                )
            )
        return runs

    def latest_run(self, model: str) -> AggregatedRun:
        """Get aggregated metrics from latest group of a model."""
        group = self._latest_group(model)
        runs = self.group_runs(model, group)

        if not runs:
            raise ValueError(f"No runs in group {group} for model: {model}")

        # Warn if test metrics missing
        test_keys = {k for r in runs for k in r.metrics if k.startswith("test/")}
        if not test_keys:
            warnings.warn(
                f"No test metrics found for {model}/g{group:03d}_*. Did you run trainer.test()?",
                stacklevel=2,
            )

        return AggregatedRun(
            model_name=model,
            group=group,
            n_seeds=len(runs),
            metrics=self._aggregate_metrics(runs),
            hparams=runs[0].hparams,  # Same for all seeds
        )

    def all_runs(self, model: str) -> Sequence[RunMetrics]:
        """Get all individual runs for a model."""
        runs: list[RunMetrics] = []
        for group, seeds in self._get_runs(model).items():
            for seed, run_dir in seeds:
                df = self._load_metrics_df(model, run_dir)
                metrics = self._extract_scalars(df)
                hparams = self._load_hparams(model, run_dir)
                runs.append(
                    RunMetrics(
                        model_name=model,
                        group=group,
                        seed=seed,
                        metrics=metrics,
                        hparams=hparams,
                    )
                )
        return runs

    def training_history(self, model: str, group: int, seed: int) -> TrainingHistory:
        """Get training history for a specific run."""
        run_dir = f"g{group:03d}_s{seed}"
        df = self._load_metrics_df(model, run_dir)

        # Extract loss curves
        train_epochs, train_loss = self._extract_history(df, "train/loss_epoch")
        if not train_epochs:
            train_epochs, train_loss = self._extract_history(df, "train/loss")

        val_epochs, val_loss = self._extract_history(df, "val/loss")

        # Extract accuracy if available
        _, train_acc = self._extract_history(df, "train/acc_epoch")
        if not train_acc:
            _, train_acc = self._extract_history(df, "train/acc")
        _, val_acc = self._extract_history(df, "val/acc")

        # Align to shortest length
        lengths = [len(t) for t in (train_loss, val_loss) if t]
        if lengths:
            min_len = min(lengths)
            epochs = (train_epochs or val_epochs)[:min_len]
            train_loss = train_loss[:min_len] if train_loss else ()
            val_loss = val_loss[:min_len] if val_loss else ()
            train_acc = train_acc[:min_len] if train_acc else ()
            val_acc = val_acc[:min_len] if val_acc else ()
        else:
            epochs = ()

        return TrainingHistory(
            epochs=epochs,
            train_loss=train_loss,
            val_loss=val_loss,
            train_acc=train_acc,
            val_acc=val_acc,
        )


class MockSource:
    """Mock metric source for testing."""

    def latest_run(self, model: str) -> AggregatedRun:
        """Return mock aggregated metrics."""
        return AggregatedRun(
            model_name=model,
            group=1,
            n_seeds=3,
            metrics={
                "test/acc": Stat(0.85, 0.02, 3),
                "test/f1": Stat(0.82, 0.01, 3),
                "test/auroc": Stat(0.92, 0.01, 3),
            },
            hparams={"lr": 1e-3},
        )

    def all_runs(self, model: str) -> Sequence[RunMetrics]:
        """Return all mock runs."""
        return [RunMetrics(model, group=1, seed=s, metrics={}, hparams={}) for s in [1, 2, 3]]

    def training_history(self, model: str, group: int, seed: int) -> TrainingHistory:
        """Return mock training history."""
        epochs = tuple(range(1, 51))
        return TrainingHistory(
            epochs=epochs,
            train_loss=tuple(1.5 - i * 0.02 for i in range(50)),
            val_loss=tuple(1.6 - i * 0.018 for i in range(50)),
            train_acc=tuple(0.5 + i * 0.008 for i in range(50)),
            val_acc=tuple(0.48 + i * 0.007 for i in range(50)),
        )

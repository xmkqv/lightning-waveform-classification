"""Static HTML export for waveform classification reports.

Schema-driven approach:
- Field definitions in schema.py (single source of truth)
- Templates iterate over schema-provided data
- Adding a metric = one line in schema.py, no template changes
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import attrs
import numpy as np
from jinja2 import Environment, FileSystemLoader
from rich import print

from waveform_classification.data.lightning import Data
from waveform_classification.report import charts
from waveform_classification.report.diagrams import model_to_base64_png
from waveform_classification.report.extract import AggregatedRun, CSVSource, MetricSource, Stat
from waveform_classification.report.schema import (
    CLASSES,
    CONFIG,
    EVALUATION,
    HPARAMS,
    METRICS,
    SUMMARY,
)

if TYPE_CHECKING:
    import plotly.graph_objects as go

    from waveform_classification.report.extract import TrainingHistory

# Jinja environment with component loader
COMPONENTS = Path(__file__).parent / "components"
ENV = Environment(loader=FileSystemLoader(COMPONENTS), autoescape=False)


# =============================================================================
# Data Extraction
# =============================================================================


MODELS = ["mlp", "fcn", "bottleneck", "dct", "dppv", "rproj", "wavelet"]


def _get_runs(source: MetricSource | None) -> list[AggregatedRun]:
    """Get latest runs for all known models."""
    if source is None:
        return []
    runs: list[AggregatedRun] = []
    for model in MODELS:
        try:
            runs.append(source.latest_run(model))
        except (ValueError, FileNotFoundError):
            continue
    return runs


def _get_histories(source: CSVSource | None, runs: list[AggregatedRun]) -> dict[str, TrainingHistory]:
    """Get training histories for runs (uses first seed of each group)."""
    if source is None:
        return {}
    histories: dict[str, TrainingHistory] = {}
    for run in runs:
        try:
            # Get first seed in the group for history
            group_runs = source.group_runs(run.model_name, run.group)
            if group_runs:
                first_seed = group_runs[0].seed
                history = source.training_history(run.model_name, run.group, first_seed)
                if history.epochs:
                    histories[run.model_name] = history
        except (ValueError, FileNotFoundError):
            continue
    return histories


def _prettify_arch(arch: str) -> str:
    """Clean up architecture string for display."""
    return arch.replace("->", " → ").replace("xk", "×k")  # noqa: RUF001


def _fig_html(fig: go.Figure | None) -> str:
    """Convert Plotly figure to embeddable HTML."""
    if fig is None:
        return ""
    return fig.to_html(full_html=False, include_plotlyjs=False)


# =============================================================================
# Schema-Driven Transformation
# =============================================================================


@attrs.frozen
class StatItem:
    """A single stat card item."""

    label: str
    value: str
    desc: str


@attrs.frozen
class TableData:
    """Schema-driven table data."""

    headers: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]


@attrs.frozen
class ArchItem:
    """Architecture card item."""

    name: str
    params: str
    arch: str
    diagram: str = ""  # base64-encoded PNG


def _stat_val(metrics: dict[str, Stat], key: str, default: float = 0.0) -> Stat:
    """Get a Stat from metrics, or create a default single-value Stat."""
    if key in metrics:
        return metrics[key]
    return Stat(mean=default, std=0.0, n=1)


def _run_to_dict(run: AggregatedRun) -> dict[str, Any]:
    """Convert AggregatedRun to flat dict for schema processing."""
    metrics = dict(run.metrics)
    return {
        "name": run.model_name.upper(),
        "n_seeds": run.n_seeds,
        "acc": _stat_val(metrics, "test/acc"),
        "f1": _stat_val(metrics, "test/f1"),
        "auroc": _stat_val(metrics, "test/auroc"),
        "precision": _stat_val(metrics, "test/precision"),
        "recall": _stat_val(metrics, "test/recall"),
        "t_per_unit_batch": Stat(
            mean=_stat_val(metrics, "test/t_per_unit_batch").mean * 1e6,
            std=_stat_val(metrics, "test/t_per_unit_batch").std * 1e6,
            n=_stat_val(metrics, "test/t_per_unit_batch").n,
        ),
        "lr": run.hparams.get("lr", 0.001),
        "dropout": run.hparams.get("dropout", 0.0),
        "params": f"{p:,}" if (p := run.params) else "-",
        "arch": _prettify_arch(charts.format_arch(run.model_name, run.hparams)),
    }


def build_summary_stats(best: AggregatedRun | None) -> list[StatItem]:
    """Build summary stat cards from schema."""
    if best is None:
        return []

    metrics = dict(best.metrics)
    data: dict[str, str | float] = {
        "best_model": best.model_name.upper(),
        "best_f1": _stat_val(metrics, "test/f1").mean,
        "best_auroc": _stat_val(metrics, "test/auroc").mean,
        "best_t_per_unit_batch": _stat_val(metrics, "test/t_per_unit_batch").mean * 1e6,
    }

    return [
        StatItem(label=field.label, value=field.render(data.get(field.key)), desc=field.desc)
        for field in sorted(SUMMARY.fields, key=lambda f: f.order)
    ]


def _rank_values(runs: list[dict[str, Any]], key: str, reverse: bool = True) -> dict[str, int]:
    """Compute ranks for a column (0 = best). Uses mean for Stat values."""
    values: list[tuple[str, float]] = []
    for r in runs:
        val = r.get(key, 0)
        mean = val.mean if isinstance(val, Stat) else val
        values.append((r["name"], mean))
    sorted_vals = sorted(values, key=lambda x: x[1], reverse=reverse)
    return {name: i for i, (name, _) in enumerate(sorted_vals)}


def _styled_cell(value: str, rank: int) -> str:
    """Wrap value in styled span based on rank."""
    if rank == 0:
        return f'<span class="rank-1">{value}</span>'
    elif rank == 1:
        return f'<span class="rank-2">{value}</span>'
    return value


def _styled_badge(name: str, rank: int) -> str:
    """Model name badge styled by rank."""
    if rank == 0:
        return f'<span class="badge rank-best">{name}</span>'
    elif rank == 1:
        return f'<span class="badge rank-good">{name}</span>'
    return f'<span class="badge badge-light">{name}</span>'


def _format_stat(s: Stat, fmt: str = ".3f", suffix: str = "") -> str:
    """Format a Stat as mean±std or just mean if n=1."""
    if s.n == 1 or s.std == 0:
        return f"{s.mean:{fmt}}{suffix}"
    return f"{s.mean:{fmt}}±{s.std:{fmt}}{suffix}"


def _format_stat_pct(s: Stat) -> str:
    """Format a Stat as percentage with mean±std."""
    if s.n == 1 or s.std == 0:
        return f"{s.mean * 100:.1f}%"
    return f"{s.mean * 100:.1f}±{s.std * 100:.1f}%"


def build_results_table(runs: list[dict[str, Any]]) -> TableData | None:
    """Build results table with rank-based styling."""
    if not runs:
        return None

    # Compute ranks per column (by mean)
    ranks = {
        "acc": _rank_values(runs, "acc"),
        "f1": _rank_values(runs, "f1"),
        "auroc": _rank_values(runs, "auroc"),
        "precision": _rank_values(runs, "precision"),
        "recall": _rank_values(runs, "recall"),
        "t_per_unit_batch": _rank_values(runs, "t_per_unit_batch", reverse=False),  # Lower is better
    }

    headers = METRICS.headers()
    rows: list[tuple[str, ...]] = []

    for r in runs:
        name = r["name"]
        f1_rank = ranks["f1"][name]

        row = (
            _styled_badge(name, f1_rank),
            _styled_cell(_format_stat_pct(r["acc"]), ranks["acc"][name]),
            _styled_cell(_format_stat(r["f1"]), ranks["f1"][name]),
            _styled_cell(_format_stat(r["auroc"]), ranks["auroc"][name]),
            _styled_cell(_format_stat(r["precision"]), ranks["precision"][name]),
            _styled_cell(_format_stat(r["recall"]), ranks["recall"][name]),
            _styled_cell(
                _format_stat(r["t_per_unit_batch"], ".1f", "μs"),
                ranks["t_per_unit_batch"][name],
            ),
        )
        rows.append(row)

    return TableData(headers=headers, rows=tuple(rows))


def build_hparams_table(runs: list[dict[str, Any]]) -> TableData | None:
    """Build hyperparameters table with badge styling."""
    if not runs:
        return None

    headers = HPARAMS.headers()
    rows: list[tuple[str, ...]] = []

    for r in runs:
        row = (
            f'<span class="badge badge-light">{r["name"]}</span>',
            f"{r['lr']:.0e}",
            f"{r['dropout']:.2f}",
            r["params"],
            f'<span class="mono">{r["arch"]}</span>',
        )
        rows.append(row)

    return TableData(headers=headers, rows=tuple(rows))


def build_arch_items(runs: list[dict[str, Any]], log_dir: Path) -> list[ArchItem]:
    """Build architecture card items with diagrams."""
    items: list[ArchItem] = []
    for r in runs:
        diagram = _generate_diagram(r["name"].lower(), log_dir)
        items.append(ArchItem(name=r["name"], params=r["params"], arch=r["arch"], diagram=diagram))
    return items


def _generate_diagram(model_name: str, log_dir: Path) -> str:
    """Generate architecture diagram for a model."""
    import re

    from waveform_classification.nets import bottleneck, dct, dppv, fcn, mlp, rproj, wavelet

    # Find checkpoint from latest run directory (g001_s1 pattern)
    model_dir = log_dir / model_name
    if not model_dir.exists():
        return ""

    run_pattern = re.compile(r"^g(\d+)_s(\d+)$")
    run_dirs = [p for p in model_dir.iterdir() if p.is_dir() and run_pattern.match(p.name)]
    if not run_dirs:
        return ""

    latest_dir = max(run_dirs, key=lambda p: p.stat().st_mtime)
    ckpts = [p for p in latest_dir.glob("checkpoints/*.ckpt") if p.name != "last.ckpt"]
    if not ckpts:
        return ""

    ckpt_path = max(ckpts, key=lambda p: p.stat().st_mtime)

    # Load model
    model_classes = {
        "mlp": mlp.Model,
        "fcn": fcn.Model,
        "bottleneck": bottleneck.Model,
        "dct": dct.Model,
        "dppv": dppv.Model,
        "rproj": rproj.Model,
        "wavelet": wavelet.Model,
    }

    model_class = model_classes.get(model_name)
    if model_class is None:
        return ""

    try:
        model = model_class.load_from_checkpoint(ckpt_path, map_location="cpu")
        model.eval()
        return model_to_base64_png(model)
    except (RuntimeError, KeyError, OSError):
        return ""


# =============================================================================
# Chart Data
# =============================================================================


@attrs.frozen
class ChartCollection:
    """All chart HTML fragments."""

    comparison: str = ""
    efficiency: str = ""
    radar: str = ""
    f1_by_class: str = ""
    training: str = ""
    class_dist: str = ""


def build_charts(
    runs_with_data: list[AggregatedRun],
    f1_scores: dict[str, float],
    histories: dict[str, TrainingHistory],
    class_counts: dict[str, int],
) -> ChartCollection:
    """Build all chart HTML from data."""
    return ChartCollection(
        comparison=_fig_html(charts.comparison_bars(runs_with_data)) if runs_with_data else "",
        efficiency=_fig_html(charts.t_per_unit_batch_scatter(runs_with_data)) if runs_with_data else "",
        radar="",  # Removed: not useful for Pareto comparison
        f1_by_class=_fig_html(charts.f1_by_class_chart(f1_scores)) if f1_scores else "",
        training=_fig_html(charts.training_chart(histories)),
        class_dist=_fig_html(charts.class_dist_chart(class_counts)),
    )


# =============================================================================
# Export
# =============================================================================


@attrs.frozen
class Footer:
    """Footer content."""

    left: str
    center: str
    right: str


def gen_report(log_dir: Path, report_path: Path) -> None:
    """Export report to static HTML."""
    charts.register_theme()

    # Extract raw data
    source = CSVSource(log_dir) if log_dir.exists() else None
    raw_runs = _get_runs(source)
    histories = _get_histories(source, raw_runs)

    data = Data()
    class_counts = {name: int((data.labels == i).sum()) for i, name in enumerate(data.classnames)}

    # Filter runs with test metrics
    runs_with_data = [r for r in raw_runs if "test/f1" in r.metrics and r.metrics["test/f1"].mean > 0]

    # Prepare run data (sorted by F1 mean)
    sorted_runs = sorted(
        raw_runs,
        key=lambda r: r.metrics.get("test/f1", Stat(0, 0, 1)).mean,
        reverse=True,
    )
    runs = [_run_to_dict(r) for r in sorted_runs]

    # Best model (by F1 mean)
    best = max(runs_with_data, key=lambda r: r.metrics["test/f1"].mean) if runs_with_data else None

    # Per-class F1 (use mean from best model)
    f1_scores: dict[str, float] = {}
    if runs_with_data:
        for name in data.classnames:
            stat = runs_with_data[0].metrics.get(f"test/f1_{name}")
            if stat is not None:
                f1_scores[name] = stat.mean

    # Waveforms
    rng = np.random.default_rng(42)
    waveforms: list[str] = []
    for i, classname in enumerate(data.classnames):
        mask = data.labels == i
        class_waveforms = data.waveforms[mask]
        n = min(3, len(class_waveforms))
        if n > 0:
            indices = rng.choice(len(class_waveforms), size=n, replace=False)
            waveforms.append(_fig_html(charts.waveform_chart(class_waveforms[indices], classname)))

    # Build schema-driven structures
    summary_stats = build_summary_stats(best)
    results_data = build_results_table(runs)
    hparams_data = build_hparams_table(runs)
    arch_data = build_arch_items(runs, log_dir)
    chart_html = build_charts(runs_with_data, f1_scores, histories, class_counts)

    # Render template
    template = ENV.get_template("layout.jinja")
    html = template.render(
        # Page metadata
        title="Lightning Waveform Classification Report",
        heading="Lightning Waveform Classification",
        subtitle="Deep Learning for Atmospheric Discharge Pattern Recognition",
        # Schema-driven data
        summary_stats=summary_stats,
        results_data=results_data,
        hparams_data=hparams_data,
        arch_data=arch_data,
        # Charts
        charts=chart_html,
        # Reference data (already schema-driven via attrs)
        classes=CLASSES,
        config=CONFIG,
        evaluation=EVALUATION,
        # Waveforms
        waveforms=waveforms,
        # Footer
        footer=Footer(
            left="Waveform Classification Lightning Classification",
            center=f"{len(data.waveforms):,} samples | {data.n_class} classes",
            right="PyTorch Lightning",
        ),
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(html)
    print(f"[green]Report exported:[/green] {report_path}")

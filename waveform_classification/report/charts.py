"""Plotly chart functions for waveform classification reports.

Pure Plotly, no Dash dependencies.
"""
# pyright: reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

if TYPE_CHECKING:
    from waveform_classification.report.extract import AggregatedRun, TrainingHistory


COLORS = {
    "surface": "#f8f7f4",
    "surface_elevated": "#ffffff",
    "text": "#1a1a18",
    "text_muted": "#5a6268",
    "accent": "#2d8a7a",
    "accent_dark": "#247a6a",
    "amber": "#c4854a",
    "success": "#2d7a4a",
    "slate": "#6a7a8a",
    "border": "#dcdad6",
}


def register_theme() -> None:
    """Register Plotly template. Call once at startup."""
    if "waveform_classification" in pio.templates:
        return

    c = COLORS
    pio.templates["waveform_classification"] = {
        "layout": {
            "paper_bgcolor": c["surface"],
            "plot_bgcolor": c["surface_elevated"],
            "font": {"family": "Source Sans 3, system-ui, sans-serif", "color": c["text"], "size": 12},
            "colorway": [c["accent"], c["amber"], c["success"], c["slate"], c["accent_dark"]],
            "xaxis": {"gridcolor": c["border"], "linecolor": c["border"]},
            "yaxis": {"gridcolor": c["border"], "linecolor": c["border"]},
            "margin": {"l": 50, "r": 20, "t": 40, "b": 40},
        }
    }
    pio.templates.default = "waveform_classification"


def format_arch(model_name: str, hparams: Mapping[str, Any]) -> str:
    """Format architecture string from hparams."""
    match model_name.lower():
        case "mlp":
            dims: Any = hparams.get("hidden_dims", (128, 64, 32))
            if isinstance(dims, (list, tuple)):
                return "->".join(str(d) for d in dims)
            return str(dims)
        case "cnn":
            channels: Any = hparams.get("channels", (32, 64, 128))
            kernels: Any = hparams.get("kernel_sizes", (7, 5, 3))
            if isinstance(channels, (list, tuple)) and isinstance(kernels, (list, tuple)):
                parts = [f"{c}xk{k}" for c, k in zip(channels, kernels, strict=False)]
                return " ".join(parts)
            return str(channels)
        case "pin" | "fourier":
            dims = hparams.get("hidden_dims", (64, 64))
            return f"PIN({dims[0]})" if isinstance(dims, (list, tuple)) else f"PIN({dims})"
        case "spectral":
            n_fft = hparams.get("n_fft", 64)
            dims = hparams.get("hidden_dims", (128, 64))
            if isinstance(dims, (list, tuple)):
                return f"DCT({n_fft})→{'→'.join(str(d) for d in dims)}"
            return f"DCT({n_fft})→{dims}"
        case "rocket":
            n_kernels = hparams.get("n_kernels", 10000)
            return f"ROCKET({n_kernels} kernels)"
        case _:
            return "-"


def training_chart(
    histories: Mapping[str, TrainingHistory],
    metric: str = "val_loss",
) -> go.Figure | None:
    """Training curves."""
    if not histories:
        return None

    import plotly.express as px

    data_rows = []
    for model_name, history in histories.items():
        values = getattr(history, metric, None)
        if values and history.epochs:
            for epoch, value in zip(history.epochs, values, strict=False):
                data_rows.append({"Epoch": int(epoch), "Value": float(value), "Model": model_name.upper()})

    if not data_rows:
        return None

    df = pd.DataFrame(data_rows)
    fig = px.line(df, x="Epoch", y="Value", color="Model")
    fig.update_traces(line={"width": 2})
    fig.update_layout(
        yaxis_title=metric.replace("_", " ").title(),
        hovermode="x unified",
        height=300,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
    )
    return fig


def comparison_bars(
    runs: Sequence[AggregatedRun],
    metrics: Sequence[str] = ("test/acc", "test/f1", "test/auroc"),
) -> go.Figure:
    """Model comparison grouped bar chart."""
    labels = [m.replace("test/", "").upper() for m in metrics]
    colors = [COLORS["accent"], COLORS["amber"], COLORS["success"]]

    # Collect all values for auto-scaling (use mean from Stat)
    all_values: list[float] = []
    fig = go.Figure()
    for metric, label, color in zip(metrics, labels, colors, strict=False):
        values = [r.metrics[metric].mean if metric in r.metrics else 0 for r in runs]
        all_values.extend(values)
        names = [r.model_name.upper() for r in runs]
        fig.add_trace(go.Bar(name=label, x=names, y=values, marker_color=color))

    # Auto-scale with padding
    min_val = min(all_values) - 0.05 if all_values else 0
    max_val = min(max(all_values) + 0.05, 1.0) if all_values else 1

    fig.update_layout(
        barmode="group",
        height=300,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
        yaxis_range=[max(0, min_val), max_val],
    )
    return fig


def class_dist_chart(counts: Mapping[str, int]) -> go.Figure:
    """Class distribution bar chart."""
    sorted_items = sorted(counts.items(), key=lambda x: -x[1])
    names = [k for k, _ in sorted_items]
    values = [v for _, v in sorted_items]

    fig = go.Figure(go.Bar(x=names, y=values, marker_color=COLORS["accent"]))
    fig.update_layout(height=300, xaxis_title="Class", yaxis_title="Samples")
    return fig


def f1_by_class_chart(f1_scores: Mapping[str, float]) -> go.Figure:
    """Per-class F1 score bar chart with performance coloring."""
    names = list(f1_scores.keys())
    values = list(f1_scores.values())
    colors = [COLORS["success"] if v >= 0.8 else COLORS["amber"] if v >= 0.6 else COLORS["slate"] for v in values]

    # Auto-scale with padding
    min_val = min(values) - 0.05 if values else 0
    max_val = min(max(values) + 0.05, 1.0) if values else 1

    fig = go.Figure(go.Bar(x=names, y=values, marker_color=colors))
    fig.update_layout(height=250, yaxis_range=[max(0, min_val), max_val], yaxis_title="F1 Score")
    return fig


def radar_chart(runs: Sequence[AggregatedRun]) -> go.Figure:
    """Radar chart comparing models across metrics."""
    metrics = ["test/acc", "test/f1", "test/auroc", "test/precision", "test/recall"]
    labels = ["ACC", "F1", "AUROC", "PREC", "RECALL"]
    colors = [COLORS["accent"], COLORS["amber"], COLORS["success"], COLORS["slate"]]

    fig = go.Figure()
    for run, color in zip(runs, colors, strict=False):
        values = [run.metrics[m].mean if m in run.metrics else 0 for m in metrics]
        values.append(values[0])
        theta = [*labels, labels[0]]
        fig.add_trace(
            go.Scatterpolar(
                r=values,
                theta=theta,
                name=run.model_name.upper(),
                fill="toself",
                fillcolor=f"rgba({int(color[1:3], 16)},{int(color[3:5], 16)},{int(color[5:7], 16)},0.2)",
                line_color=color,
            )
        )

    fig.update_layout(
        polar={"radialaxis": {"range": [0.5, 1]}},
        height=300,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
    )
    return fig


def t_per_unit_batch_scatter(runs: Sequence[AggregatedRun], baseline: str = "mlp") -> go.Figure:
    """Pareto plot: t_per_unit_batch vs F1 with baseline reference.

    Shows which models beat the baseline (default: MLP) on BOTH metrics.
    Top-left quadrant = faster AND more accurate than baseline.
    """
    import numpy as np

    # Extract data (use mean from Stat values)
    data = [
        {
            "name": run.model_name.upper(),
            "time": run.metrics["test/t_per_unit_batch"].mean * 1e6 if "test/t_per_unit_batch" in run.metrics else 0,
            "f1": run.metrics["test/f1"].mean if "test/f1" in run.metrics else 0,
            "params": run.params or 100000,
        }
        for run in runs
        if "test/t_per_unit_batch" in run.metrics and run.metrics["test/t_per_unit_batch"].mean > 0
    ]

    if not data:
        return go.Figure()

    # Find baseline
    baseline_data = next((d for d in data if d["name"].lower() == baseline.lower()), None)
    if baseline_data is None:
        baseline_data = data[0]  # Fallback to first model

    baseline_time = baseline_data["time"]
    baseline_f1 = baseline_data["f1"]

    # Compute Pareto frontier (non-dominated points: lower time OR higher f1)
    points = sorted(data, key=lambda d: d["time"])
    pareto = []
    max_f1 = -1
    for p in points:
        if p["f1"] > max_f1:
            pareto.append(p)
            max_f1 = p["f1"]

    fig = go.Figure()

    # Compute axis bounds from data with padding
    all_times = [d["time"] for d in data]
    all_f1s = [d["f1"] for d in data]
    min_time = min(all_times) * 0.8
    max_time = max(all_times) * 1.2
    min_f1 = min(all_f1s) - 0.05
    max_f1 = max(all_f1s) + 0.05

    # Shade "wins both" quadrant (top-left of baseline)
    fig.add_shape(
        type="rect",
        x0=min_time,
        x1=baseline_time,
        y0=baseline_f1,
        y1=1,
        fillcolor="rgba(45, 138, 122, 0.1)",
        line_width=0,
        layer="below",
    )

    # Baseline reference lines
    fig.add_hline(y=baseline_f1, line_dash="dash", line_color=COLORS["text_muted"], line_width=1)
    fig.add_vline(x=baseline_time, line_dash="dash", line_color=COLORS["text_muted"], line_width=1)

    # Pareto frontier line
    if len(pareto) > 1:
        pareto_times = [p["time"] for p in pareto]
        pareto_f1s = [p["f1"] for p in pareto]
        fig.add_trace(
            go.Scatter(
                x=pareto_times,
                y=pareto_f1s,
                mode="lines",
                line={"color": COLORS["accent"], "width": 2, "dash": "dot"},
                name="Pareto frontier",
                showlegend=False,
            )
        )

    # Plot points with color based on "beats baseline on both"
    for d in data:
        beats_both = d["time"] < baseline_time and d["f1"] > baseline_f1
        is_baseline = d["name"].lower() == baseline.lower()

        color = COLORS["success"] if beats_both else (COLORS["amber"] if is_baseline else COLORS["slate"])
        symbol = "diamond" if is_baseline else ("star" if beats_both else "circle")

        fig.add_trace(
            go.Scatter(
                x=[d["time"]],
                y=[d["f1"]],
                mode="markers+text",
                marker={"size": 15, "color": color, "symbol": symbol, "line": {"width": 1, "color": "white"}},
                text=[d["name"]],
                textposition="top center",
                textfont={"size": 11, "color": COLORS["text"]},
                name=d["name"],
                showlegend=False,
            )
        )

    # Layout with log scale on time axis
    fig.update_layout(
        xaxis={
            "title": "Inference Time (μs) — log scale",
            "type": "log",
            "range": [np.log10(min_time), np.log10(max_time)],
            "gridcolor": COLORS["border"],
        },
        yaxis={
            "title": "F1 Score",
            "range": [min_f1, max_f1],
            "gridcolor": COLORS["border"],
        },
        height=350,
        margin={"t": 40, "b": 50},
        annotations=[
            {
                "x": min_time * 1.5,
                "y": (baseline_f1 + 1) / 2,
                "text": "✓ Beats " + baseline.upper(),
                "showarrow": False,
                "font": {"size": 10, "color": COLORS["success"]},
            }
        ],
    )

    return fig


def waveform_chart(waveforms: Any, classname: str) -> go.Figure:
    """Overlaid waveform traces."""
    import numpy as np
    import plotly.express as px

    n_samples, n_points = waveforms.shape  # type: ignore[union-attr]
    df = pd.DataFrame(
        {
            "Time": np.tile(np.arange(n_points), n_samples),
            "Amplitude": waveforms.flatten(),
            "Sample": np.repeat([f"S{i + 1}" for i in range(n_samples)], n_points),
        }
    )

    fig = px.line(df, x="Time", y="Amplitude", color="Sample")
    fig.update_traces(line={"width": 1.2}, opacity=0.7)
    fig.update_layout(showlegend=False, height=200, title=classname, margin={"t": 30, "b": 20})
    return fig

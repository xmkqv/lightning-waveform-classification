"""Tests for static HTML export."""

from pathlib import Path

import numpy as np

from waveform_classification.report.export import gen_report


def test_export_renders_a_standalone_report(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Export produces a report from the documented dataset layout."""
    data_dir = tmp_path / "datasets/waveform/lightning"
    data_dir.mkdir(parents=True)
    for name in ("01", "02", "03", "04", "05", "06", "07", "08", "09", "010"):
        np.save(data_dir / f"{name}.npy", np.zeros((2, 1000), dtype=np.float32))

    monkeypatch.chdir(tmp_path)
    output = tmp_path / "report.html"
    gen_report(tmp_path / "logs", output)
    html = output.read_text()

    assert all(
        text in html
        for text in (
            "Lightning Waveform Classification",
            "Reference",
            "Data Overview",
            "Sample Waveforms",
            "plotly-graph-div",
        )
    )
    assert "CCA Lightning Classification" not in html

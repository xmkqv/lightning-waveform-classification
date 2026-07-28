"""Report generation for waveform classification.

Pure Jinja/Plotly export system. No Dash.
"""

from waveform_classification.report import charts
from waveform_classification.report.charts import COLORS, register_theme
from waveform_classification.report.export import gen_report
from waveform_classification.report.extract import CSVSource, MetricSource, MockSource

__all__ = [
    "COLORS",
    "CSVSource",
    "MetricSource",
    "MockSource",
    "charts",
    "gen_report",
    "register_theme",
]

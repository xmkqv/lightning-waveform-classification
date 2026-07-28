"""Schema-driven field definitions for report generation.

Single source of truth for:
- Field keys and display labels
- Formatting specifications
- Groupings (metrics, hparams, stats)
- Display order

Adding a new field = one line here. No template changes.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from enum import Enum
from typing import Any, Literal

import attrs

# =============================================================================
# Field Types
# =============================================================================


class FieldKind(Enum):
    """Classification of field source."""

    METRIC = "metric"  # test/val metrics from TensorBoard
    HPARAM = "hparam"  # hyperparameters
    DERIVED = "derived"  # computed fields (arch string, etc.)
    STATIC = "static"  # constant/config values


@attrs.frozen
class Field:
    """Definition of a single displayable field.

    Attributes:
        key: Internal data key (e.g., "test/f1", "lr")
        label: Display label (e.g., "F1 Score", "Learning Rate")
        format: Python format spec (e.g., ".3f", ".0e", ",d")
        suffix: Optional suffix (e.g., "μs", "%")
        desc: Description for tooltips/help
        kind: Field classification
        order: Display ordering within group
        transform: Optional value transform before formatting
    """

    key: str
    label: str
    format: str = "s"
    suffix: str = ""
    desc: str = ""
    kind: FieldKind = FieldKind.METRIC
    order: int = 0
    transform: Callable[[Any], Any] | None = None

    def render(self, value: Any) -> str:
        """Format value according to field spec."""
        if value is None:
            return "-"
        if self.transform:
            value = self.transform(value)
        try:
            formatted = f"{value:{self.format}}"
        except (ValueError, TypeError):
            formatted = str(value)
        return f"{formatted}{self.suffix}"


@attrs.frozen
class FieldGroup:
    """Named group of related fields."""

    name: str
    title: str
    fields: tuple[Field, ...]

    def headers(self) -> tuple[str, ...]:
        """Column headers in display order."""
        return tuple(f.label for f in sorted(self.fields, key=lambda f: f.order))

    def keys(self) -> tuple[str, ...]:
        """Field keys in display order."""
        return tuple(f.key for f in sorted(self.fields, key=lambda f: f.order))

    def get(self, key: str) -> Field | None:
        """Lookup field by key."""
        for f in self.fields:
            if f.key == key:
                return f
        return None


# =============================================================================
# Schema Definitions
# =============================================================================


def _pct(v: float) -> float:
    """Convert ratio to percentage."""
    return v * 100


# Test metrics displayed in results table
METRICS = FieldGroup(
    name="metrics",
    title="Test Results",
    fields=(
        Field("name", "Model", "s", "", "Model name", FieldKind.DERIVED, 0),
        Field("acc", "Acc", ".1f", "%", "Test accuracy", transform=_pct, order=1),
        Field("f1", "F1", ".3f", "", "Macro F1 score", order=2),
        Field("auroc", "AUROC", ".3f", "", "Area under ROC curve", order=3),
        Field("precision", "Prec", ".3f", "", "Macro precision", order=4),
        Field("recall", "Recall", ".3f", "", "Macro recall", order=5),
        Field("t_per_unit_batch", "t_per_unit_batch", ".3f", "μs", "Inference time (batch=1)", order=6),
    ),
)

# Hyperparameters displayed in hparams table
HPARAMS = FieldGroup(
    name="hparams",
    title="Hyperparameters",
    fields=(
        Field("name", "Model", "s", "", "Model name", FieldKind.DERIVED, 0),
        Field("lr", "LR", ".0e", "", "Learning rate", FieldKind.HPARAM, 1),
        Field("dropout", "Dropout", ".2f", "", "Dropout rate", FieldKind.HPARAM, 2),
        Field("params", "Params", "s", "", "Parameter count", FieldKind.DERIVED, 3),
        Field("arch", "Arch", "s", "", "Architecture", FieldKind.DERIVED, 4),
    ),
)

# Summary statistics (hero cards at top)
SUMMARY = FieldGroup(
    name="summary",
    title="Summary",
    fields=(
        Field("best_model", "Best Model", "s", "", "by F1 score", order=0),
        Field("best_f1", "F1 Score", ".3f", "", "macro average", order=1),
        Field("best_auroc", "AUROC", ".3f", "", "macro average", order=2),
        Field("best_t_per_unit_batch", "t_per_unit_batch", ".1f", "μs", "batch=1", order=3),
    ),
)

# =============================================================================
# Display Row
# =============================================================================


@attrs.define
class DisplayRow:
    """A row of data with schema-driven formatting.

    Provides:
    - Attribute access for raw values
    - row_tuple: pre-formatted tuple (for generic table macro)
    - cells: dict for named access to formatted values
    """

    data: Mapping[str, Any]
    schema: FieldGroup

    def __getattr__(self, key: str) -> Any:
        return self.data.get(key)

    @property
    def row_tuple(self) -> tuple[str, ...]:
        """Pre-formatted row tuple in schema order."""
        return tuple(
            field.render(self.data.get(field.key)) for field in sorted(self.schema.fields, key=lambda f: f.order)
        )

    @property
    def cells(self) -> dict[str, str]:
        """Dict of field key -> formatted value."""
        return {field.key: field.render(self.data.get(field.key)) for field in self.schema.fields}


def make_rows(data: Sequence[Mapping[str, Any]], schema: FieldGroup) -> list[DisplayRow]:
    """Create DisplayRows from raw data dicts."""
    return [DisplayRow(d, schema) for d in data]


# =============================================================================
# Section Definitions
# =============================================================================


SectionKind = Literal["stats", "table", "chart", "chart_grid", "cards", "list"]


@attrs.frozen
class Section:
    """Definition of a report section.

    Attributes:
        id: Unique section identifier
        title: Display title (None for untitled sections)
        kind: Section type for template dispatch
        schema: Optional field group for data sections
        grid_cols: Column count for grids
    """

    id: str
    title: str | None
    kind: SectionKind
    schema: FieldGroup | None = None
    grid_cols: int = 2


# Report layout as ordered sections
SECTIONS: tuple[Section, ...] = (
    Section("summary", None, "stats", SUMMARY),
    Section("results", "Test Results", "table", METRICS),
    Section("comparison", "Model Comparison", "chart_grid", grid_cols=2),
    Section("metrics_detail", None, "chart_grid", grid_cols=2),
    Section("training", "Training Dynamics", "chart"),
    Section("hparams", "Hyperparameters", "table", HPARAMS),
    Section("reference", "Reference", "chart_grid", grid_cols=2),
    Section("data_overview", "Data Overview", "chart"),
    Section("waveforms", "Sample Waveforms", "chart_grid", grid_cols=3),
)


def get_section(section_id: str) -> Section | None:
    """Lookup section by ID."""
    for s in SECTIONS:
        if s.id == section_id:
            return s
    return None


# =============================================================================
# Static Reference Data
# =============================================================================


@attrs.frozen
class ClassInfo:
    """Lightning discharge class metadata."""

    code: str
    type: str
    desc: str


@attrs.frozen
class ConfigItem:
    """Training configuration key-value pair."""

    key: str
    value: str


CLASSES: tuple[ClassInfo, ...] = (
    ClassInfo("-CG", "Cloud-to-Ground", "Negative polarity"),
    ClassInfo("+CG", "Cloud-to-Ground", "Positive polarity"),
    ClassInfo("CG-IR", "CG + Initial Return", "Multi-stroke"),
    ClassInfo("NBE", "Narrow Bipolar", "Fast transient"),
    ClassInfo("CC", "Cloud-to-Cloud", "Intra-cloud"),
    ClassInfo("PBP", "Preliminary Breakdown", "Pre-discharge"),
    ClassInfo("Sky Wave", "Ionospheric", "Long-range"),
)

CONFIG: tuple[ConfigItem, ...] = (
    ConfigItem("Loss", "Class-weighted cross-entropy"),
    ConfigItem("Optimizer", "AdamW (lr=1e-3)"),
    ConfigItem("Scheduler", "ReduceLROnPlateau"),
    ConfigItem("Early stopping", "patience=10"),
    ConfigItem("Split", "80/10/10 (stratified)"),
)

EVALUATION: tuple[ConfigItem, ...] = (
    ConfigItem("Classification", "Macro F1, Precision, Recall"),
    ConfigItem("Discrimination", "Macro AUROC"),
    ConfigItem("Efficiency", "t_per_unit_batch (μs)"),
)

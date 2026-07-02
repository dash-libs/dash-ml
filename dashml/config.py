"""Plain-dataclass config objects for an ML monitoring/lifecycle run.

Built from ipywidgets form values (see ui.py), not loaded from a config
file — there's no CLI or project scaffold in this package, just the
notebook widget.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelSpec:
    type: str = "classifier"  # classifier | regressor | clustering | knn | anomaly
    target_column: Optional[str] = None
    feature_columns: list[str] = field(default_factory=list)

    @property
    def is_supervised(self) -> bool:
        return self.type in ("classifier", "regressor")


@dataclass
class DerivedColumn:
    name: str
    expression: str  # evaluated with DataFrame.eval()
    as_binary: bool = False


@dataclass
class CleaningSpec:
    standardize_column_names: bool = True
    dedup_by: Optional[str] = None
    drop_nulls_in: list[str] = field(default_factory=list)
    date_columns: list[str] = field(default_factory=list)
    categorical_mappings: dict[str, dict] = field(default_factory=dict)
    derived_columns: list[DerivedColumn] = field(default_factory=list)
    filters: list[dict] = field(default_factory=list)


@dataclass
class EvaluationSpec:
    min_thresholds: dict[str, float] = field(default_factory=dict)
    compare_to_champion: bool = True
    run_shap: bool = True
    build_model_card: bool = True


@dataclass
class MonitoringSpec:
    drift_threshold: float = 0.15  # PSI: >0.25 significant, 0.10-0.25 moderate
    row_count_delta_threshold: float = 0.05
    auto_retrain: bool = False
    retrain_job_name: Optional[str] = None


@dataclass
class RunConfig:
    """Top-level config for one model's lifecycle run."""
    name: str
    catalog: str
    schema_name: str
    model: ModelSpec = field(default_factory=ModelSpec)
    cleaning: CleaningSpec = field(default_factory=CleaningSpec)
    evaluation: EvaluationSpec = field(default_factory=EvaluationSpec)
    monitoring: MonitoringSpec = field(default_factory=MonitoringSpec)

    @property
    def registered_name(self) -> str:
        return f"{self.catalog}.{self.schema_name}.{self.name}"

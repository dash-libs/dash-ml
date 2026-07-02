"""DashML — ML lifecycle management for Databricks: preprocessing, evaluation,
drift monitoring, governance, and champion/challenger promotion."""
from dashml.config import CleaningSpec, DerivedColumn, EvaluationSpec, ModelSpec, MonitoringSpec, RunConfig
from dashml.evaluation import build_model_card, check_thresholds, explain_features, plot_feature_importance
from dashml.governance import build_governance_artifacts
from dashml.monitor import ModelMonitor
from dashml.preprocessing import clean_dataframe
from dashml.registry import RunTracker, promote_challenger, register_model
from dashml.ui import launch

__version__ = "0.1.0"
__all__ = [
    "RunConfig",
    "ModelSpec",
    "CleaningSpec",
    "DerivedColumn",
    "EvaluationSpec",
    "MonitoringSpec",
    "ModelMonitor",
    "clean_dataframe",
    "check_thresholds",
    "explain_features",
    "plot_feature_importance",
    "build_model_card",
    "build_governance_artifacts",
    "RunTracker",
    "register_model",
    "promote_challenger",
    "launch",
]

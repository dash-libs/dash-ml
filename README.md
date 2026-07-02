# DashML — Databricks Library

[![CI](https://github.com/dash-libs/dash-ml/actions/workflows/ci.yml/badge.svg)](https://github.com/dash-libs/dash-ml/actions)
[![PyPI](https://img.shields.io/pypi/v/dash-mlops)](https://pypi.org/project/dash-mlops/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)

Part of the **[Dashlibs](https://github.com/dash-libs)** suite — Databricks libraries built for business users.

ML lifecycle management: preprocessing, drift monitoring, evaluation (SHAP +
model cards), governance artifacts, and champion/challenger promotion —
driven from one notebook UI, backed by Unity Catalog and MLflow.

## Installation

```bash
%pip install dash-mlops
```

## Quick Start

```python
import dashml
dashml.launch()   # Opens interactive UI in your Databricks notebook
```

## What it covers

| Area | Entry points |
|---|---|
| Preprocessing | `clean_dataframe()`, `dashml.transforms` (outlier removal, binning, lag features, ...) |
| Drift monitoring | `ModelMonitor` (PSI + chi-squared, optional auto-retrain trigger) |
| Evaluation | `explain_features()` (SHAP), `build_model_card()`, `check_thresholds()` |
| Governance | `build_governance_artifacts()` (signature, features, fairness, approval record) |
| Registry | `RunTracker`, `register_model()`, `promote_challenger()` (UC `@champion` alias) |
| Experimentation | `dashml.experiment` — compare/promote MLflow runs |
| Serving | `dashml.serving.sync_serving_endpoint()` |

Everything beyond `ModelMonitor` and the notebook UI is also directly
importable for use in a training script — `launch()` is the guided path,
not the only path.

## Part of Dashlibs

| Library | Purpose |
|---|---|
| dash-dq | Data Quality |
| dash-synthetic | Synthetic Data Generation |
| dash-ml | ML Lifecycle Management |
| dash-ingest | Data Ingestion |
| dash-gov | Data Governance |
| dash-ontology | Ontology & Lineage for AI |

## License

Apache 2.0

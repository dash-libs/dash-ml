from __future__ import annotations


class ModelMonitor:
    """
    Monitor ML model performance and data drift on Databricks.

    Usage::
        monitor = ModelMonitor(model_name="customer_churn", model_version=3)
        monitor.set_baseline(table="catalog.schema.training_data")
        monitor.set_production(table="catalog.schema.inference_logs")
        report = monitor.run()
    """

    def __init__(self, model_name: str, model_version: int | None = None, drift_threshold: float = 0.15):
        self.model_name = model_name
        self.model_version = model_version
        self.drift_threshold = drift_threshold
        self._baseline_df = None
        self._production_df = None
        self._target_col: str | None = None
        self._prediction_col: str | None = None
        self._feature_cols: list[str] = []
        self._retrain_job_name: str | None = None

    def set_baseline(self, df=None, table: str | None = None):
        self._baseline_df = self._load(df, table)
        return self

    def set_production(self, df=None, table: str | None = None):
        self._production_df = self._load(df, table)
        return self

    def set_target(self, column: str):
        self._target_col = column
        return self

    def set_prediction(self, column: str):
        self._prediction_col = column
        return self

    def set_features(self, columns: list[str]):
        self._feature_cols = columns
        return self

    def set_auto_retrain(self, job_name: str):
        """Trigger this Databricks job by name if drift exceeds the threshold on run()."""
        self._retrain_job_name = job_name
        return self

    def _load(self, df, table):
        if df is not None:
            return df
        from pyspark.sql import SparkSession
        return SparkSession.getActiveSession().table(table)

    def run(self, save_to: str | None = None) -> MonitorReport:
        from dashml.metrics import compute_drift, compute_performance, compute_prediction_drift

        results = {}
        if self._baseline_df is not None and self._production_df is not None:
            if self._feature_cols:
                results["drift"] = compute_drift(
                    self._baseline_df, self._production_df, self._feature_cols, threshold=self.drift_threshold
                )
            if self._prediction_col:
                results["prediction_drift"] = compute_prediction_drift(
                    self._baseline_df, self._production_df, self._prediction_col
                )
        if self._target_col and self._prediction_col and self._production_df is not None:
            results["performance"] = compute_performance(
                self._production_df, self._target_col, self._prediction_col
            )

        if self._retrain_job_name and any(f.get("drifted") for f in results.get("drift", {}).values()):
            from dashml.retraining import trigger_job

            results["retraining"] = {"job": self._retrain_job_name, "status": trigger_job(self._retrain_job_name)}

        report = MonitorReport(self.model_name, self.model_version, results)
        if save_to:
            report.save(save_to)
        return report


class MonitorReport:
    def __init__(self, model_name: str, model_version, results: dict):
        self.model_name = model_name
        self.model_version = model_version
        self.results = results

    def display(self):
        print(f"Model: {self.model_name} v{self.model_version}")
        for section, data in self.results.items():
            print(f"\n── {section.upper()} ──")
            if isinstance(data, dict):
                for k, v in data.items():
                    print(f"  {k}: {v}")

    def save(self, delta_table: str):
        from pyspark.sql import SparkSession, functions as F
        spark = SparkSession.getActiveSession()
        rows = [{"model_name": self.model_name, "model_version": str(self.model_version),
                 "section": s, "metric": k, "value": str(v)}
                for s, data in self.results.items()
                for k, v in (data.items() if isinstance(data, dict) else {s: data}.items())]
        spark.createDataFrame(rows).withColumn("run_ts", F.current_timestamp()) \
             .write.format("delta").mode("append").saveAsTable(delta_table)

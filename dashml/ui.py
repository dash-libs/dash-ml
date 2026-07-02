"""DashML interactive UI for Databricks notebooks."""
from __future__ import annotations


def launch():
    try:
        import ipywidgets as w
        from IPython.display import display
    except ImportError:
        raise RuntimeError("ipywidgets required. Run: %pip install ipywidgets")

    import dashui

    tabs = w.Tab(children=[
        _monitor_tab(w, dashui),
        _preprocess_tab(w, dashui),
        _evaluate_tab(w, dashui),
        _promote_tab(w, dashui),
    ])
    for i, title in enumerate(["Monitor", "Preprocess", "Evaluate", "Promote"]):
        tabs.set_title(i, title)

    ui = dashui.card([
        dashui.header("DashML — ML Lifecycle Management", library="dashml", emoji="🧠"),
        tabs,
    ])
    display(ui)


def _monitor_tab(w, dashui):
    model_name = w.Text(description="Model name:", placeholder="e.g. customer_churn")
    model_version = w.IntText(description="Version:", value=1, min=1)
    baseline_table = w.Text(description="Baseline table:", placeholder="catalog.schema.training_data")
    prod_table = w.Text(description="Production table:", placeholder="catalog.schema.inference_logs")
    target_col = w.Text(description="Target column:", placeholder="label")
    pred_col = w.Text(description="Prediction column:", placeholder="prediction")
    feature_cols = w.Text(description="Feature columns:", placeholder="col1, col2, col3 (comma separated)")
    drift_threshold = w.FloatText(description="PSI threshold:", value=0.15, step=0.01)

    retrain_toggle = w.Checkbox(value=False, description="Auto-trigger retraining job on drift")
    retrain_job = w.Text(placeholder="prod-churn-training-job", description="Job name:", disabled=True)
    retrain_toggle.observe(lambda c: setattr(retrain_job, "disabled", not c["new"]), names="value")

    save_toggle = w.Checkbox(value=False, description="Save results to Delta")
    save_input = w.Text(placeholder="catalog.schema.ml_monitor", description="Output table:", disabled=True)
    save_toggle.observe(lambda c: setattr(save_input, "disabled", not c["new"]), names="value")

    run_btn = dashui.action_button("Run Monitoring", style="success", emoji="▶")
    output = dashui.output_panel()

    def on_run(b):
        with output:
            output.clear_output()
            try:
                from dashml.monitor import ModelMonitor

                monitor = ModelMonitor(model_name.value.strip(), model_version.value, drift_threshold.value)
                if baseline_table.value.strip():
                    monitor.set_baseline(table=baseline_table.value.strip())
                if prod_table.value.strip():
                    monitor.set_production(table=prod_table.value.strip())
                if target_col.value.strip():
                    monitor.set_target(target_col.value.strip())
                if pred_col.value.strip():
                    monitor.set_prediction(pred_col.value.strip())
                if feature_cols.value.strip():
                    monitor.set_features([c.strip() for c in feature_cols.value.split(",") if c.strip()])
                if retrain_toggle.value and retrain_job.value.strip():
                    monitor.set_auto_retrain(retrain_job.value.strip())
                report = monitor.run(save_to=save_input.value.strip() if save_toggle.value else None)
                report.display()
            except Exception as e:
                print(f"❌ {e}")

    run_btn.on_click(on_run)

    return w.VBox([
        dashui.section("Model"), w.HBox([model_name, model_version]),
        dashui.section("Data sources"), baseline_table, prod_table,
        dashui.section("Columns"), target_col, pred_col, feature_cols,
        dashui.section("Drift"), drift_threshold,
        w.HBox([retrain_toggle, retrain_job]),
        dashui.section("Output"), save_toggle, save_input,
        run_btn, output,
    ])


def _preprocess_tab(w, dashui):
    src = dashui.source_selector("Source:")
    standardize = w.Checkbox(value=True, description="Standardize column names")
    dedup_by = w.Text(description="Dedup by:", placeholder="id (optional)")
    drop_nulls_in = w.Text(description="Drop nulls in:", placeholder="col1, col2 (optional)")
    date_cols = w.Text(description="Date columns:", placeholder="signup_date (optional)")

    run_btn = dashui.action_button("Preview Cleaned Data", style="info", emoji="🧹")
    output = dashui.output_panel()

    def on_run(b):
        with output:
            output.clear_output()
            try:
                from dashml.config import CleaningSpec
                from dashml.preprocessing import clean_dataframe

                df = src.resolve_df()
                pdf = df.toPandas() if hasattr(df, "toPandas") else df
                spec = CleaningSpec(
                    standardize_column_names=standardize.value,
                    dedup_by=dedup_by.value.strip() or None,
                    drop_nulls_in=[c.strip() for c in drop_nulls_in.value.split(",") if c.strip()],
                    date_columns=[c.strip() for c in date_cols.value.split(",") if c.strip()],
                )
                cleaned = clean_dataframe(pdf, spec)
                print(f"Cleaned shape: {cleaned.shape}")
                print(cleaned.head(10))
            except Exception as e:
                print(f"❌ {e}")

    run_btn.on_click(on_run)

    return w.VBox([
        dashui.section("Source"), src.toggle, src.box,
        dashui.section("Cleaning"), standardize, dedup_by, drop_nulls_in, date_cols,
        run_btn, output,
    ])


def _evaluate_tab(w, dashui):
    run_id_input = w.Text(description="MLflow run ID:", placeholder="(optional) for the model card header")
    model_name = w.Text(description="Model name:", placeholder="churn_predictor")
    version = w.Text(description="Version:", value="1")
    model_type = w.Dropdown(options=["classifier", "regressor", "clustering", "knn", "anomaly"], description="Type:")
    catalog = w.Text(description="Catalog:")
    schema = w.Text(description="Schema:")
    tables = w.Text(description="Tables:", placeholder="table1, table2")
    target_col = w.Text(description="Target col:")
    feature_cols = w.Text(description="Feature cols:", placeholder="col1, col2")
    metrics_input = w.Textarea(description="Metrics:", placeholder="accuracy=0.91\nf1_score=0.87", rows=3)

    run_btn = dashui.action_button("Build Model Card", style="info", emoji="📋")
    output = dashui.output_panel()

    def _parse_kv(text):
        result = {}
        for line in text.splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                try:
                    result[k.strip()] = float(v.strip())
                except ValueError:
                    result[k.strip()] = v.strip()
        return result

    def on_run(b):
        with output:
            output.clear_output()
            try:
                from dashml.evaluation import build_model_card

                card_md = build_model_card(
                    name=model_name.value.strip(),
                    version=version.value.strip(),
                    model_type=model_type.value,
                    catalog=catalog.value.strip(),
                    schema_name=schema.value.strip(),
                    tables=[t.strip() for t in tables.value.split(",") if t.strip()],
                    target_column=target_col.value.strip() or None,
                    feature_columns=[c.strip() for c in feature_cols.value.split(",") if c.strip()],
                    params={},
                    metrics=_parse_kv(metrics_input.value),
                    run_id=run_id_input.value.strip() or None,
                )
                print(card_md)
            except Exception as e:
                print(f"❌ {e}")

    run_btn.on_click(on_run)

    return w.VBox([
        dashui.section("Model"), w.HBox([model_name, version, model_type]),
        dashui.section("Data"), w.HBox([catalog, schema]), tables, target_col, feature_cols,
        dashui.section("Metrics"), metrics_input, run_id_input,
        run_btn, output,
        dashui.html(
            "<div style='font-size:12px;color:#666;margin-top:8px'>"
            "For SHAP feature importance and governance artifacts (fairness report, "
            "approval record, data-source manifest), call "
            "<code>dashml.explain_features()</code> / <code>dashml.build_governance_artifacts()</code> "
            "directly — they need a fitted model / live metrics in scope.</div>"
        ),
    ])


def _promote_tab(w, dashui):
    catalog = w.Text(description="Catalog:")
    schema = w.Text(description="Schema:")
    model_name = w.Text(description="Model name:")
    run_id = w.Text(description="Run ID:", placeholder="MLflow run to evaluate")
    metric_key = w.Text(description="Metric:", value="accuracy")
    direction = w.ToggleButtons(options=["max", "min"], description="Direction:")
    min_threshold = w.Text(description="Min threshold:", placeholder="(optional)")
    dry_run = w.Checkbox(value=True, description="Dry run (don't actually promote)")

    run_btn = dashui.action_button("Evaluate Promotion", style="warning", emoji="🏆")
    output = dashui.output_panel()

    def on_run(b):
        with output:
            output.clear_output()
            try:
                from dashml.registry import promote_challenger

                threshold = float(min_threshold.value.strip()) if min_threshold.value.strip() else None
                status = promote_challenger(
                    catalog=catalog.value.strip(),
                    schema_name=schema.value.strip(),
                    model_name=model_name.value.strip(),
                    run_id=run_id.value.strip(),
                    metric_key=metric_key.value.strip(),
                    min_threshold=threshold,
                    direction=direction.value,
                    dry_run=dry_run.value,
                )
                print(f"Status: {status}")
            except Exception as e:
                print(f"❌ {e}")

    run_btn.on_click(on_run)

    return w.VBox([
        dashui.section("Model"), w.HBox([catalog, schema]), model_name, run_id,
        dashui.section("Gate"), w.HBox([metric_key, direction]), min_threshold, dry_run,
        run_btn, output,
        dashui.html(
            "<div style='font-size:12px;color:#666;margin-top:8px'>"
            "Compares this run's metric against the current <code>@champion</code> alias and, "
            "if it wins and dry run is off, promotes it. For serving-endpoint sync, call "
            "<code>dashml.serving.sync_serving_endpoint()</code> after promoting.</div>"
        ),
    ])

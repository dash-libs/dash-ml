"""DashML interactive UI for Databricks notebooks."""
from __future__ import annotations


def launch():
    try:
        import ipywidgets as w
        from IPython.display import display
    except ImportError:
        raise RuntimeError("ipywidgets required. Run: %pip install ipywidgets")

    from dashml.monitor import ModelMonitor

    model_name = w.Text(description="Model name:", placeholder="e.g. customer_churn")
    model_version = w.IntText(description="Version:", value=1, min=1)
    baseline_table = w.Text(description="Baseline table:", placeholder="catalog.schema.training_data")
    prod_table = w.Text(description="Production table:", placeholder="catalog.schema.inference_logs")
    target_col = w.Text(description="Target column:", placeholder="label")
    pred_col = w.Text(description="Prediction column:", placeholder="prediction")
    feature_cols = w.Text(description="Feature columns:", placeholder="col1, col2, col3 (comma separated)")

    save_toggle = w.Checkbox(value=False, description="Save results to Delta")
    save_input = w.Text(placeholder="catalog.schema.ml_monitor", description="Output table:", disabled=True)
    save_toggle.observe(lambda c: setattr(save_input, "disabled", not c["new"]), names="value")

    run_btn = w.Button(description="▶ Run Monitoring", button_style="success",
                       layout=w.Layout(height="40px"))
    output = w.Output()

    def on_run(b):
        with output:
            output.clear_output()
            try:
                monitor = ModelMonitor(model_name.value.strip(), model_version.value)
                if baseline_table.value.strip():
                    monitor.set_baseline(table=baseline_table.value.strip())
                if prod_table.value.strip():
                    monitor.set_production(table=prod_table.value.strip())
                if target_col.value.strip():
                    monitor.set_target(target_col.value.strip())
                if pred_col.value.strip():
                    monitor.set_prediction(pred_col.value.strip())
                if feature_cols.value.strip():
                    cols = [c.strip() for c in feature_cols.value.split(",") if c.strip()]
                    monitor.set_features(cols)
                save_to = save_input.value.strip() if save_toggle.value else None
                report = monitor.run(save_to=save_to)
                report.display()
            except Exception as e:
                print(f"❌ {e}")

    run_btn.on_click(on_run)

    ui = w.VBox([
        w.HTML("<h2 style='color:#E65100'>📊 DashML — Model Monitoring</h2>"),
        w.HTML("<b>Model</b>"), w.HBox([model_name, model_version]),
        w.HTML("<b>Data sources</b>"), baseline_table, prod_table,
        w.HTML("<b>Columns</b>"), target_col, pred_col, feature_cols,
        w.HTML("<hr>"), save_toggle, save_input, run_btn, output,
    ], layout=w.Layout(padding="16px", border="1px solid #ddd", border_radius="8px"))

    display(ui)

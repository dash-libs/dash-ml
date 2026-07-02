"""Trigger a Databricks job by name — used to kick off retraining when
monitor.py detects drift past the configured threshold."""
from __future__ import annotations


def trigger_job(job_name: str) -> str:
    """Look up a job by name and run it. Returns a status string:
    triggered:run-<id> | skipped:job-not-found | skipped:no-sdk | error:<message>."""
    try:
        from databricks.sdk import WorkspaceClient
    except ImportError:
        return "skipped:no-sdk"

    client = WorkspaceClient()
    try:
        matches = list(client.jobs.list(name=job_name))
        if not matches:
            return "skipped:job-not-found"
        run = client.jobs.run_now(job_id=matches[0].job_id)
        return f"triggered:run-{run.run_id}"
    except Exception as e:
        return f"error:{e}"

"""Sync a Databricks Model Serving endpoint to a given UC model version."""
from __future__ import annotations


def sync_serving_endpoint(
    endpoint_name: str,
    catalog: str,
    schema_name: str,
    model_name: str,
    version: str,
    workload_size: str = "Small",
    scale_to_zero: bool = True,
) -> bool:
    """Create the endpoint if it doesn't exist, else update it to the given
    model version. Returns False (rather than raising) if the Databricks SDK
    isn't available — this package doesn't require it as a hard dependency."""
    try:
        from databricks.sdk import WorkspaceClient
        from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput
    except ImportError:
        return False

    client = WorkspaceClient()
    uc_name = f"{catalog}.{schema_name}.{model_name}"
    served_entity = ServedEntityInput(
        entity_name=uc_name,
        entity_version=version,
        workload_size=workload_size,
        scale_to_zero_enabled=scale_to_zero,
    )

    exists = True
    try:
        client.serving_endpoints.get(endpoint_name)
    except Exception:
        exists = False

    if exists:
        client.serving_endpoints.update_config(name=endpoint_name, served_entities=[served_entity])
    else:
        client.serving_endpoints.create(
            name=endpoint_name,
            config=EndpointCoreConfigInput(served_entities=[served_entity]),
        )
    return True

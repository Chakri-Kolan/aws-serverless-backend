from __future__ import annotations

import argparse
import json
import re
import traceback
import uuid
from datetime import datetime, timezone
from typing import Any

from pyspark.sql import SparkSession

from serving_governor.audit import append_audit_events, ensure_audit_table
from serving_governor.config import load_policies, parse_runtime_defaults
from serving_governor.databricks_api import DatabricksApi
from serving_governor.permissions import build_group_acl_updates
from serving_governor.resolver import (
    build_ai_gateway_payload,
    flatten_policy,
    resolve_policy,
)
from serving_governor.teams import get_webhook_url, post_adaptive_card


def str_to_bool(value: str | bool | None) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Databricks AI Gateway governance controller"
    )
    parser.add_argument("--config-table", default="")
    parser.add_argument("--config-path", default="")
    parser.add_argument("--audit-table", required=True)
    parser.add_argument("--workspace-url", default="")

    parser.add_argument("--teams-webhook-url", default="")
    parser.add_argument("--teams-webhook-secret-scope", default="")
    parser.add_argument("--teams-webhook-secret-key", default="")

    parser.add_argument("--include-endpoints", default="")
    parser.add_argument("--exclude-endpoints", default="")
    parser.add_argument("--dry-run", default="true")

    parser.add_argument("--default-usage-tracking-enabled", default="true")
    parser.add_argument("--default-inference-table-enabled", default="false")
    parser.add_argument("--default-inference-table-catalog", default="workspace_devconunity")
    parser.add_argument("--default-inference-table-schema", default="ai_gateway_governance")
    parser.add_argument("--default-inference-table-prefix", default="ai_gw")
    parser.add_argument("--default-rate-limits-enabled", default="true")
    parser.add_argument("--default-endpoint-qpm", default="1000")
    parser.add_argument("--default-endpoint-tpm", default="100000")
    parser.add_argument("--default-user-qpm", default="50")
    parser.add_argument("--default-user-tpm", default="5000")
    parser.add_argument("--default-principal-rate-limits-json", default="[]")
    parser.add_argument("--default-manage-groups", default="admins")
    parser.add_argument("--default-query-groups", default="")
    parser.add_argument("--default-view-groups", default="")
    parser.add_argument("--downgrade-manage-groups-to", default="CAN_QUERY")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spark = SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()
    api = DatabricksApi()
    run_id = str(uuid.uuid4())
    dry_run = str_to_bool(args.dry_run)

    ensure_audit_table(spark, args.audit_table)

    workspace_url = args.workspace_url or api.workspace_url()
    runtime_defaults = parse_runtime_defaults(args)
    policies = load_policies(spark, args.config_table or None, args.config_path or None)

    if not policies:
        policies = [runtime_defaults.as_policy()]

    groups = api.list_groups()
    endpoints = api.list_serving_endpoints()

    include_re = re.compile(args.include_endpoints) if args.include_endpoints else None
    exclude_re = re.compile(args.exclude_endpoints) if args.exclude_endpoints else None

    webhook_url = get_webhook_url(
        args.teams_webhook_url,
        args.teams_webhook_secret_scope,
        args.teams_webhook_secret_key,
    )

    events: list[dict[str, Any]] = []
    failures: list[str] = []
    undefined_endpoints: list[str] = []

    for endpoint in endpoints:
        endpoint_name = endpoint.get("name") or endpoint.get("endpoint_name")
        if not endpoint_name:
            continue

        if include_re and not include_re.search(endpoint_name):
            continue

        if exclude_re and exclude_re.search(endpoint_name):
            continue

        try:
            detail = api.get_serving_endpoint(endpoint_name)
            endpoint_id = (
                detail.get("id")
                or detail.get("endpoint_id")
                or endpoint.get("id")
                or endpoint.get("endpoint_id")
            )

            policy, match_level = resolve_policy(policies, workspace_url, endpoint_name)
            if policy is None:
                policy = runtime_defaults.as_policy()
                match_level = "global_default"

            if match_level == "global_default":
                undefined_endpoints.append(endpoint_name)

            current_ai_gateway = detail.get("ai_gateway") or {}
            desired_ai_gateway = build_ai_gateway_payload(policy)

            events.append(
                _audit_event(
                    run_id=run_id,
                    workspace_url=workspace_url,
                    endpoint_name=endpoint_name,
                    endpoint_id=endpoint_id,
                    action_type="resolve_policy",
                    success=True,
                    dry_run=dry_run,
                    match_level=match_level,
                    message=f"Resolved policy for endpoint {endpoint_name}",
                    old_state=flatten_policy(policy),
                    new_state={"ai_gateway": desired_ai_gateway},
                )
            )

            if dry_run:
                ai_response = {"dry_run": True}
            else:
                ai_response = api.update_ai_gateway(endpoint_name, desired_ai_gateway)

            events.append(
                _audit_event(
                    run_id=run_id,
                    workspace_url=workspace_url,
                    endpoint_name=endpoint_name,
                    endpoint_id=endpoint_id,
                    action_type="update_ai_gateway",
                    success=True,
                    dry_run=dry_run,
                    match_level=match_level,
                    message=f"Applied AI Gateway config for {endpoint_name}",
                    old_state=current_ai_gateway,
                    new_state=desired_ai_gateway,
                    details=ai_response,
                )
            )

            if endpoint_id:
                perm_payload = api.get_serving_endpoint_permissions(endpoint_id)
                acl_updates = build_group_acl_updates(perm_payload, groups, policy)

                if acl_updates:
                    if dry_run:
                        perm_response = {
                            "dry_run": True,
                            "access_control_list": acl_updates,
                        }
                    else:
                        perm_response = api.update_serving_endpoint_permissions(
                            endpoint_id,
                            acl_updates,
                        )

                    events.append(
                        _audit_event(
                            run_id=run_id,
                            workspace_url=workspace_url,
                            endpoint_name=endpoint_name,
                            endpoint_id=endpoint_id,
                            action_type="update_permissions",
                            success=True,
                            dry_run=dry_run,
                            match_level=match_level,
                            message=f"Updated endpoint permissions for {endpoint_name}",
                            old_state=perm_payload,
                            new_state={"access_control_list": acl_updates},
                            details=perm_response,
                        )
                    )
                else:
                    events.append(
                        _audit_event(
                            run_id=run_id,
                            workspace_url=workspace_url,
                            endpoint_name=endpoint_name,
                            endpoint_id=endpoint_id,
                            action_type="update_permissions",
                            success=True,
                            dry_run=dry_run,
                            match_level=match_level,
                            message=f"No permission changes required for {endpoint_name}",
                            old_state=perm_payload,
                            new_state={},
                        )
                    )

        except Exception as exc:
            failures.append(f"{endpoint_name}: {exc}")
            events.append(
                _audit_event(
                    run_id=run_id,
                    workspace_url=workspace_url,
                    endpoint_name=endpoint_name,
                    endpoint_id=endpoint.get("id") or endpoint.get("endpoint_id"),
                    action_type="endpoint_failure",
                    success=False,
                    dry_run=dry_run,
                    match_level=None,
                    message=str(exc),
                    details={"traceback": traceback.format_exc()},
                )
            )

    append_audit_events(spark, args.audit_table, events)

    if failures or undefined_endpoints:
        summary_lines: list[str] = []

        if failures:
            summary_lines.append(f"Failures: {len(failures)}")
            summary_lines.extend(failures[:10])

        if undefined_endpoints:
            summary_lines.append(
                f"Undefined endpoints using global safety-net: {len(undefined_endpoints)}"
            )
            summary_lines.extend(undefined_endpoints[:20])

        if webhook_url and webhook_url.lower() not in ("none", "", "null"):
            post_adaptive_card(
                webhook_url,
                title="Databricks AI Gateway governance run completed with alerts",
                summary_lines=summary_lines,
                facts={
                    "run_id": run_id,
                    "workspace_url": workspace_url,
                    "dry_run": dry_run,
                    "failure_count": len(failures),
                    "undefined_endpoint_count": len(undefined_endpoints),
                },
            )
        else:
            print("[INFO] No Teams webhook configured - skipping notification.")

    result = {
        "run_id": run_id,
        "workspace_url": workspace_url,
        "dry_run": dry_run,
        "endpoints_seen": len(endpoints),
        "failures": len(failures),
        "undefined_endpoints": len(undefined_endpoints),
    }
    print(json.dumps(result, indent=2))


def _audit_event(
    *,
    run_id: str,
    workspace_url: str,
    endpoint_name: str | None,
    endpoint_id: str | None,
    action_type: str,
    success: bool,
    dry_run: bool,
    match_level: str | None,
    message: str,
    old_state: Any | None = None,
    new_state: Any | None = None,
    details: Any | None = None,
) -> dict[str, Any]:
    return {
        "event_time": datetime.now(timezone.utc),
        "run_id": run_id,
        "workspace_url": workspace_url,
        "endpoint_name": endpoint_name,
        "endpoint_id": endpoint_id,
        "action_type": action_type,
        "success": success,
        "dry_run": dry_run,
        "match_level": match_level,
        "message": message,
        "old_state": old_state,
        "new_state": new_state,
        "details": details,
    }


if __name__ == "__main__":
    main()

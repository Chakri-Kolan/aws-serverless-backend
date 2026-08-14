"""AWS Lambda entry point for the serverless task API.

The module keeps HTTP concerns separate from the DynamoDB repository so the
business behaviour can be tested without AWS credentials.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import uuid
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Protocol

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(os.getenv("LOG_LEVEL", "INFO"))

ALLOWED_STATUSES = {"todo", "in_progress", "done"}
DEFAULT_PAGE_SIZE = 20
MAX_PAGE_SIZE = 100


class ValidationError(ValueError):
    """Raised when a client request cannot be validated."""


class TaskNotFoundError(LookupError):
    """Raised when a requested task does not exist."""


class Table(Protocol):
    def put_item(self, **kwargs: Any) -> Mapping[str, Any]: ...
    def get_item(self, **kwargs: Any) -> Mapping[str, Any]: ...
    def update_item(self, **kwargs: Any) -> Mapping[str, Any]: ...
    def delete_item(self, **kwargs: Any) -> Mapping[str, Any]: ...
    def scan(self, **kwargs: Any) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class HttpResponse:
    status_code: int
    body: Mapping[str, Any] | list[Any] | None

    def as_lambda_response(self, request_id: str) -> dict[str, Any]:
        return {
            "statusCode": self.status_code,
            "headers": {
                "content-type": "application/json",
                "cache-control": "no-store",
                "x-request-id": request_id,
            },
            "body": json.dumps(self.body, default=_json_default),
        }


class TaskRepository:
    """Small DynamoDB repository with explicit conditional writes."""

    def __init__(self, table: Table) -> None:
        self.table = table

    def create(self, title: str, description: str = "") -> dict[str, Any]:
        now = _utc_now()
        task = {
            "id": str(uuid.uuid4()),
            "title": title,
            "description": description,
            "status": "todo",
            "created_at": now,
            "updated_at": now,
        }
        self.table.put_item(Item=task, ConditionExpression="attribute_not_exists(id)")
        return task

    def get(self, task_id: str) -> dict[str, Any]:
        result = self.table.get_item(Key={"id": task_id}, ConsistentRead=True)
        item = result.get("Item")
        if not item:
            raise TaskNotFoundError(task_id)
        return dict(item)

    def list(self, limit: int, cursor: str | None) -> tuple[list[dict[str, Any]], str | None]:
        request: dict[str, Any] = {"Limit": limit}
        if cursor:
            request["ExclusiveStartKey"] = _decode_cursor(cursor)
        result = self.table.scan(**request)
        next_key = result.get("LastEvaluatedKey")
        return [dict(item) for item in result.get("Items", [])], _encode_cursor(next_key)

    def update(self, task_id: str, changes: Mapping[str, Any]) -> dict[str, Any]:
        names: dict[str, str] = {}
        values: dict[str, Any] = {}
        assignments: list[str] = []
        for index, (field, value) in enumerate(changes.items()):
            name_key = f"#field{index}"
            value_key = f":value{index}"
            names[name_key] = field
            values[value_key] = value
            assignments.append(f"{name_key} = {value_key}")

        names["#updated_at"] = "updated_at"
        values[":updated_at"] = _utc_now()
        assignments.append("#updated_at = :updated_at")

        try:
            result = self.table.update_item(
                Key={"id": task_id},
                UpdateExpression="SET " + ", ".join(assignments),
                ExpressionAttributeNames=names,
                ExpressionAttributeValues=values,
                ConditionExpression="attribute_exists(id)",
                ReturnValues="ALL_NEW",
            )
        except Exception as exc:
            if _is_conditional_failure(exc):
                raise TaskNotFoundError(task_id) from exc
            raise
        return dict(result["Attributes"])

    def delete(self, task_id: str) -> None:
        try:
            self.table.delete_item(
                Key={"id": task_id},
                ConditionExpression="attribute_exists(id)",
            )
        except Exception as exc:
            if _is_conditional_failure(exc):
                raise TaskNotFoundError(task_id) from exc
            raise


def lambda_handler(event: Mapping[str, Any], context: Any) -> dict[str, Any]:
    """Route an API Gateway HTTP API v2 event."""

    request_id = getattr(context, "aws_request_id", None) or str(uuid.uuid4())
    try:
        response = route(event, TaskRepository(_get_table()))
    except ValidationError as exc:
        response = HttpResponse(400, {"error": "validation_error", "message": str(exc)})
    except TaskNotFoundError:
        response = HttpResponse(404, {"error": "not_found", "message": "Task not found"})
    except Exception:
        LOGGER.exception("Unhandled request failure", extra={"request_id": request_id})
        response = HttpResponse(500, {"error": "internal_error", "message": "Unexpected server error"})
    return response.as_lambda_response(request_id)


def route(event: Mapping[str, Any], repository: TaskRepository) -> HttpResponse:
    method = _http_method(event)
    path = _path(event)

    if method == "GET" and path == "/health":
        return HttpResponse(200, {"status": "ok", "service": "aws-serverless-backend"})

    if path == "/tasks":
        if method == "POST":
            payload = _json_body(event)
            title = _required_text(payload, "title", max_length=200)
            description = _optional_text(payload, "description", max_length=2000)
            return HttpResponse(201, {"data": repository.create(title, description)})
        if method == "GET":
            query = event.get("queryStringParameters") or {}
            limit = _parse_limit(query.get("limit"))
            items, next_cursor = repository.list(limit, query.get("cursor"))
            return HttpResponse(200, {"data": items, "next_cursor": next_cursor})

    task_id = _task_id(path)
    if task_id:
        if method == "GET":
            return HttpResponse(200, {"data": repository.get(task_id)})
        if method == "PATCH":
            changes = _validate_changes(_json_body(event))
            return HttpResponse(200, {"data": repository.update(task_id, changes)})
        if method == "DELETE":
            repository.delete(task_id)
            return HttpResponse(200, {"deleted": True, "id": task_id})

    return HttpResponse(404, {"error": "not_found", "message": "Route not found"})


def _get_table() -> Table:
    table_name = os.environ.get("TASKS_TABLE")
    if not table_name:
        raise RuntimeError("TASKS_TABLE is not configured")
    import boto3  # Lambda runtime dependency; imported lazily for unit tests.

    return boto3.resource("dynamodb").Table(table_name)


def _http_method(event: Mapping[str, Any]) -> str:
    return str(
        (event.get("requestContext") or {}).get("http", {}).get("method")
        or event.get("httpMethod")
        or ""
    ).upper()


def _path(event: Mapping[str, Any]) -> str:
    path = str(event.get("rawPath") or event.get("path") or "/")
    stage = str((event.get("requestContext") or {}).get("stage") or "")
    if stage and stage != "$default" and path.startswith(f"/{stage}/"):
        path = path[len(stage) + 1 :]
    return path.rstrip("/") or "/"


def _task_id(path: str) -> str | None:
    parts = path.strip("/").split("/")
    return parts[1] if len(parts) == 2 and parts[0] == "tasks" and parts[1] else None


def _json_body(event: Mapping[str, Any]) -> dict[str, Any]:
    raw = event.get("body")
    if raw is None:
        raise ValidationError("A JSON request body is required")
    if event.get("isBase64Encoded"):
        try:
            raw = base64.b64decode(str(raw)).decode("utf-8")
        except Exception as exc:
            raise ValidationError("Request body is not valid base64") from exc
    try:
        value = json.loads(str(raw))
    except json.JSONDecodeError as exc:
        raise ValidationError("Request body must contain valid JSON") from exc
    if not isinstance(value, dict):
        raise ValidationError("Request body must be a JSON object")
    return value


def _required_text(payload: Mapping[str, Any], field: str, max_length: int) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{field} must be a non-empty string")
    value = value.strip()
    if len(value) > max_length:
        raise ValidationError(f"{field} must be at most {max_length} characters")
    return value


def _optional_text(payload: Mapping[str, Any], field: str, max_length: int) -> str:
    value = payload.get(field, "")
    if not isinstance(value, str):
        raise ValidationError(f"{field} must be a string")
    value = value.strip()
    if len(value) > max_length:
        raise ValidationError(f"{field} must be at most {max_length} characters")
    return value


def _validate_changes(payload: Mapping[str, Any]) -> dict[str, Any]:
    unknown = set(payload) - {"title", "description", "status"}
    if unknown:
        raise ValidationError(f"Unsupported fields: {', '.join(sorted(unknown))}")
    changes: dict[str, Any] = {}
    if "title" in payload:
        changes["title"] = _required_text(payload, "title", 200)
    if "description" in payload:
        changes["description"] = _optional_text(payload, "description", 2000)
    if "status" in payload:
        status = payload["status"]
        if status not in ALLOWED_STATUSES:
            raise ValidationError(f"status must be one of: {', '.join(sorted(ALLOWED_STATUSES))}")
        changes["status"] = status
    if not changes:
        raise ValidationError("At least one supported field is required")
    return changes


def _parse_limit(raw: Any) -> int:
    if raw in (None, ""):
        return DEFAULT_PAGE_SIZE
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValidationError("limit must be an integer") from exc
    if not 1 <= value <= MAX_PAGE_SIZE:
        raise ValidationError(f"limit must be between 1 and {MAX_PAGE_SIZE}")
    return value


def _encode_cursor(key: Any) -> str | None:
    if not key:
        return None
    raw = json.dumps(key, default=_json_default, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii")


def _decode_cursor(cursor: str) -> dict[str, Any]:
    try:
        padding = "=" * (-len(cursor) % 4)
        value = json.loads(base64.urlsafe_b64decode(cursor + padding).decode("utf-8"))
    except Exception as exc:
        raise ValidationError("cursor is invalid") from exc
    if not isinstance(value, dict) or not value:
        raise ValidationError("cursor is invalid")
    return value


def _is_conditional_failure(exc: Exception) -> bool:
    response = getattr(exc, "response", {})
    return response.get("Error", {}).get("Code") == "ConditionalCheckFailedException"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json_default(value: Any) -> Any:
    if isinstance(value, Decimal):
        return int(value) if value % 1 == 0 else float(value)
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")

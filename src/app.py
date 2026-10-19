from __future__ import annotations

import base64
import binascii
import json
import logging
import os
import re
import time
import uuid
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import boto3
from boto3.dynamodb.conditions import Key
from botocore.exceptions import ClientError

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())

TABLE_NAME = os.getenv("TABLE_NAME", "")
CORS_ALLOW_ORIGIN = os.getenv("CORS_ALLOW_ORIGIN", "*")
DEFAULT_PAGE_SIZE = 25
MAX_PAGE_SIZE = 100
ITEM_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
ALLOWED_FIELDS = {"name", "description", "metadata"}
_TABLE: Any | None = None


class ApiError(Exception):
    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


def lambda_handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    started = time.monotonic()
    method = str(event.get("httpMethod") or "").upper()
    path = str(event.get("resource") or event.get("path") or "")
    request_id = _request_id(event, context)
    status_code = 500
    error_code: str | None = None

    try:
        if method == "GET" and path == "/health":
            response = _response(200, {"status": "ok", "service": "aws-serverless-backend"})
        else:
            owner_id = _authenticated_owner(event)
            response = _route_authenticated(event, method, path, owner_id)
        status_code = response["statusCode"]
        return response
    except ApiError as exc:
        status_code = exc.status_code
        error_code = exc.code
        return _error_response(exc.status_code, exc.code, exc.message, request_id)
    except ClientError as exc:
        error_code = exc.response.get("Error", {}).get("Code", "AWS_ERROR")
        LOGGER.exception(
            json.dumps(
                {
                    "event": "aws_client_error",
                    "request_id": request_id,
                    "aws_error_code": error_code,
                }
            )
        )
        return _error_response(
            500,
            "INTERNAL_ERROR",
            "The request could not be completed",
            request_id,
        )
    except Exception:
        error_code = "INTERNAL_ERROR"
        LOGGER.exception(json.dumps({"event": "unhandled_error", "request_id": request_id}))
        return _error_response(500, error_code, "The request could not be completed", request_id)
    finally:
        LOGGER.info(
            json.dumps(
                {
                    "event": "request_completed",
                    "request_id": request_id,
                    "method": method,
                    "path": path,
                    "status_code": status_code,
                    "error_code": error_code,
                    "duration_ms": round((time.monotonic() - started) * 1000, 2),
                },
                separators=(",", ":"),
            )
        )


def _route_authenticated(
    event: dict[str, Any], method: str, path: str, owner_id: str
) -> dict[str, Any]:
    path_parameters = event.get("pathParameters") or {}
    item_id = path_parameters.get("itemId")

    if method == "POST" and path == "/items":
        return _create_item(owner_id, _parse_body(event))
    if method == "GET" and path == "/items":
        return _list_items(owner_id, event.get("queryStringParameters") or {})
    if item_id is not None:
        _validate_item_id(str(item_id))
        if method == "GET" and path == "/items/{itemId}":
            return _get_item(owner_id, str(item_id))
        if method == "PUT" and path == "/items/{itemId}":
            return _update_item(owner_id, str(item_id), _parse_body(event))
        if method == "DELETE" and path == "/items/{itemId}":
            return _delete_item(owner_id, str(item_id))

    raise ApiError(404, "NOT_FOUND", "Route not found")


def _create_item(owner_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    fields = _validate_payload(payload, require_name=True)
    now = _utc_now()
    item = {
        "owner_id": owner_id,
        "item_id": str(uuid.uuid4()),
        **fields,
        "created_at": now,
        "updated_at": now,
        "version": 1,
    }

    try:
        _table().put_item(
            Item=item,
            ConditionExpression="attribute_not_exists(owner_id) AND attribute_not_exists(item_id)",
        )
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") == "ConditionalCheckFailedException":
            raise ApiError(409, "CONFLICT", "Item already exists") from exc
        raise

    return _response(201, {"item": _public_item(item)})


def _get_item(owner_id: str, item_id: str) -> dict[str, Any]:
    result = _table().get_item(Key={"owner_id": owner_id, "item_id": item_id})
    item = result.get("Item")
    if item is None:
        raise ApiError(404, "NOT_FOUND", "Item not found")
    return _response(200, {"item": _public_item(item)})


def _list_items(owner_id: str, query: dict[str, str]) -> dict[str, Any]:
    limit = _parse_limit(query.get("limit"))
    kwargs: dict[str, Any] = {
        "KeyConditionExpression": Key("owner_id").eq(owner_id),
        "Limit": limit,
        "ScanIndexForward": False,
    }
    if query.get("next_token"):
        kwargs["ExclusiveStartKey"] = _decode_next_token(query["next_token"], owner_id)

    result = _table().query(**kwargs)
    body: dict[str, Any] = {
        "items": [_public_item(item) for item in result.get("Items", [])],
        "count": len(result.get("Items", [])),
    }
    if result.get("LastEvaluatedKey"):
        body["next_token"] = _encode_next_token(result["LastEvaluatedKey"])
    return _response(200, body)


def _update_item(
    owner_id: str, item_id: str, payload: dict[str, Any]
) -> dict[str, Any]:
    fields = _validate_payload(payload, require_name=False)
    if not fields:
        raise ApiError(400, "VALIDATION_ERROR", "At least one updatable field is required")

    existing_result = _table().get_item(Key={"owner_id": owner_id, "item_id": item_id})
    existing = existing_result.get("Item")
    if existing is None:
        raise ApiError(404, "NOT_FOUND", "Item not found")

    expected_version = int(existing.get("version", 1))
    updated = {
        **existing,
        **fields,
        "updated_at": _utc_now(),
        "version": expected_version + 1,
    }
    try:
        _table().put_item(
            Item=updated,
            ConditionExpression=(
                "attribute_exists(owner_id) AND attribute_exists(item_id) "
                "AND version = :expected_version"
            ),
            ExpressionAttributeValues={":expected_version": expected_version},
        )
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") == "ConditionalCheckFailedException":
            raise ApiError(409, "CONFLICT", "Item was modified by another request") from exc
        raise

    return _response(200, {"item": _public_item(updated)})


def _delete_item(owner_id: str, item_id: str) -> dict[str, Any]:
    try:
        result = _table().delete_item(
            Key={"owner_id": owner_id, "item_id": item_id},
            ConditionExpression="attribute_exists(owner_id) AND attribute_exists(item_id)",
            ReturnValues="ALL_OLD",
        )
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") == "ConditionalCheckFailedException":
            raise ApiError(404, "NOT_FOUND", "Item not found") from exc
        raise

    if not result.get("Attributes"):
        raise ApiError(404, "NOT_FOUND", "Item not found")
    return _response(204, None)


def _authenticated_owner(event: dict[str, Any]) -> str:
    claims = ((event.get("requestContext") or {}).get("authorizer") or {}).get("claims") or {}
    owner_id = claims.get("sub")
    token_use = claims.get("token_use")
    if not isinstance(owner_id, str) or not owner_id:
        raise ApiError(401, "UNAUTHORIZED", "A valid Cognito token is required")
    if token_use != "id":
        raise ApiError(401, "UNAUTHORIZED", "A Cognito ID token is required")
    return owner_id


def _parse_body(event: dict[str, Any]) -> dict[str, Any]:
    raw_body = event.get("body")
    if not isinstance(raw_body, str) or not raw_body.strip():
        raise ApiError(400, "VALIDATION_ERROR", "A JSON request body is required")

    if event.get("isBase64Encoded"):
        try:
            raw_body = base64.b64decode(raw_body, validate=True).decode("utf-8")
        except (binascii.Error, UnicodeDecodeError) as exc:
            raise ApiError(400, "INVALID_JSON", "Request body is not valid base64 JSON") from exc

    try:
        payload = json.loads(raw_body, parse_float=Decimal)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ApiError(400, "INVALID_JSON", "Request body must contain valid JSON") from exc
    if not isinstance(payload, dict):
        raise ApiError(400, "VALIDATION_ERROR", "Request body must be a JSON object")
    return payload


def _validate_payload(payload: dict[str, Any], *, require_name: bool) -> dict[str, Any]:
    unknown = sorted(set(payload) - ALLOWED_FIELDS)
    if unknown:
        raise ApiError(
            400,
            "VALIDATION_ERROR",
            f"Unknown field(s): {', '.join(unknown)}",
        )

    if require_name and "name" not in payload:
        raise ApiError(400, "VALIDATION_ERROR", "name is required")

    validated: dict[str, Any] = {}
    if "name" in payload:
        name = payload["name"]
        if not isinstance(name, str) or not name.strip() or len(name.strip()) > 200:
            raise ApiError(400, "VALIDATION_ERROR", "name must be 1 to 200 characters")
        validated["name"] = name.strip()

    if "description" in payload:
        description = payload["description"]
        if not isinstance(description, str) or len(description) > 2000:
            raise ApiError(
                400,
                "VALIDATION_ERROR",
                "description must be a string up to 2000 characters",
            )
        validated["description"] = description

    if "metadata" in payload:
        metadata = payload["metadata"]
        if not isinstance(metadata, dict):
            raise ApiError(400, "VALIDATION_ERROR", "metadata must be a JSON object")
        encoded = json.dumps(metadata, default=_json_default, separators=(",", ":"))
        if len(encoded.encode("utf-8")) > 8192:
            raise ApiError(400, "VALIDATION_ERROR", "metadata must not exceed 8192 bytes")
        validated["metadata"] = metadata

    return validated


def _validate_item_id(item_id: str) -> None:
    if not ITEM_ID_PATTERN.fullmatch(item_id):
        raise ApiError(400, "VALIDATION_ERROR", "itemId has an invalid format")


def _parse_limit(value: str | None) -> int:
    if value in (None, ""):
        return DEFAULT_PAGE_SIZE
    try:
        limit = int(value)
    except (TypeError, ValueError) as exc:
        raise ApiError(400, "VALIDATION_ERROR", "limit must be an integer") from exc
    if not 1 <= limit <= MAX_PAGE_SIZE:
        raise ApiError(400, "VALIDATION_ERROR", f"limit must be between 1 and {MAX_PAGE_SIZE}")
    return limit


def _encode_next_token(key: dict[str, Any]) -> str:
    raw = json.dumps(key, default=_json_default, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _decode_next_token(token: str, owner_id: str) -> dict[str, str]:
    try:
        padded = token + "=" * (-len(token) % 4)
        decoded = json.loads(base64.b64decode(padded, altchars=b"-_", validate=True))
    except (binascii.Error, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ApiError(400, "INVALID_NEXT_TOKEN", "next_token is invalid") from exc

    if (
        not isinstance(decoded, dict)
        or decoded.get("owner_id") != owner_id
        or not isinstance(decoded.get("item_id"), str)
        or set(decoded) != {"owner_id", "item_id"}
    ):
        raise ApiError(400, "INVALID_NEXT_TOKEN", "next_token is invalid")
    return {"owner_id": owner_id, "item_id": decoded["item_id"]}


def _table() -> Any:
    global _TABLE
    if _TABLE is None:
        if not TABLE_NAME:
            raise RuntimeError("TABLE_NAME is not configured")
        _TABLE = boto3.resource("dynamodb").Table(TABLE_NAME)
    return _TABLE


def _public_item(item: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in item.items() if key != "owner_id"}


def _response(status_code: int, body: dict[str, Any] | None) -> dict[str, Any]:
    response = {
        "statusCode": status_code,
        "headers": {
            "Access-Control-Allow-Origin": CORS_ALLOW_ORIGIN,
            "Content-Type": "application/json",
            "Cache-Control": "no-store",
        },
        "body": (
            ""
            if body is None
            else json.dumps(body, default=_json_default, separators=(",", ":"))
        ),
    }
    return response


def _error_response(status_code: int, code: str, message: str, request_id: str) -> dict[str, Any]:
    return _response(
        status_code,
        {"error": {"code": code, "message": message, "request_id": request_id}},
    )


def _request_id(event: dict[str, Any], context: Any) -> str:
    return str(
        ((event.get("requestContext") or {}).get("requestId"))
        or getattr(context, "aws_request_id", None)
        or uuid.uuid4()
    )


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _json_default(value: Any) -> Any:
    if isinstance(value, Decimal):
        return int(value) if value == value.to_integral_value() else float(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

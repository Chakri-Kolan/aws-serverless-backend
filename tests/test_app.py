from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import boto3
import pytest
from botocore.exceptions import ClientError
from moto import mock_aws

from src import app


@pytest.fixture()
def table(monkeypatch: pytest.MonkeyPatch):
    with mock_aws():
        dynamodb = boto3.resource("dynamodb", region_name="us-east-1")
        test_table = dynamodb.create_table(
            TableName="test-items",
            BillingMode="PAY_PER_REQUEST",
            AttributeDefinitions=[
                {"AttributeName": "owner_id", "AttributeType": "S"},
                {"AttributeName": "item_id", "AttributeType": "S"},
            ],
            KeySchema=[
                {"AttributeName": "owner_id", "KeyType": "HASH"},
                {"AttributeName": "item_id", "KeyType": "RANGE"},
            ],
        )
        monkeypatch.setattr(app, "_TABLE", test_table)
        yield test_table
        monkeypatch.setattr(app, "_TABLE", None)


def event(
    method: str,
    resource: str,
    *,
    owner: str | None = "user-1",
    body: dict[str, Any] | str | None = None,
    item_id: str | None = None,
    query: dict[str, str] | None = None,
) -> dict[str, Any]:
    request: dict[str, Any] = {
        "httpMethod": method,
        "resource": resource,
        "path": resource.replace("{itemId}", item_id or ""),
        "requestContext": {"requestId": "request-123", "authorizer": {}},
        "pathParameters": {"itemId": item_id} if item_id else None,
        "queryStringParameters": query,
    }
    if owner is not None:
        request["requestContext"]["authorizer"]["claims"] = {
            "sub": owner,
            "token_use": "id",
        }
    if isinstance(body, dict):
        request["body"] = json.dumps(body)
    elif body is not None:
        request["body"] = body
    return request


def invoke(request: dict[str, Any]) -> tuple[int, Any]:
    response = app.lambda_handler(request, SimpleNamespace(aws_request_id="lambda-123"))
    body = json.loads(response["body"]) if response["body"] else None
    return response["statusCode"], body


def create_item(owner: str = "user-1", **overrides: Any) -> dict[str, Any]:
    payload = {"name": "Example", "description": "An item", "metadata": {"score": 1.5}}
    payload.update(overrides)
    status, body = invoke(event("POST", "/items", owner=owner, body=payload))
    assert status == 201
    return body["item"]


def test_health_is_public_and_does_not_require_dynamodb() -> None:
    response = app.lambda_handler(
        event("GET", "/health", owner=None),
        SimpleNamespace(aws_request_id="lambda-123"),
    )
    assert response["statusCode"] == 200
    assert response["headers"]["Access-Control-Allow-Origin"] == "*"
    assert json.loads(response["body"]) == {
        "status": "ok",
        "service": "aws-serverless-backend",
    }


def test_protected_route_requires_cognito_claims(table: Any) -> None:
    status, body = invoke(event("GET", "/items", owner=None))
    assert status == 401
    assert body["error"] == {
        "code": "UNAUTHORIZED",
        "message": "A valid Cognito token is required",
        "request_id": "request-123",
    }


def test_protected_route_rejects_non_id_token_claims(table: Any) -> None:
    request = event("GET", "/items")
    request["requestContext"]["authorizer"]["claims"]["token_use"] = "access"

    status, body = invoke(request)

    assert status == 401
    assert body["error"]["code"] == "UNAUTHORIZED"
    assert body["error"]["message"] == "A Cognito ID token is required"


def test_crud_lifecycle_and_owner_isolation(table: Any) -> None:
    item = create_item()
    assert "owner_id" not in item
    assert item["version"] == 1

    status, body = invoke(event("GET", "/items/{itemId}", item_id=item["item_id"]))
    assert status == 200
    assert body["item"]["name"] == "Example"

    other_status, _ = invoke(
        event("GET", "/items/{itemId}", item_id=item["item_id"], owner="user-2")
    )
    assert other_status == 404

    status, body = invoke(
        event(
            "PUT",
            "/items/{itemId}",
            item_id=item["item_id"],
            body={"name": "Updated", "metadata": {"enabled": True}},
        )
    )
    assert status == 200
    assert body["item"]["name"] == "Updated"
    assert body["item"]["version"] == 2

    status, body = invoke(event("DELETE", "/items/{itemId}", item_id=item["item_id"]))
    assert status == 204
    assert body is None

    status, _ = invoke(event("GET", "/items/{itemId}", item_id=item["item_id"]))
    assert status == 404


def test_list_uses_owner_scoped_cursor_pagination(table: Any) -> None:
    create_item(name="First")
    create_item(name="Second")
    create_item(owner="user-2", name="Other user's item")

    status, first_page = invoke(event("GET", "/items", query={"limit": "1"}))
    assert status == 200
    assert first_page["count"] == 1
    assert len(first_page["items"]) == 1
    assert first_page.get("next_token")

    status, second_page = invoke(
        event("GET", "/items", query={"limit": "1", "next_token": first_page["next_token"]})
    )
    assert status == 200
    assert second_page["count"] == 1
    assert second_page["items"][0]["item_id"] != first_page["items"][0]["item_id"]

    status, body = invoke(
        event(
            "GET",
            "/items",
            owner="user-2",
            query={"limit": "1", "next_token": first_page["next_token"]},
        )
    )
    assert status == 400
    assert body["error"]["code"] == "INVALID_NEXT_TOKEN"


@pytest.mark.parametrize(
    ("api_event", "expected_code"),
    [
        (event("POST", "/items", body="{"), "INVALID_JSON"),
        (event("POST", "/items", body={"description": "missing name"}), "VALIDATION_ERROR"),
        (event("POST", "/items", body={"name": "x", "admin": True}), "VALIDATION_ERROR"),
        (event("GET", "/items", query={"limit": "0"}), "VALIDATION_ERROR"),
        (event("GET", "/items", query={"next_token": "not-base64"}), "INVALID_NEXT_TOKEN"),
    ],
)
def test_invalid_requests_are_rejected(
    table: Any, api_event: dict[str, Any], expected_code: str
) -> None:
    status, body = invoke(api_event)
    assert status == 400
    assert body["error"]["code"] == expected_code
    assert body["error"]["request_id"] == "request-123"


def test_unknown_route_returns_structured_404(table: Any) -> None:
    status, body = invoke(event("PATCH", "/items"))
    assert status == 404
    assert body["error"]["code"] == "NOT_FOUND"


def test_missing_update_and_delete_are_structured(table: Any) -> None:
    missing_id = "00000000-0000-4000-8000-000000000000"

    status, body = invoke(
        event("PUT", "/items/{itemId}", item_id=missing_id, body={"name": "Updated"})
    )
    assert status == 404
    assert body["error"]["code"] == "NOT_FOUND"

    status, body = invoke(event("DELETE", "/items/{itemId}", item_id=missing_id))
    assert status == 404
    assert body["error"]["code"] == "NOT_FOUND"


def test_aws_client_errors_are_not_exposed(monkeypatch: pytest.MonkeyPatch) -> None:
    class FailingTable:
        def query(self, **_: Any) -> None:
            raise ClientError(
                {"Error": {"Code": "ProvisionedThroughputExceededException"}},
                "Query",
            )

    monkeypatch.setattr(app, "_TABLE", FailingTable())
    status, body = invoke(event("GET", "/items"))
    assert status == 500
    assert body["error"]["code"] == "INTERNAL_ERROR"
    assert "ProvisionedThroughputExceededException" not in body["error"]["message"]


def test_unhandled_errors_are_not_exposed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app, "_TABLE", None)
    monkeypatch.setattr(app, "TABLE_NAME", "")
    status, body = invoke(event("GET", "/items"))
    assert status == 500
    assert body["error"]["code"] == "INTERNAL_ERROR"
    assert body["error"]["message"] == "The request could not be completed"


def test_request_completion_is_logged_as_json(table: Any, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("INFO")
    invoke(event("GET", "/items"))
    log_record = next(record for record in caplog.records if "request_completed" in record.message)
    entry = json.loads(log_record.message)
    assert entry["request_id"] == "request-123"
    assert entry["status_code"] == 200
    assert "duration_ms" in entry
    assert "owner_id" not in entry

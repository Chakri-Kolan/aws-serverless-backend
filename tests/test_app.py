import json
import unittest
from typing import Any

from src.app import TaskRepository, ValidationError, route


class FakeTable:
    def __init__(self) -> None:
        self.items: dict[str, dict[str, Any]] = {}

    def put_item(self, **kwargs: Any) -> dict[str, Any]:
        item = dict(kwargs["Item"])
        self.items[item["id"]] = item
        return {}

    def get_item(self, **kwargs: Any) -> dict[str, Any]:
        item = self.items.get(kwargs["Key"]["id"])
        return {"Item": dict(item)} if item else {}

    def scan(self, **kwargs: Any) -> dict[str, Any]:
        items = list(self.items.values())
        start_key = kwargs.get("ExclusiveStartKey")
        if start_key:
            start = next(i for i, item in enumerate(items) if item["id"] == start_key["id"]) + 1
            items = items[start:]
        limit = kwargs["Limit"]
        page = items[:limit]
        result: dict[str, Any] = {"Items": [dict(item) for item in page]}
        if len(items) > limit:
            result["LastEvaluatedKey"] = {"id": page[-1]["id"]}
        return result

    def update_item(self, **kwargs: Any) -> dict[str, Any]:
        item = self.items[kwargs["Key"]["id"]]
        for name_key, field in kwargs["ExpressionAttributeNames"].items():
            value_key = name_key.replace("#field", ":value")
            if value_key in kwargs["ExpressionAttributeValues"]:
                item[field] = kwargs["ExpressionAttributeValues"][value_key]
        item["updated_at"] = kwargs["ExpressionAttributeValues"][":updated_at"]
        return {"Attributes": dict(item)}

    def delete_item(self, **kwargs: Any) -> dict[str, Any]:
        del self.items[kwargs["Key"]["id"]]
        return {}


def event(method: str, path: str, body: dict[str, Any] | None = None, query=None):
    return {
        "rawPath": path,
        "requestContext": {"http": {"method": method}},
        "body": json.dumps(body) if body is not None else None,
        "queryStringParameters": query,
    }


class RouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.table = FakeTable()
        self.repository = TaskRepository(self.table)

    def test_health(self) -> None:
        response = route(event("GET", "/health"), self.repository)
        self.assertEqual(200, response.status_code)
        self.assertEqual("ok", response.body["status"])

    def test_crud_lifecycle(self) -> None:
        created = route(event("POST", "/tasks", {"title": "Ship API"}), self.repository)
        self.assertEqual(201, created.status_code)
        task = created.body["data"]
        self.assertEqual("todo", task["status"])

        fetched = route(event("GET", f"/tasks/{task['id']}"), self.repository)
        self.assertEqual("Ship API", fetched.body["data"]["title"])

        updated = route(
            event("PATCH", f"/tasks/{task['id']}", {"status": "done"}), self.repository
        )
        self.assertEqual("done", updated.body["data"]["status"])

        deleted = route(event("DELETE", f"/tasks/{task['id']}"), self.repository)
        self.assertEqual({"deleted": True, "id": task["id"]}, deleted.body)

    def test_cursor_pagination(self) -> None:
        for title in ("one", "two", "three"):
            self.repository.create(title)
        first = route(event("GET", "/tasks", query={"limit": "2"}), self.repository)
        self.assertEqual(2, len(first.body["data"]))
        self.assertIsNotNone(first.body["next_cursor"])
        second = route(
            event("GET", "/tasks", query={"limit": "2", "cursor": first.body["next_cursor"]}),
            self.repository,
        )
        self.assertEqual(1, len(second.body["data"]))
        self.assertIsNone(second.body["next_cursor"])

    def test_rejects_invalid_payloads(self) -> None:
        with self.assertRaisesRegex(ValidationError, "title"):
            route(event("POST", "/tasks", {"title": "  "}), self.repository)
        with self.assertRaisesRegex(ValidationError, "status"):
            route(event("PATCH", "/tasks/abc", {"status": "unknown"}), self.repository)


if __name__ == "__main__":
    unittest.main()

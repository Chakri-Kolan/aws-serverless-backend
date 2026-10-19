# Architecture

```text
Client
  |
  | Authorization: <Cognito ID token>
  v
API Gateway REST API -- Cognito user-pool authorizer
  |
  v
Lambda (Python 3.12)
  |
  v
DynamoDB table (owner_id, item_id)
```

`GET /health` deliberately bypasses the authorizer. Every `/items` method is
protected by the API Gateway Cognito authorizer. API Gateway verifies the token
before Lambda invocation and passes validated claims to Lambda. The handler
requires the `sub` claim and uses it as the DynamoDB partition key.

## Data model and authorization boundary

| Attribute | Purpose |
| --- | --- |
| `owner_id` | Partition key derived only from the validated Cognito `sub` claim |
| `item_id` | Sort key generated as a UUID by the service |
| `name`, `description`, `metadata` | Validated client data |
| `created_at`, `updated_at`, `version` | Service-managed concurrency and audit fields |

Reads, queries, updates, and deletes always include `owner_id`. An item ID alone
is insufficient to cross the user boundary. Pagination uses DynamoDB `Query`,
never table `Scan`, and rejects a cursor issued for a different owner.

## Infrastructure and security

- One stage-scoped API, Lambda, DynamoDB table, Cognito user pool, and public app client.
- Lambda IAM permits only `GetItem`, `PutItem`, `DeleteItem`, and `Query` on the one table.
- DynamoDB uses on-demand billing, server-side encryption, point-in-time recovery, and retain policies.
- Cognito enforces a strong password policy, email verification, optional authenticator-app MFA, token revocation, and one-hour access/ID tokens.
- API Gateway validates create/update JSON schemas; Lambda repeats semantic validation because it is the final trust boundary.
- Logs are JSON records containing request metadata and status, without tokens, request bodies, or user identifiers.
- Expected errors use stable JSON error codes; unexpected exceptions are logged and returned as generic `500` responses.

## Operational behavior

Resources are named by the SAM `Stage` parameter, so `dev`, `test`, and `prod` do not
share state. The DynamoDB table and Cognito user pool are retained if the stack
is removed; delete retained data only through an explicitly reviewed process.

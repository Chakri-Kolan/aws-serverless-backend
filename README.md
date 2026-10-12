# AWS Serverless Backend

A deployable, stage-isolated CRUD API using API Gateway, Lambda, Cognito, and DynamoDB.

## Implemented features

- Public `GET /health` endpoint
- Cognito-authenticated create, read, update, delete, and list endpoints
- Per-user authorization using the validated Cognito `sub` as the DynamoDB partition key
- API Gateway JSON-schema validation plus Lambda semantic validation
- Cursor pagination backed by DynamoDB `Query` (no table scans)
- Optimistic version checks for concurrent updates
- Structured JSON logs and stable error responses with request IDs
- Least-privilege Lambda access to one DynamoDB table
- Encrypted, point-in-time-recoverable DynamoDB storage with retained data resources
- Unit/integration-style tests against a mocked DynamoDB service
- CI lint, coverage, tests, and AWS SAM/CloudFormation validation and build

## API

| Method | Path | Authentication | Description |
| --- | --- | --- | --- |
| `GET` | `/health` | Public | Liveness response |
| `POST` | `/items` | Cognito ID token | Create an item |
| `GET` | `/items` | Cognito ID token | List caller-owned items |
| `GET` | `/items/{itemId}` | Cognito ID token | Get a caller-owned item |
| `PUT` | `/items/{itemId}` | Cognito ID token | Partially update a caller-owned item |
| `DELETE` | `/items/{itemId}` | Cognito ID token | Delete a caller-owned item |

List parameters:

- `limit`: integer from 1 to 100; default 25
- `next_token`: opaque cursor returned by the preceding page

Create request:

```json
{
  "name": "Example",
  "description": "Optional text",
  "metadata": {"source": "demo"}
}
```

Errors have a stable shape:

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "name is required",
    "request_id": "api-gateway-request-id"
  }
}
```

## Architecture

```text
Client -> API Gateway REST API -> Lambda -> DynamoDB
                 |
                 +-> Cognito user-pool authorizer (all /items methods)
```

See [architecture](docs/architecture.md) and [authentication](docs/authentication.md) for trust boundaries, token storage, and lifecycle guidance.

## Local development

Prerequisites: Python 3.12 and the AWS SAM CLI.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --requirement requirements-dev.txt
ruff check src tests
pytest
sam validate --lint --template-file template.yaml
sam build --template-file template.yaml
```

Tests, `sam validate`, and `sam build` do not deploy resources.

## Deployment

Deployment creates billable AWS resources. Review the packaged CloudFormation and
estimated cost before running these commands. Use a dedicated AWS account/stage and
short-lived AWS SSO credentials; never store access keys in this repository.

```bash
aws sso login --profile my-sandbox
sam validate --lint --template-file template.yaml
sam build --template-file template.yaml

# Deploy only after reviewing .aws-sam/build/template.yaml
AWS_PROFILE=my-sandbox sam deploy \
  --template-file .aws-sam/build/template.yaml \
  --stack-name aws-serverless-backend-dev \
  --region us-east-1 \
  --capabilities CAPABILITY_IAM \
  --parameter-overrides Stage=dev \
  --resolve-s3
```

The deployment outputs the API URL, Cognito user pool ID, public app client ID,
and DynamoDB table name. Create/confirm users through an approved admin workflow,
then call protected endpoints with a Cognito ID token as documented in
[authentication](docs/authentication.md).

## Removal and retained data

Deleting the CloudFormation stack removes the API and Lambda but intentionally
retains the DynamoDB table and Cognito user pool. Retained resources can continue
to incur cost. Delete them only after explicit data-retention review and backup verification.

## Security notes

- `/health` is the only public application route.
- Lambda does not validate JWT signatures itself; API Gateway's Cognito authorizer performs cryptographic token validation before invoking protected routes.
- Lambda still requires the validated `sub` claim and does not trust user-supplied owner IDs.
- CORS defaults to `*` for this API example. Restrict origins before a browser production deployment.
- Cognito MFA is not forced by this baseline. Enable an MFA policy appropriate to the application before production use.
- CI packages but does not deploy. Add a separately reviewed OIDC deployment role/workflow if automated deployment is required.

## Project layout

```text
.github/workflows/ci.yml   CI lint, tests, and packaging
docs/                      Architecture and authentication guidance
src/app.py                 Lambda routing, validation, persistence, and errors
src/requirements.txt       Pinned Lambda runtime dependencies
template.yaml              AWS SAM/CloudFormation infrastructure
tests/test_app.py          API and DynamoDB behavior tests
```

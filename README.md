# AWS Serverless Task API

[![CI](https://github.com/Chakri-Kolan/aws-serverless-backend/actions/workflows/ci.yml/badge.svg)](https://github.com/Chakri-Kolan/aws-serverless-backend/actions/workflows/ci.yml)

A production-style REST API built with Python, AWS Lambda, API Gateway HTTP API, and DynamoDB. The project demonstrates request validation, conditional writes, cursor pagination, least-privilege IAM, infrastructure as code, and credential-free unit testing.

## Architecture

```text
Client -> API Gateway -> Lambda -> DynamoDB
                           |
                           +----> CloudWatch Logs
```

See [docs/architecture.md](docs/architecture.md) for design decisions and production extensions.

## API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Service health |
| `POST` | `/tasks` | Create a task |
| `GET` | `/tasks?limit=20&cursor=...` | Paginated task list |
| `GET` | `/tasks/{id}` | Retrieve a task |
| `PATCH` | `/tasks/{id}` | Update title, description, or status |
| `DELETE` | `/tasks/{id}` | Delete a task |

Valid task statuses are `todo`, `in_progress`, and `done`.

```bash
curl -X POST "$API_URL/tasks" \
  -H 'content-type: application/json' \
  -d '{"title":"Ship portfolio API","description":"Deploy and verify"}'
```

## Local validation

Requirements: Python 3.11+ and the AWS SAM CLI.

```bash
python -m unittest discover -v
sam validate --lint
sam build
```

Unit tests use an in-memory DynamoDB test double, so they need no AWS account or credentials.

## Deploy

Authenticate the AWS CLI, then run:

```bash
sam build
sam deploy --guided --stack-name aws-serverless-backend-dev
```

SAM prints the deployed API URL. To avoid charges after evaluation:

```bash
aws cloudformation delete-stack --stack-name aws-serverless-backend-dev
```

The DynamoDB table has a retention policy to protect data, so remove it explicitly only when data deletion is intended.

## Engineering highlights

- AWS resources defined in one repeatable AWS SAM/CloudFormation stack
- DynamoDB encryption, point-in-time recovery, and pay-per-request capacity
- Least-privilege function role scoped to the generated table
- Structured API errors and request correlation IDs
- Base64 URL-safe continuation cursors
- CI linting, unit tests, and deploy-package validation

## Cost

The architecture is scale-to-zero and typically remains within AWS free-tier allowances for portfolio traffic. Actual charges depend on usage and account eligibility.

## License

MIT

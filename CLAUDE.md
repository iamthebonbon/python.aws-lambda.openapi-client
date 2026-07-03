# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

This is an AWS SAM (Serverless Application Model) Python 3.12 Lambda application (stack name `openapi-client`). It defines two functions behind a single API-key-protected API Gateway stage: `MainFunction` (`GET /main`) and `AgentFunction` (`POST /agent`), an OpenAI-backed tool-calling agent. Despite the repo directory name (`python.aws-lambda.openapi-client`), no OpenAPI client code exists yet.

## Commands

Build (requires Docker):
```bash
sam build --use-container
```

Invoke a single function locally with a test event:
```bash
sam local invoke MainFunction --event events/event.json
sam local invoke AgentFunction --event events/agent_event.json
```

Run the API locally (port 3000):
```bash
sam local start-api
curl http://localhost:3000/main
curl -X POST http://localhost:3000/agent -d '{"prompt": "hello"}'
```

Deploy:
```bash
sam deploy --guided   # first time, prompts for stack config, saves to samconfig.toml
sam deploy             # subsequent deploys
```

Validate/lint the template:
```bash
sam validate --lint
```

Tests:
```bash
pip install -r tests/requirements.txt --user

# unit tests
python -m pytest tests/unit -v

# single test
python -m pytest tests/unit/test_handler.py::test_lambda_handler -v

# integration tests — require a deployed stack
AWS_SAM_STACK_NAME="openapi-client" python -m pytest tests/integration -v
```

Tail deployed function logs:
```bash
sam logs -n MainFunction --stack-name "openapi-client" --tail
```

Delete the deployed stack:
```bash
sam delete --stack-name "openapi-client"
```

## Architecture

- `template.yaml` is the single source of truth for AWS resources (SAM/CloudFormation). It defines an explicit API Gateway REST API, `OpenApiClientApi` (stage `Prod`), with `Auth.ApiKeyRequired: true` set once at the API level — this single top-level Auth block covers every method on the stage (neither the `Main` nor `Agent` `Api` events define a per-event `Auth` override), so both endpoints below already require the same API key. Adding an endpoint requires both new Lambda code and a matching `Events` entry in `template.yaml`.
- `MainFunction` (`app/app.py`, handler `app.lambda_handler`) — `GET /main`.
- `AgentFunction` (`agent/app.py`, handler `app.lambda_handler`) — `POST /agent`; an OpenAI tool-calling agent, reads `OPENAI_API_KEY` from SSM Parameter Store (`/openai-client/openai-api-key`).
- `ApiKey` / `UsagePlan` / `UsagePlanKey` — the API key required by `OpenApiClientApi`, with throttle (10 rps / burst 20) and a 10,000/month quota; fetch the key value with `aws apigateway get-api-key --api-key <ApiKeyId> --include-value`.
- `app/`, `agent/` — function code, each with its own `requirements.txt`; SAM packages dependencies per-function, not shared across functions.
- `events/` — sample API Gateway Lambda-proxy event JSON used with `sam local invoke` (`event.json` for `MainFunction`, `agent_event.json` for `AgentFunction`).
- `tests/unit/` — pure unit tests that import `app.app` / `agent.app` directly, with no AWS calls.
- `tests/integration/` — tests that call the deployed API Gateway endpoint, resolving the URL via `boto3` CloudFormation stack outputs (`MainApi`, `ApiKeyId`); require the `AWS_SAM_STACK_NAME` env var and a real deployed stack.
- `samconfig.toml` — default CLI parameters (stack name, build cache/parallel, deploy `CAPABILITY_IAM`, sync `--watch`, etc.) so `sam build`/`deploy`/`sync` work without extra flags.

# Instructions
- Follow to operational, memory and cost efficiency
- Don't do cognitive complexity
- Do self-documented code
- Be concise

# Temperature
0.2

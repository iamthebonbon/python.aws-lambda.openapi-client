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

- `template.yaml` is the root source of truth for AWS resources (SAM/CloudFormation). It defines a plain `AWS::ApiGateway::RestApi`, `OpenApiClientApi`, plus a hand-managed `ApiDeployment`/`ApiStage` (stage `Prod`) — because both functions' routes live in nested stacks that the root template's SAM transform can't see, `OpenApiClientApi` can't be a SAM-managed `AWS::Serverless::Api` (its implicit Deployment/Stage only covers routes defined in the same template). Consequently there's no template-level `Auth` block anymore: each `Api` event sets `Auth.ApiKeyRequired: true` individually. `ApiDeployment` depends on both nested stacks (`MainFunctionStack`, `AgentFunctionStack`) so its snapshot always includes both routes — a nested stack's completion is atomic and guarantees its method resource exists first, so this `DependsOn` has no ordering ambiguity. Adding an endpoint requires a new nested stack (or a new route in an existing one) plus adding that stack to `ApiDeployment`'s `DependsOn`.
- `MainFunction` — nested stack (`templates/main.yaml`, `AWS::Serverless::Application`), code at `app/app.py` (handler `app.lambda_handler`) — `GET /main`. Receives the shared `RestApiId` as a parameter from the root template and outputs `MainFunctionArn`.
- `AgentFunction` — nested stack (`templates/agent.yaml`, `AWS::Serverless::Application`), code at `agent/app.py` (handler `app.lambda_handler`) — `POST /agent`; an OpenAI tool-calling agent, reads `OPENAI_API_KEY` from SSM Parameter Store (`/openai-client/openai-api-key`). Receives the shared `RestApiId` as a parameter from the root template and outputs `AgentFunctionArn`.
- `ApiKey` / `UsagePlan` / `UsagePlanKey` — the API key required by `OpenApiClientApi`, with throttle (10 rps / burst 20) and a 10,000/month quota; fetch the key value with `aws apigateway get-api-key --api-key <ApiKeyId> --include-value`. Both depend on `ApiStage` (the hand-managed stage), not an auto-generated one.
- `app/`, `agent/` — function code, each with its own `requirements.txt`; SAM packages dependencies per-function, not shared across functions.
- `events/` — sample API Gateway Lambda-proxy event JSON used with `sam local invoke` (`event.json` for `MainFunction`, `agent_event.json` for `AgentFunction`).
- `tests/unit/` — pure unit tests that import `app.app` / `agent.app` directly, with no AWS calls.
- `tests/integration/` — tests that call the deployed API Gateway endpoint, resolving the URL via `boto3` CloudFormation stack outputs (`MainApi`, `ApiKeyId`); require the `AWS_SAM_STACK_NAME` env var and a real deployed stack.
- `samconfig.toml` — default CLI parameters (stack name, build cache/parallel, deploy `CAPABILITY_IAM CAPABILITY_AUTO_EXPAND` — `CAPABILITY_AUTO_EXPAND` is required because the nested stack applications are themselves SAM templates needing macro expansion —, sync `--watch`, etc.) so `sam build`/`deploy`/`sync` work without extra flags.

# Instructions
- Follow to operational, memory and cost efficiency
- Don't do cognitive complexity
- Do self-documented code
- Be concise

# Temperature
0.2

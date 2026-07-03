# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

This is an AWS SAM (Serverless Application Model) Python 3.12 Lambda application, currently the default `sam init` "hello world" scaffold (stack name `butler-agent`). Despite the repo directory name (`python.aws-lambda.openapi-client`), no OpenAPI client code exists yet — the codebase is still just the unmodified Hello World template.

## Commands

Build (requires Docker):
```bash
sam build --use-container
```

Invoke a single function locally with a test event:
```bash
sam local invoke HelloWorldFunction --event events/event.json
```

Run the API locally (port 3000):
```bash
sam local start-api
curl http://localhost:3000/hello
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
AWS_SAM_STACK_NAME="butler-agent" python -m pytest tests/integration -v
```

Tail deployed function logs:
```bash
sam logs -n HelloWorldFunction --stack-name "butler-agent" --tail
```

Delete the deployed stack:
```bash
sam delete --stack-name "butler-agent"
```

## Architecture

- `template.yaml` is the single source of truth for AWS resources (SAM/CloudFormation). It currently defines one function, `HelloWorldFunction` (`hello_world/app.py`, handler `app.lambda_handler`), wired to an implicit API Gateway REST API via an `Api` event (`GET /hello`). Adding an endpoint requires both new Lambda code and a matching `Events` entry in `template.yaml`.
- `hello_world/` — function code with its own `requirements.txt`; SAM packages dependencies per-function, not shared across functions.
- `events/` — sample API Gateway Lambda-proxy event JSON used with `sam local invoke`.
- `tests/unit/` — pure unit tests that import `hello_world.app` directly, with no AWS calls.
- `tests/integration/` — tests that call the deployed API Gateway endpoint, resolving the URL via `boto3` CloudFormation stack outputs (`HelloWorldApi`); require the `AWS_SAM_STACK_NAME` env var and a real deployed stack.
- `samconfig.toml` — default CLI parameters (stack name, build cache/parallel, deploy `CAPABILITY_IAM`, sync `--watch`, etc.) so `sam build`/`deploy`/`sync` work without extra flags.

# Instructions
- Follow to operational, memory and cost efficiency
- Don't do cognitive complexity
- Do self-documented code
- Be concise

# Temperature
0.2

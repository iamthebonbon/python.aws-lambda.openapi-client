import json
import math
import os
import logging
import uuid

logger = logging.getLogger()
logger.setLevel(logging.INFO)

import boto3
from openai import OpenAI

SYSTEM_PROMPT = """
You are a helpful assistant, your goal is to help user.
You have an access to the wardrobe and weather through tools, you can also clean dirty items.
Use remember_fact to save useful facts about the user for later, and recall_facts to search
those facts semantically when they might help with the current request.
Don't ask for permission to do the task, just do everything you can to help user.
If you see dirty cloth then pass it to wash.
"""

MAX_ITERATIONS = 5
MODEL_NAME = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"

HISTORY_LIMIT = 10
SUMMARY_PROMPT = "Summarize this conversation excerpt in 2-3 sentences, keeping any facts relevant to future turns."

# Long-term semantic memory, stored directly in S3: each fact is its own object
# (text + embedding), so S3 is the memory itself rather than a snapshot of a
# local database. Unlike archiving a shared sqlite file, this makes concurrent
# writes across containers safe (each write is an independent PutObject, not a
# read-modify-write of one file). recall_facts loads every fact into memory,
# per-request, and ranks them with a plain cosine-similarity scan — no vector
# database needed at this project's memory scale.
MEMORY_BUCKET = os.getenv("MEMORY_BUCKET")
MEMORY_PREFIX = os.getenv("MEMORY_PREFIX", "facts")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
_s3 = boto3.client("s3")


def _embed(text):
    return client.embeddings.create(model=EMBEDDING_MODEL, input=[text]).data[0].embedding


def _fact_key(fact_id):
    return f"{MEMORY_PREFIX.rstrip('/')}/{fact_id}.json"


def remember_fact(text):
    """Embed a fact and write it straight to S3 as its own object."""
    fact = {"text": text, "embedding": _embed(text)}
    _s3.put_object(
        Bucket=MEMORY_BUCKET,
        Key=_fact_key(str(uuid.uuid4())),
        Body=json.dumps(fact).encode("utf-8"),
        ContentType="application/json",
    )
    return {"stored": text}


def _load_facts():
    """Fetch every fact object from S3. Memory is small at this project's scale, so a full scan is cheap."""
    facts = []
    paginator = _s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=MEMORY_BUCKET, Prefix=f"{MEMORY_PREFIX.rstrip('/')}/"):
        for obj in page.get("Contents", []):
            body = _s3.get_object(Bucket=MEMORY_BUCKET, Key=obj["Key"])["Body"].read()
            facts.append(json.loads(body))
    return facts


def _cosine_similarity(a, b):
    """Cosine similarity between two equal-length embedding vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def recall_facts(query, n_results=3):
    """Load facts from S3 and return the texts closest to a query by cosine similarity."""
    facts = _load_facts()
    if not facts:
        return {"matches": []}

    query_embedding = _embed(query)
    ranked = sorted(facts, key=lambda fact: _cosine_similarity(fact["embedding"], query_embedding), reverse=True)
    return {"matches": [fact["text"] for fact in ranked[:n_results]]}


# Seed data for the wardrobe tools. A fresh copy is built for every request
# (see build_tools) so nothing persists between invocations.
WARDROBE_SEED = [
    {"id": 1, "name": "blue jeans", "clean": True},
    {"id": 2, "name": "white t-shirt", "clean": False},
    {"id": 3, "name": "rain jacket", "clean": True},
]

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_wardrobe",
            "description": "List every item in the user's wardrobe, including whether it is clean or dirty.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather forecast for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "City to get the weather for"}},
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "wash_item",
            "description": "Wash a dirty wardrobe item, marking it clean.",
            "parameters": {
                "type": "object",
                "properties": {"item_id": {"type": "integer", "description": "id of the wardrobe item to wash"}},
                "required": ["item_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remember_fact",
            "description": "Save a fact about the user or conversation to long-term semantic memory for future recall.",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string", "description": "The fact to remember"}},
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recall_facts",
            "description": "Search long-term semantic memory for facts related to a query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search for"},
                    "n_results": {"type": "integer", "description": "Max number of facts to return (default 3)"},
                },
                "required": ["query"],
            },
        },
    },
]


def build_tools():
    """Build a fresh hardcoded tool map, scoped to this single request only."""
    wardrobe = [dict(item) for item in WARDROBE_SEED]

    def get_wardrobe():
        return wardrobe

    def get_weather(city):
        # Mocked forecast; this POC has no real weather API integration.
        return {"city": city, "forecast": "sunny", "temperature_celsius": 22}

    def wash_item(item_id):
        for item in wardrobe:
            if item["id"] == item_id:
                item["clean"] = True
                return item
        return {"error": f"item {item_id} not found"}

    return {
        "get_wardrobe": get_wardrobe,
        "get_weather": get_weather,
        "wash_item": wash_item,
        "remember_fact": remember_fact,
        "recall_facts": recall_facts,
    }


def call_tool(tools, tool_call):
    function = tools[tool_call.function.name]
    arguments = json.loads(tool_call.function.arguments or "{}")
    return function(**arguments)


def summarize_history(history):
    """Collapse the oldest half of a long history into one summary message, keeping the rest intact."""
    if len(history) <= HISTORY_LIMIT:
        return history

    has_system = history[0]["role"] == "system"
    head, rest = ([history[0]], history[1:]) if has_system else ([], history)

    midpoint = len(rest) // 2
    while midpoint > 0 and rest[midpoint]["role"] == "tool":
        midpoint -= 1
    old, recent = rest[:midpoint], rest[midpoint:]
    transcript = "\n".join(f"{m['role']}: {m.get('content', '')}" for m in old)

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SUMMARY_PROMPT},
            {"role": "user", "content": transcript},
        ],
    )
    summary = {"role": "assistant", "content": f"[Summary of earlier conversation] {response.choices[0].message.content}"}

    return head + [summary] + recent


def run_agent(messages, tools):
    """Run the tool-calling loop for up to MAX_ITERATIONS, returning the updated messages."""
    for _ in range(MAX_ITERATIONS):
        response = client.chat.completions.create(model=MODEL_NAME, messages=messages, tools=TOOL_SCHEMAS)
        message = response.choices[0].message
        messages.append(message.model_dump(exclude_none=True))

        if not message.tool_calls:
            break

        for tool_call in message.tool_calls:
            result = call_tool(tools, tool_call)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result),
            })

    return messages


def lambda_handler(event, context):
    """Run the agent for a single request.

    Expects a JSON body of ``{"prompt": str, "history": list | None}``. History
    is passed back in the response so the caller can resend it on the next
    request; nothing is persisted between invocations.
    """
    try:
        body = json.loads(event.get("body") or "{}")
        prompt = body.get("prompt", "")
        history = body.get("history") or [{"role": "system", "content": SYSTEM_PROMPT}]

        history = summarize_history(history)
        history.append({"role": "user", "content": prompt})
        history = run_agent(history, build_tools())

        result = {
            "statusCode": 200,
            "body": json.dumps({"reply": history[-1]["content"], "history": history}),
        }
        logger.info("Returning: %s", result)
        return result
    except Exception as e:
        logger.exception("Handler failed")  # logs full traceback
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}

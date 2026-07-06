import json
import os
import logging
import tarfile
import tempfile
import uuid

logger = logging.getLogger()
logger.setLevel(logging.INFO)

import boto3
import chromadb
from botocore.exceptions import ClientError
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

# Long-term semantic memory: a Chroma collection persisted to S3. Chroma only
# writes to local disk, so the store lives in /tmp (reused for free across warm
# invocations of the same container) and is archived to/from S3 so a cold start,
# or a different concurrent container, can pick up what was last saved.
MEMORY_BUCKET = os.getenv("MEMORY_BUCKET")
MEMORY_PREFIX = os.getenv("MEMORY_PREFIX", "")
MEMORY_LOCAL_DIR = "/tmp/chroma"
MEMORY_COLLECTION_NAME = "agent-memory"
MEMORY_ARCHIVE_KEY = f"{MEMORY_PREFIX.rstrip('/')}/memory.tar.gz" if MEMORY_PREFIX else "memory.tar.gz"

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
_s3 = boto3.client("s3")
_memory_collection = None  # cached across warm invocations, alongside the /tmp files it wraps


def _download_memory_snapshot():
    """Restore the persisted Chroma store from S3 into /tmp, once per cold start."""
    if os.path.isdir(MEMORY_LOCAL_DIR):
        return  # container is warm; the on-disk store is already there

    os.makedirs(MEMORY_LOCAL_DIR, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".tar.gz") as archive:
        try:
            _s3.download_file(MEMORY_BUCKET, MEMORY_ARCHIVE_KEY, archive.name)
        except ClientError as e:
            if e.response["Error"]["Code"] not in ("404", "NoSuchKey"):
                raise
            return  # nothing saved yet; start with an empty collection
        with tarfile.open(archive.name) as tar:
            tar.extractall(MEMORY_LOCAL_DIR)


def _upload_memory_snapshot():
    """Archive the current Chroma store back to S3 so it survives the next cold start."""
    with tempfile.NamedTemporaryFile(suffix=".tar.gz") as archive:
        with tarfile.open(archive.name, "w:gz") as tar:
            tar.add(MEMORY_LOCAL_DIR, arcname=".")
        _s3.upload_file(archive.name, MEMORY_BUCKET, MEMORY_ARCHIVE_KEY)


def get_memory_collection():
    """Return the Chroma collection backing semantic memory, restoring it from S3 on first use."""
    global _memory_collection
    if _memory_collection is None:
        _download_memory_snapshot()
        chroma_client = chromadb.PersistentClient(path=MEMORY_LOCAL_DIR)
        _memory_collection = chroma_client.get_or_create_collection(MEMORY_COLLECTION_NAME)
    return _memory_collection


def _embed(text):
    return client.embeddings.create(model=EMBEDDING_MODEL, input=[text]).data[0].embedding


def remember_fact(text):
    """Store a fact in semantic memory and persist the updated store to S3."""
    collection = get_memory_collection()
    collection.add(ids=[str(uuid.uuid4())], embeddings=[_embed(text)], documents=[text])
    _upload_memory_snapshot()
    return {"stored": text}


def recall_facts(query, n_results=3):
    """Return the remembered facts most semantically similar to a query."""
    collection = get_memory_collection()
    if collection.count() == 0:
        return {"matches": []}
    results = collection.query(query_embeddings=[_embed(query)], n_results=min(n_results, collection.count()))
    return {"matches": results["documents"][0]}


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

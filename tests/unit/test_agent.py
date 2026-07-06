import json
from types import SimpleNamespace
from unittest.mock import patch

from agent import app


def _completion(content=None, tool_calls=None):
    dumped = {"role": "assistant", "content": content}
    if tool_calls:
        dumped["tool_calls"] = [
            {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
            for tc in tool_calls
        ]
    message = SimpleNamespace(content=content, tool_calls=tool_calls, model_dump=lambda exclude_none=True: dumped)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _tool_call(call_id, name, arguments):
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=json.dumps(arguments)))


def test_lambda_handler_replies_without_tool_calls():
    with patch.object(app.client.chat.completions, "create", return_value=_completion(content="Hi there!")) as mock_create:
        event = {"body": json.dumps({"prompt": "hello"})}
        response = app.lambda_handler(event, None)

    body = json.loads(response["body"])
    assert response["statusCode"] == 200
    assert body["reply"] == "Hi there!"
    assert body["history"][0] == {"role": "system", "content": app.SYSTEM_PROMPT}
    assert body["history"][1] == {"role": "user", "content": "hello"}
    mock_create.assert_called_once()


def test_lambda_handler_executes_tool_call_before_replying():
    tool_call = _tool_call("call_1", "get_wardrobe", {})
    responses = [_completion(tool_calls=[tool_call]), _completion(content="Your rain jacket is clean.")]

    with patch.object(app.client.chat.completions, "create", side_effect=responses):
        event = {"body": json.dumps({"prompt": "what's clean?"})}
        response = app.lambda_handler(event, None)

    body = json.loads(response["body"])
    tool_messages = [m for m in body["history"] if m["role"] == "tool"]
    assert body["reply"] == "Your rain jacket is clean."
    assert len(tool_messages) == 1
    assert tool_messages[0]["tool_call_id"] == "call_1"
    assert json.loads(tool_messages[0]["content"])[1]["name"] == "white t-shirt"


def test_lambda_handler_stops_after_max_iterations():
    tool_call = _tool_call("call_1", "get_wardrobe", {})
    responses = [_completion(tool_calls=[tool_call]) for _ in range(app.MAX_ITERATIONS)]

    with patch.object(app.client.chat.completions, "create", side_effect=responses) as mock_create:
        event = {"body": json.dumps({"prompt": "keep checking my wardrobe"})}
        app.lambda_handler(event, None)

    assert mock_create.call_count == app.MAX_ITERATIONS


def test_lambda_handler_reuses_supplied_history_without_duplicate_system_prompt():
    history = [
        {"role": "system", "content": app.SYSTEM_PROMPT},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello!"},
    ]

    with patch.object(app.client.chat.completions, "create", return_value=_completion(content="sure, washing it")):
        event = {"body": json.dumps({"prompt": "wash my shirt", "history": history})}
        response = app.lambda_handler(event, None)

    body = json.loads(response["body"])
    system_messages = [m for m in body["history"] if m["role"] == "system"]
    assert len(system_messages) == 1
    assert body["history"][3] == {"role": "user", "content": "wash my shirt"}


def test_summarize_history_leaves_short_history_untouched():
    history = [{"role": "system", "content": app.SYSTEM_PROMPT}] + [
        {"role": "user", "content": f"message {i}"} for i in range(app.HISTORY_LIMIT - 1)
    ]

    result = app.summarize_history(history)

    assert result == history


def test_summarize_history_collapses_oldest_half_when_over_limit():
    history = [{"role": "system", "content": app.SYSTEM_PROMPT}] + [
        {"role": "user", "content": f"message {i}"} for i in range(app.HISTORY_LIMIT + 1)
    ]

    with patch.object(app.client.chat.completions, "create", return_value=_completion(content="a short summary")) as mock_create:
        result = app.summarize_history(history)

    mock_create.assert_called_once()
    assert result[0] == {"role": "system", "content": app.SYSTEM_PROMPT}
    assert result[1] == {"role": "assistant", "content": "[Summary of earlier conversation] a short summary"}
    assert result[2:] == history[1 + (len(history) - 1) // 2:]


def test_wash_item_marks_matching_item_clean():
    tools = app.build_tools()

    result = tools["wash_item"](item_id=2)

    assert result == {"id": 2, "name": "white t-shirt", "clean": True}
    assert tools["get_wardrobe"]()[1]["clean"] is True


def test_build_tools_returns_fresh_state_each_call():
    tools_a = app.build_tools()
    tools_a["wash_item"](item_id=2)

    tools_b = app.build_tools()

    assert tools_b["get_wardrobe"]()[1]["clean"] is False

import json
from unittest.mock import MagicMock

import pytest

from agent import app


@pytest.fixture(autouse=True)
def isolate_conversation_table(monkeypatch):
    """Point conversation storage at a scratch table so tests don't depend on real env vars."""
    monkeypatch.setattr(app, "CONVERSATION_TABLE", "test-conversations")


def test_save_conversation_puts_a_single_item_keyed_by_id(monkeypatch):
    mock_dynamodb = MagicMock()
    monkeypatch.setattr(app, "_dynamodb", mock_dynamodb)
    history = [{"role": "system", "content": app.SYSTEM_PROMPT}, {"role": "user", "content": "hi"}]

    app._save_conversation("abc-123", history)

    _, kwargs = mock_dynamodb.put_item.call_args
    assert kwargs["TableName"] == "test-conversations"
    item = kwargs["Item"]
    assert item["conversation_id"] == {"S": "abc-123"}
    assert json.loads(item["history"]["S"]) == history


def test_load_conversation_returns_stored_history(monkeypatch):
    history = [{"role": "system", "content": app.SYSTEM_PROMPT}, {"role": "user", "content": "hi"}]
    mock_dynamodb = MagicMock()
    mock_dynamodb.get_item.return_value = {
        "Item": {"conversation_id": {"S": "abc-123"}, "history": {"S": json.dumps(history)}}
    }
    monkeypatch.setattr(app, "_dynamodb", mock_dynamodb)

    result = app._load_conversation("abc-123")

    mock_dynamodb.get_item.assert_called_once_with(
        TableName="test-conversations", Key={"conversation_id": {"S": "abc-123"}}
    )
    assert result == history


def test_load_conversation_returns_none_when_not_found(monkeypatch):
    mock_dynamodb = MagicMock()
    mock_dynamodb.get_item.return_value = {}
    monkeypatch.setattr(app, "_dynamodb", mock_dynamodb)

    assert app._load_conversation("missing") is None

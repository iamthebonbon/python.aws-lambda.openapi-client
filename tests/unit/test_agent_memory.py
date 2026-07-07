import json
from unittest.mock import MagicMock

import pytest

from agent import app


@pytest.fixture(autouse=True)
def isolate_memory_state(monkeypatch):
    """Point memory at a scratch table so tests don't depend on real env vars."""
    monkeypatch.setattr(app, "MEMORY_TABLE", "test-table")


def test_remember_fact_embeds_and_puts_a_standalone_item(monkeypatch):
    monkeypatch.setattr(app, "_embed", MagicMock(return_value=[0.1, 0.2]))
    mock_dynamodb = MagicMock()
    monkeypatch.setattr(app, "_dynamodb", mock_dynamodb)

    result = app.remember_fact("the user prefers cold showers")

    assert result == {"stored": "the user prefers cold showers"}
    _, kwargs = mock_dynamodb.put_item.call_args
    assert kwargs["TableName"] == "test-table"
    item = kwargs["Item"]
    assert "fact_id" in item and item["fact_id"]["S"]
    assert item["text"] == {"S": "the user prefers cold showers"}
    assert json.loads(item["embedding"]["S"]) == [0.1, 0.2]


def _paginator_with_pages(pages):
    paginator = MagicMock()
    paginator.paginate.return_value = pages
    return paginator


def test_load_facts_scans_every_item_in_the_table(monkeypatch):
    mock_dynamodb = MagicMock()
    mock_dynamodb.get_paginator.return_value = _paginator_with_pages(
        [
            {
                "Items": [
                    {"fact_id": {"S": "a"}, "text": {"S": "fact a"}, "embedding": {"S": json.dumps([0.1])}},
                    {"fact_id": {"S": "b"}, "text": {"S": "fact b"}, "embedding": {"S": json.dumps([0.2])}},
                ]
            }
        ]
    )
    monkeypatch.setattr(app, "_dynamodb", mock_dynamodb)

    facts = app._load_facts()

    mock_dynamodb.get_paginator.assert_called_once_with("scan")
    assert facts == [
        {"text": "fact a", "embedding": [0.1]},
        {"text": "fact b", "embedding": [0.2]},
    ]


def test_load_facts_returns_empty_list_when_nothing_stored(monkeypatch):
    mock_dynamodb = MagicMock()
    mock_dynamodb.get_paginator.return_value = _paginator_with_pages([{"Items": []}])
    monkeypatch.setattr(app, "_dynamodb", mock_dynamodb)

    assert app._load_facts() == []


def test_recall_facts_short_circuits_when_no_facts_are_stored(monkeypatch):
    monkeypatch.setattr(app, "_load_facts", MagicMock(return_value=[]))

    result = app.recall_facts("what does the user like?")

    assert result == {"matches": []}


def test_recall_facts_ranks_matches_by_cosine_similarity_to_the_query(monkeypatch):
    facts = [
        {"text": "fact one", "embedding": [1.0, 0.0]},  # parallel to query -> similarity 1.0
        {"text": "fact two", "embedding": [0.0, 1.0]},  # orthogonal to query -> similarity 0.0
        {"text": "fact three", "embedding": [1.0, 1.0]},  # 45 degrees off -> similarity ~0.707
    ]
    monkeypatch.setattr(app, "_load_facts", MagicMock(return_value=facts))
    monkeypatch.setattr(app, "_embed", MagicMock(return_value=[1.0, 0.0]))

    result = app.recall_facts("preferences", n_results=2)

    assert result == {"matches": ["fact one", "fact three"]}

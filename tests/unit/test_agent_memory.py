from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from agent import app


@pytest.fixture(autouse=True)
def isolate_memory_state(monkeypatch):
    """Point memory at a scratch table so tests don't depend on real env vars."""
    monkeypatch.setattr(app, "MEMORY_TABLE", "test-table")


def _mock_table(monkeypatch):
    mock_table = MagicMock()
    mock_dynamodb = MagicMock()
    mock_dynamodb.Table.return_value = mock_table
    monkeypatch.setattr(app, "_dynamodb", mock_dynamodb)
    return mock_table


def test_remember_fact_embeds_and_puts_a_standalone_item(monkeypatch):
    monkeypatch.setattr(app, "_embed", MagicMock(return_value=[0.1, 0.2]))
    mock_table = _mock_table(monkeypatch)

    result = app.remember_fact("the user prefers cold showers")

    assert result == {"stored": "the user prefers cold showers"}
    _, kwargs = mock_table.put_item.call_args
    item = kwargs["Item"]
    assert item["text"] == "the user prefers cold showers"
    assert item["embedding"] == [Decimal("0.1"), Decimal("0.2")]
    assert isinstance(item["fact_id"], str) and item["fact_id"]


def test_load_facts_scans_every_item_and_converts_embeddings_to_float(monkeypatch):
    mock_table = _mock_table(monkeypatch)
    mock_table.scan.return_value = {
        "Items": [
            {"fact_id": "a", "text": "fact a", "embedding": [Decimal("0.1")]},
            {"fact_id": "b", "text": "fact b", "embedding": [Decimal("0.2")]},
        ]
    }

    facts = app._load_facts()

    mock_table.scan.assert_called_once_with()
    assert facts == [
        {"text": "fact a", "embedding": [0.1]},
        {"text": "fact b", "embedding": [0.2]},
    ]


def test_load_facts_follows_pagination(monkeypatch):
    mock_table = _mock_table(monkeypatch)
    mock_table.scan.side_effect = [
        {"Items": [{"fact_id": "a", "text": "fact a", "embedding": [Decimal("0.1")]}], "LastEvaluatedKey": {"fact_id": "a"}},
        {"Items": [{"fact_id": "b", "text": "fact b", "embedding": [Decimal("0.2")]}]},
    ]

    facts = app._load_facts()

    assert mock_table.scan.call_count == 2
    mock_table.scan.assert_any_call(ExclusiveStartKey={"fact_id": "a"})
    assert facts == [
        {"text": "fact a", "embedding": [0.1]},
        {"text": "fact b", "embedding": [0.2]},
    ]


def test_load_facts_returns_empty_list_when_nothing_stored(monkeypatch):
    mock_table = _mock_table(monkeypatch)
    mock_table.scan.return_value = {"Items": []}

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

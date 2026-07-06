import json
from unittest.mock import MagicMock

import pytest

from agent import app


@pytest.fixture(autouse=True)
def isolate_memory_state(monkeypatch):
    """Point memory at a scratch bucket/prefix so tests don't depend on real env vars."""
    monkeypatch.setattr(app, "MEMORY_BUCKET", "test-bucket")
    monkeypatch.setattr(app, "MEMORY_PREFIX", "facts")


def test_fact_key_is_namespaced_under_the_prefix():
    assert app._fact_key("abc") == "facts/abc.json"


def test_remember_fact_embeds_and_puts_a_standalone_object(monkeypatch):
    monkeypatch.setattr(app, "_embed", MagicMock(return_value=[0.1, 0.2]))
    mock_s3 = MagicMock()
    monkeypatch.setattr(app, "_s3", mock_s3)

    result = app.remember_fact("the user prefers cold showers")

    assert result == {"stored": "the user prefers cold showers"}
    _, kwargs = mock_s3.put_object.call_args
    assert kwargs["Bucket"] == "test-bucket"
    assert kwargs["Key"].startswith("facts/") and kwargs["Key"].endswith(".json")
    assert json.loads(kwargs["Body"]) == {"text": "the user prefers cold showers", "embedding": [0.1, 0.2]}


def _paginator_with_pages(pages):
    paginator = MagicMock()
    paginator.paginate.return_value = pages
    return paginator


def test_load_facts_downloads_every_object_under_the_prefix(monkeypatch):
    mock_s3 = MagicMock()
    mock_s3.get_paginator.return_value = _paginator_with_pages(
        [{"Contents": [{"Key": "facts/a.json"}, {"Key": "facts/b.json"}]}]
    )
    bodies = {
        "facts/a.json": {"text": "fact a", "embedding": [0.1]},
        "facts/b.json": {"text": "fact b", "embedding": [0.2]},
    }
    mock_s3.get_object.side_effect = lambda Bucket, Key: {
        "Body": MagicMock(read=MagicMock(return_value=json.dumps(bodies[Key]).encode("utf-8")))
    }
    monkeypatch.setattr(app, "_s3", mock_s3)

    facts = app._load_facts()

    mock_s3.get_paginator.assert_called_once_with("list_objects_v2")
    assert facts == [bodies["facts/a.json"], bodies["facts/b.json"]]


def test_load_facts_returns_empty_list_when_nothing_stored(monkeypatch):
    mock_s3 = MagicMock()
    mock_s3.get_paginator.return_value = _paginator_with_pages([{"Contents": []}])
    monkeypatch.setattr(app, "_s3", mock_s3)

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

import io
import os
import shutil
import tarfile
from unittest.mock import MagicMock

import pytest
from botocore.exceptions import ClientError

from agent import app


@pytest.fixture(autouse=True)
def isolate_memory_state(tmp_path, monkeypatch):
    """Point memory at a scratch dir/bucket and drop any cached collection between tests."""
    monkeypatch.setattr(app, "MEMORY_LOCAL_DIR", str(tmp_path / "chroma"))
    monkeypatch.setattr(app, "MEMORY_BUCKET", "test-bucket")
    monkeypatch.setattr(app, "_memory_collection", None)


def test_download_memory_snapshot_skips_when_local_dir_already_exists(monkeypatch):
    os.makedirs(app.MEMORY_LOCAL_DIR)
    mock_s3 = MagicMock()
    monkeypatch.setattr(app, "_s3", mock_s3)

    app._download_memory_snapshot()

    mock_s3.download_file.assert_not_called()


def test_download_memory_snapshot_starts_empty_when_nothing_saved_yet(monkeypatch):
    mock_s3 = MagicMock()
    mock_s3.download_file.side_effect = ClientError({"Error": {"Code": "404"}}, "GetObject")
    monkeypatch.setattr(app, "_s3", mock_s3)

    app._download_memory_snapshot()

    assert os.listdir(app.MEMORY_LOCAL_DIR) == []


def test_download_memory_snapshot_reraises_unexpected_client_errors(monkeypatch):
    mock_s3 = MagicMock()
    mock_s3.download_file.side_effect = ClientError({"Error": {"Code": "403"}}, "GetObject")
    monkeypatch.setattr(app, "_s3", mock_s3)

    with pytest.raises(ClientError):
        app._download_memory_snapshot()


def test_download_memory_snapshot_extracts_the_archived_store(monkeypatch, tmp_path):
    archive_path = tmp_path / "source.tar.gz"
    with tarfile.open(archive_path, "w:gz") as tar:
        data = b"fake-chroma-bytes"
        info = tarfile.TarInfo(name="chroma.sqlite3")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))

    mock_s3 = MagicMock()
    mock_s3.download_file.side_effect = lambda bucket, key, filename: shutil.copyfile(archive_path, filename)
    monkeypatch.setattr(app, "_s3", mock_s3)

    app._download_memory_snapshot()

    with open(os.path.join(app.MEMORY_LOCAL_DIR, "chroma.sqlite3"), "rb") as f:
        assert f.read() == b"fake-chroma-bytes"


def test_upload_memory_snapshot_archives_the_local_store(monkeypatch):
    os.makedirs(app.MEMORY_LOCAL_DIR)
    with open(os.path.join(app.MEMORY_LOCAL_DIR, "chroma.sqlite3"), "w") as f:
        f.write("data")

    captured = {}

    def fake_upload_file(filename, bucket, key):
        captured["bucket"] = bucket
        captured["key"] = key
        with tarfile.open(filename) as tar:
            captured["names"] = tar.getnames()

    mock_s3 = MagicMock()
    mock_s3.upload_file.side_effect = fake_upload_file
    monkeypatch.setattr(app, "_s3", mock_s3)

    app._upload_memory_snapshot()

    assert captured["bucket"] == "test-bucket"
    assert captured["key"] == app.MEMORY_ARCHIVE_KEY
    assert "./chroma.sqlite3" in captured["names"]


def test_get_memory_collection_is_cached_across_calls(monkeypatch):
    fake_collection = MagicMock()
    fake_chroma_client = MagicMock(get_or_create_collection=MagicMock(return_value=fake_collection))
    mock_persistent_client = MagicMock(return_value=fake_chroma_client)
    monkeypatch.setattr(app.chromadb, "PersistentClient", mock_persistent_client)
    monkeypatch.setattr(app, "_download_memory_snapshot", MagicMock())

    first = app.get_memory_collection()
    second = app.get_memory_collection()

    assert first is fake_collection
    assert second is fake_collection
    mock_persistent_client.assert_called_once()


def test_remember_fact_stores_the_embedding_and_persists_to_s3(monkeypatch):
    fake_collection = MagicMock()
    monkeypatch.setattr(app, "get_memory_collection", MagicMock(return_value=fake_collection))
    monkeypatch.setattr(app, "_embed", MagicMock(return_value=[0.1, 0.2]))
    mock_upload = MagicMock()
    monkeypatch.setattr(app, "_upload_memory_snapshot", mock_upload)

    result = app.remember_fact("the user prefers cold showers")

    assert result == {"stored": "the user prefers cold showers"}
    _, kwargs = fake_collection.add.call_args
    assert kwargs["documents"] == ["the user prefers cold showers"]
    assert kwargs["embeddings"] == [[0.1, 0.2]]
    mock_upload.assert_called_once()


def test_recall_facts_short_circuits_on_an_empty_collection(monkeypatch):
    fake_collection = MagicMock(count=MagicMock(return_value=0))
    monkeypatch.setattr(app, "get_memory_collection", MagicMock(return_value=fake_collection))

    result = app.recall_facts("what does the user like?")

    assert result == {"matches": []}
    fake_collection.query.assert_not_called()


def test_recall_facts_queries_with_the_embedded_query_and_caps_n_results(monkeypatch):
    fake_collection = MagicMock(count=MagicMock(return_value=2))
    fake_collection.query.return_value = {"documents": [["fact one", "fact two"]]}
    monkeypatch.setattr(app, "get_memory_collection", MagicMock(return_value=fake_collection))
    monkeypatch.setattr(app, "_embed", MagicMock(return_value=[0.3, 0.4]))

    result = app.recall_facts("preferences", n_results=5)

    assert result == {"matches": ["fact one", "fact two"]}
    fake_collection.query.assert_called_once_with(query_embeddings=[[0.3, 0.4]], n_results=2)

import socket

import pytest

from config import WEAVIATE_HOST, WEAVIATE_HTTP_PORT
from schemas import Chunk
from weaviate_store import chunk_uuid, connect, fetch_chunk, replace_chunks

_TEST_COLLECTION = "KnowledgeChunkTest"


def test_chunk_uuid_is_stable() -> None:
    assert chunk_uuid("hours", "hours.txt:0") == chunk_uuid("hours", "hours.txt:0")
    assert chunk_uuid("hours", "hours.txt:0") != chunk_uuid("hours", "hours.txt:1")
    assert chunk_uuid("hours", "hours.txt:0") != chunk_uuid("events", "hours.txt:0")


def _weaviate_port_open() -> bool:
    try:
        with socket.create_connection((WEAVIATE_HOST, WEAVIATE_HTTP_PORT), timeout=1):
            return True
    except OSError:
        return False


@pytest.fixture
def weaviate_client():
    if not _weaviate_port_open():
        pytest.skip("Weaviate не запущен")
    client = connect()
    try:
        yield client
    finally:
        if client.collections.exists(_TEST_COLLECTION):
            client.collections.delete(_TEST_COLLECTION)
        client.close()


def _sample_chunks() -> tuple[list[Chunk], list[list[float]]]:
    chunks = [
        Chunk(
            document_id="hours",
            source_name="hours.txt",
            chunk_id="hours.txt:0",
            text="Библиотека открыта в будни с девяти.",
        ),
        Chunk(
            document_id="hours",
            source_name="hours.txt",
            chunk_id="hours.txt:1",
            text="В воскресенье зал закрыт.",
        ),
    ]
    vectors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    return chunks, vectors


@pytest.mark.integration
def test_reload_keeps_the_same_object_count(weaviate_client) -> None:
    chunks, vectors = _sample_chunks()

    first = replace_chunks(weaviate_client, chunks, vectors, _TEST_COLLECTION)
    second = replace_chunks(weaviate_client, chunks, vectors, _TEST_COLLECTION)

    assert first == second == len(chunks)


@pytest.mark.integration
def test_stored_chunk_keeps_metadata(weaviate_client) -> None:
    chunks, vectors = _sample_chunks()
    replace_chunks(weaviate_client, chunks, vectors, _TEST_COLLECTION)

    stored = fetch_chunk(weaviate_client, chunks[0].chunk_id, _TEST_COLLECTION)

    assert stored == chunks[0]


@pytest.mark.integration
def test_near_vector_returns_the_same_chunk(weaviate_client) -> None:
    chunks, vectors = _sample_chunks()
    replace_chunks(weaviate_client, chunks, vectors, _TEST_COLLECTION)

    collection = weaviate_client.collections.get(_TEST_COLLECTION)
    found = collection.query.near_vector(near_vector=vectors[0], limit=1)

    assert found.objects
    assert found.objects[0].properties["chunk_id"] == chunks[0].chunk_id

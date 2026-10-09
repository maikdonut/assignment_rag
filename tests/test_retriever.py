import json
import socket
from pathlib import Path
from typing import cast

import pytest
from pydantic import BaseModel, TypeAdapter
from weaviate.client import WeaviateClient

from chunking import chunk_documents
from config import KNOWLEDGE_BASE_DIR, TOP_K, WEAVIATE_HOST, WEAVIATE_HTTP_PORT
from embedder import embed_texts
from loader import load_documents
from retriever import distance_to_score, search
from schemas import Chunk
from weaviate_store import connect, replace_chunks

_TEST_COLLECTION = "KnowledgeChunkRetrievalTest"
_QUESTIONS_PATH = Path(__file__).parent / "data" / "retrieval_questions.json"


class _QuestionCase(BaseModel):
    question: str
    source_name: str


class _FixedVectorModel:
    def encode(self, texts: list[str], **kwargs: object) -> list[list[float]]:
        return [[1.0, 0.0, 0.0] for _ in texts]


def test_distance_to_score_is_one_minus_distance() -> None:
    assert distance_to_score(0.0) == 1.0
    assert distance_to_score(0.25) == 0.75
    assert distance_to_score(1.0) == 0.0


def test_search_rejects_blank_query() -> None:
    with pytest.raises(ValueError, match="вопрос пустой"):
        search(cast(WeaviateClient, object()), "  \n")


def test_search_rejects_non_positive_top_k() -> None:
    with pytest.raises(ValueError, match="top_k"):
        search(cast(WeaviateClient, object()), "когда открыто", top_k=0)


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


def _orthogonal_chunks() -> tuple[list[Chunk], list[list[float]]]:
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
        Chunk(
            document_id="events",
            source_name="events.txt",
            chunk_id="events.txt:0",
            text="Книжный клуб собирается в четверг.",
        ),
    ]
    vectors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    return chunks, vectors


@pytest.mark.integration
def test_search_returns_nearest_chunk_with_score(weaviate_client) -> None:
    chunks, vectors = _orthogonal_chunks()
    replace_chunks(weaviate_client, chunks, vectors, _TEST_COLLECTION)

    top_k = 2
    hits = search(
        weaviate_client,
        "когда открыто в будни",
        top_k=top_k,
        collection_name=_TEST_COLLECTION,
        model=_FixedVectorModel(),
    )

    assert len(hits) == top_k
    nearest = hits[0]
    assert nearest.chunk_id == chunks[0].chunk_id
    assert nearest.source_name == chunks[0].source_name
    assert nearest.distance == pytest.approx(0.0, abs=1e-5)
    assert nearest.score == pytest.approx(distance_to_score(nearest.distance))
    assert nearest.text == chunks[0].text


def _load_questions() -> list[_QuestionCase]:
    raw = json.loads(_QUESTIONS_PATH.read_text(encoding="utf-8"))
    return TypeAdapter(list[_QuestionCase]).validate_python(raw)


@pytest.mark.integration
def test_questions_find_expected_source(weaviate_client) -> None:
    documents = load_documents(KNOWLEDGE_BASE_DIR)
    chunks = chunk_documents(documents)
    vectors = embed_texts([chunk.text for chunk in chunks])
    replace_chunks(weaviate_client, chunks, vectors, _TEST_COLLECTION)

    questions = _load_questions()
    assert len(questions) >= 5
    for case in questions:
        hits = search(
            weaviate_client,
            case.question,
            top_k=TOP_K,
            collection_name=_TEST_COLLECTION,
        )
        sources = [hit.source_name for hit in hits]
        assert case.source_name in sources, case.question

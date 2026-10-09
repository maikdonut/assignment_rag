import pytest

from chunking import chunk_document, chunk_documents, split_text
from config import CHUNK_OVERLAP, CHUNK_SIZE, KNOWLEDGE_BASE_DIR
from loader import load_documents
from schemas import Chunk, Document


def test_short_text_is_one_chunk_with_metadata() -> None:
    document = Document(document_id="hours", source_name="hours.txt", text="Короткий текст")

    chunks = chunk_document(document, chunk_size=800, overlap=150)

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.text == "Короткий текст"
    assert chunk.document_id == "hours"
    assert chunk.source_name == "hours.txt"
    assert chunk.chunk_id == "hours.txt:0"


def test_long_text_respects_size_and_overlap() -> None:
    chunk_size = 10
    overlap = 3
    text = "abcdefghijklmnopqrstuvwxyz"

    parts = split_text(text, chunk_size, overlap)

    assert parts == [
        "abcdefghij",
        "hijklmnopq",
        "opqrstuvwx",
        "vwxyz",
    ]
    assert len(parts[-1]) < chunk_size
    for index, part in enumerate(parts[:-1]):
        assert len(part) == chunk_size
        assert part[-overlap:] == parts[index + 1][:overlap]


def test_empty_text_yields_no_chunks() -> None:
    assert split_text("", chunk_size=10, overlap=2) == []


def test_invalid_size_or_overlap_raises() -> None:
    with pytest.raises(ValueError):
        split_text("abc", chunk_size=0, overlap=0)
    with pytest.raises(ValueError):
        split_text("abc", chunk_size=10, overlap=10)
    with pytest.raises(ValueError):
        split_text("abc", chunk_size=10, overlap=-1)


def test_knowledge_base_chunks_have_metadata() -> None:
    documents = load_documents(KNOWLEDGE_BASE_DIR)
    chunks = chunk_documents(documents)
    by_source: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        by_source.setdefault(chunk.source_name, []).append(chunk)

    assert documents
    for document in documents:
        assert len(document.text) >= 3000
        document_chunks = by_source[document.source_name]
        assert len(document_chunks) > 1
        for index, chunk in enumerate(document_chunks):
            assert chunk.document_id == document.document_id
            assert chunk.source_name == document.source_name
            assert chunk.chunk_id == f"{document.source_name}:{index}"
            assert chunk.text
            assert len(chunk.text) <= CHUNK_SIZE
        for index in range(len(document_chunks) - 1):
            left = document_chunks[index].text
            right = document_chunks[index + 1].text
            assert left[-CHUNK_OVERLAP:] == right[:CHUNK_OVERLAP]

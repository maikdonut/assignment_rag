import logging

from config import CHUNK_OVERLAP, CHUNK_SIZE
from schemas import Chunk, Document

logger = logging.getLogger(__name__)


def split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Режет текст на окна не длиннее ``chunk_size`` с перекрытием ``overlap`` символов."""
    if chunk_size < 1:
        raise ValueError(f"размер чанка должен быть положительным: {chunk_size}")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            f"overlap должен быть от 0 до размера чанка, не включая его: {overlap}"
        )
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    step = chunk_size - overlap
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start += step
    return chunks


def chunk_document(
    document: Document,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[Chunk]:
    """Режет один документ и проставляет метаданные каждого чанка."""
    parts = split_text(document.text, chunk_size, overlap)
    return [
        Chunk(
            document_id=document.document_id,
            source_name=document.source_name,
            chunk_id=f"{document.source_name}:{index}",
            text=part,
        )
        for index, part in enumerate(parts)
    ]


def chunk_documents(
    documents: list[Document],
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[Chunk]:
    """Режет документы в исходном порядке и возвращает общий список чанков."""
    chunks: list[Chunk] = []
    for document in documents:
        chunks.extend(chunk_document(document, chunk_size, overlap))
    logger.info("документы разрезаны на чанков: %s", len(chunks))
    return chunks

import argparse
import logging

from chunking import chunk_documents
from config import COLLECTION_NAME, KNOWLEDGE_BASE_DIR, RELEVANCE_THRESHOLD
from embedder import embed_texts, resolve_device
from loader import load_documents
from rag_pipeline import answer, answer_without_retrieval
from schemas import Chunk, RagAnswer, RetrievedChunk
from weaviate_store import connect, fetch_chunk, replace_chunks

_EXAMPLE_COUNT = 3
_PREVIEW_CHARS = 200


def main(argv: list[str] | None = None) -> None:
    """Печатает документы, загружает чанки в Weaviate или отвечает на вопрос."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    for logger_name in ("httpx", "httpx2", "httpcore", "openai", "huggingface_hub", "sentence_transformers"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(description="Локальная база знаний библиотеки")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("ingest", help="Загрузить чанки и эмбеддинги в Weaviate")
    ask_parser = subparsers.add_parser("ask", help="Ответить на вопрос по базе знаний")
    ask_parser.add_argument("question", help="Вопрос к базе знаний")
    ask_parser.add_argument(
        "--compare",
        action="store_true",
        help="Дополнительно показать ответ без поиска",
    )
    args = parser.parse_args(argv)
    if args.command == "ingest":
        ingest()
        return
    if args.command == "ask":
        ask(args.question, compare=args.compare)
        return
    show_documents()


def show_documents() -> None:
    """Печатает список документов и несколько примеров чанков с метаданными."""
    documents = load_documents(KNOWLEDGE_BASE_DIR)
    chunks = chunk_documents(documents)
    print(f"Загружено документов: {len(documents)}")
    for document in documents:
        print(f"{document.source_name}\t{len(document.text)}")
    print(f"Чанков: {len(chunks)}")
    print("Примеры чанков:")
    for chunk in chunks[:_EXAMPLE_COUNT]:
        print(f"chunk_id={chunk.chunk_id}")
        print(f"document_id={chunk.document_id}")
        print(f"source_name={chunk.source_name}")
        print(f"длина={len(chunk.text)}")
        print(chunk.text)
        print("---")


def ingest() -> None:
    """Считает эмбеддинги и заменяет коллекцию чанков в Weaviate."""
    documents = load_documents(KNOWLEDGE_BASE_DIR)
    chunks = chunk_documents(documents)
    device = resolve_device()
    vectors = embed_texts([chunk.text for chunk in chunks], device=device)
    try:
        client = connect()
    except (ConnectionError, OSError):
        raise SystemExit(1) from None
    try:
        count = replace_chunks(client, chunks, vectors, COLLECTION_NAME)
        sample = fetch_chunk(client, chunks[0].chunk_id, COLLECTION_NAME) if chunks else None
    finally:
        client.close()
    print(f"Устройство: {device}")
    print(f"Объектов в Weaviate: {count}")
    if sample is not None:
        _print_stored_chunk(sample)


def ask(question: str, *, compare: bool = False) -> None:
    """Печатает ответ по найденным фрагментам, источники и top-k."""
    try:
        client = connect()
    except (ConnectionError, OSError):
        raise SystemExit(1) from None
    try:
        result = answer(client, question)
    except (ValueError, LookupError, RuntimeError) as exc:
        print(exc)
        raise SystemExit(1) from None
    finally:
        client.close()

    bare: str | None = None
    if compare:
        try:
            bare = answer_without_retrieval(question)
        except (ValueError, RuntimeError) as exc:
            print(exc)
            raise SystemExit(1) from None

    _print_answer(result)
    if bare is not None:
        print("Ответ без поиска:")
        print(bare)


def _print_answer(result: RagAnswer) -> None:
    print("Ответ:")
    print(result.answer)
    print(_context_line(result))
    print("Источники:")
    if not result.sources:
        print("нет")
    for source in result.sources:
        print(f"source_name={source.source_name} chunk_id={source.chunk_id}")
    print(f"Найдено фрагментов: {len(result.chunks)}")
    for index, hit in enumerate(result.chunks, start=1):
        _print_hit(index, hit)


def _context_line(result: RagAnswer) -> str:
    """Одна строка: прошёл ли лучший контекст порог и вызывалась ли модель."""
    if not result.chunks:
        return "Контекст: фрагментов нет, модель не вызывалась"
    if result.is_grounded:
        return f"Контекст: прошёл порог {RELEVANCE_THRESHOLD:.2f}"
    return f"Контекст: ниже порога {RELEVANCE_THRESHOLD:.2f}, модель не вызывалась"


def _print_hit(index: int, hit: RetrievedChunk) -> None:
    print(f"{index}. score={hit.score:.4f} distance={hit.distance:.4f}")
    print(f"source_name={hit.source_name}")
    print(f"chunk_id={hit.chunk_id}")
    preview = hit.text
    if len(preview) > _PREVIEW_CHARS:
        preview = preview[:_PREVIEW_CHARS] + "..."
    print(preview)
    print("---")


def _print_stored_chunk(chunk: Chunk) -> None:
    print("Объект из базы:")
    print(f"chunk_id={chunk.chunk_id}")
    print(f"document_id={chunk.document_id}")
    print(f"source_name={chunk.source_name}")
    preview = chunk.text
    if len(preview) > _PREVIEW_CHARS:
        preview = preview[:_PREVIEW_CHARS] + "..."
    print(preview)


if __name__ == "__main__":
    main()

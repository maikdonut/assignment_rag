import logging

from chunking import chunk_documents
from config import KNOWLEDGE_BASE_DIR
from loader import load_documents

_EXAMPLE_COUNT = 3


def main() -> None:
    """Печатает список документов и несколько примеров чанков с метаданными."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
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


if __name__ == "__main__":
    main()

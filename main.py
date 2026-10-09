import logging

from config import KNOWLEDGE_BASE_DIR
from loader import load_documents


def main() -> None:
    """Печатает список загруженных документов и длину очищенного текста."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    documents = load_documents(KNOWLEDGE_BASE_DIR)
    print(f"Загружено документов: {len(documents)}")
    for document in documents:
        print(f"{document.source_name}\t{len(document.text)}")


if __name__ == "__main__":
    main()

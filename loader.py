import logging
import re
from pathlib import Path

from schemas import Document

logger = logging.getLogger(__name__)

_INLINE_WHITESPACE = re.compile(r"[ \t]+")


def clean_text(raw: str) -> str:
    """Нормализует переводы строк, схлопывает пробелы в строке и убирает пустые строки."""
    normalized = raw.replace("\r\n", "\n").replace("\r", "\n")
    lines: list[str] = []
    for line in normalized.split("\n"):
        collapsed = _INLINE_WHITESPACE.sub(" ", line).strip()
        if collapsed:
            lines.append(collapsed)
    return "\n".join(lines).strip()


def load_documents(directory: Path) -> list[Document]:
    """Загружает и очищает все файлы ``*.txt`` в ``directory``, по имени файла."""
    if not directory.is_dir():
        logger.error("каталог базы знаний не существует: %s", directory)
        raise FileNotFoundError(directory)

    documents: list[Document] = []
    for path in sorted(directory.glob("*.txt")):
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError:
            logger.exception("не удалось прочитать документ: %s", path)
            raise
        except UnicodeError:
            logger.exception("не удалось декодировать документ: %s", path)
            raise

        text = clean_text(raw)
        if not text:
            logger.warning("пропускаю пустой документ после очистки: %s", path.name)
            continue

        documents.append(
            Document(
                document_id=path.stem,
                source_name=path.name,
                text=text,
            )
        )
    return documents

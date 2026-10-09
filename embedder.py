import logging
from typing import Protocol, cast

from config import EMBEDDING_BATCH_SIZE, EMBEDDING_MODEL

logger = logging.getLogger(__name__)

_model: object | None = None


class EmbeddingModel(Protocol):
    """Модель, которая превращает список текстов в список векторов."""

    def encode(self, texts: list[str], **kwargs: object) -> list[list[float]]:
        """Считает вектор для каждого текста в том же порядке."""
        ...


def resolve_device() -> str:
    """Возвращает ``cuda``, если GPU доступен, иначе ``cpu``."""
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("устройство для эмбеддингов: %s", device)
    return device


def get_model(device: str | None = None) -> EmbeddingModel:
    """Загружает ``deepvk/USER2-base`` один раз на выбранное устройство."""
    global _model
    if _model is None:
        chosen = device if device is not None else resolve_device()
        logger.info("загружаю модель эмбеддингов: %s", EMBEDDING_MODEL)
        try:
            from sentence_transformers import SentenceTransformer

            _model = SentenceTransformer(EMBEDDING_MODEL, device=chosen)
        except Exception:
            logger.exception("не удалось загрузить модель эмбеддингов: %s", EMBEDDING_MODEL)
            raise
    return cast(EmbeddingModel, _model)


def embed_texts(
    texts: list[str],
    model: EmbeddingModel | None = None,
    batch_size: int = EMBEDDING_BATCH_SIZE,
    device: str | None = None,
) -> list[list[float]]:
    """Считает эмбеддинги батчами. Пустой список текстов даёт пустой результат."""
    if batch_size < 1:
        raise ValueError(f"размер батча должен быть положительным: {batch_size}")
    if not texts:
        return []

    embedder = model if model is not None else get_model(device)
    vectors: list[list[float]] = []
    try:
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            vectors.extend(_as_vectors(embedder.encode(batch, show_progress_bar=False)))
    except Exception:
        logger.exception("не удалось посчитать эмбеддинги, текстов: %s", len(texts))
        raise

    if len(vectors) != len(texts):
        logger.error(
            "число эмбеддингов не совпало с числом текстов: %s и %s",
            len(vectors),
            len(texts),
        )
        raise RuntimeError("число эмбеддингов не совпало с числом текстов")
    return vectors


def _as_vectors(encoded: object) -> list[list[float]]:
    raw = encoded.tolist() if hasattr(encoded, "tolist") else encoded
    return [[float(value) for value in vector] for vector in raw]

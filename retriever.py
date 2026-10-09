import logging

from weaviate.classes.query import MetadataQuery
from weaviate.client import WeaviateClient
from weaviate.exceptions import WeaviateConnectionError, WeaviateQueryError

from config import COLLECTION_NAME, TOP_K
from embedder import EmbeddingModel, embed_texts
from schemas import RetrievedChunk

logger = logging.getLogger(__name__)


def distance_to_score(distance: float) -> float:
    """Переводит косинусную дистанцию Weaviate в близость: ``1`` — совпадение векторов."""
    return 1.0 - distance


def search(
    client: WeaviateClient,
    query: str,
    *,
    top_k: int = TOP_K,
    collection_name: str = COLLECTION_NAME,
    model: EmbeddingModel | None = None,
) -> list[RetrievedChunk]:
    """Ищет ``top_k`` ближайших чанков по эмбеддингу вопроса."""
    cleaned = query.strip()
    if not cleaned:
        raise ValueError("вопрос пустой")
    if top_k < 1:
        raise ValueError(f"top_k должен быть положительным: {top_k}")
    if not client.collections.exists(collection_name):
        logger.error("коллекция не найдена: %s", collection_name)
        raise LookupError(f"коллекция не найдена: {collection_name}")

    vector = embed_texts([cleaned], model=model)[0]
    collection = client.collections.get(collection_name)
    logger.info("поиск: %s, top_k=%s", cleaned, top_k)
    try:
        response = collection.query.near_vector(
            near_vector=vector,
            limit=top_k,
            return_metadata=MetadataQuery(distance=True),
        )
    except (WeaviateQueryError, WeaviateConnectionError):
        logger.exception("не удалось выполнить поиск в %s", collection_name)
        raise

    hits: list[RetrievedChunk] = []
    for item in response.objects:
        distance = item.metadata.distance
        if distance is None:
            logger.error("Weaviate не вернул distance")
            raise RuntimeError("Weaviate не вернул distance")
        properties = item.properties
        hit = RetrievedChunk(
            document_id=str(properties["document_id"]),
            source_name=str(properties["source_name"]),
            chunk_id=str(properties["chunk_id"]),
            text=str(properties["text"]),
            distance=float(distance),
            score=distance_to_score(float(distance)),
        )
        logger.info(
            "найден фрагмент: source_name=%s chunk_id=%s score=%.4f",
            hit.source_name,
            hit.chunk_id,
            hit.score,
        )
        hits.append(hit)
    return hits

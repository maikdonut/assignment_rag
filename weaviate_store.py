import logging

import weaviate
from weaviate.classes.config import Configure, DataType, Property, VectorDistances
from weaviate.classes.data import DataObject
from weaviate.classes.query import Filter
from weaviate.client import WeaviateClient
from weaviate.exceptions import (
    WeaviateBatchError,
    WeaviateConnectionError,
    WeaviateInsertInvalidPropertyError,
    WeaviateInsertManyAllFailedError,
)
from weaviate.util import generate_uuid5

from config import (
    COLLECTION_NAME,
    WEAVIATE_GRPC_PORT,
    WEAVIATE_HOST,
    WEAVIATE_HTTP_PORT,
)
from schemas import Chunk

logger = logging.getLogger(__name__)


def chunk_uuid(document_id: str, chunk_id: str) -> str:
    """Стабильный UUID объекта Weaviate для пары ``document_id`` и ``chunk_id``."""
    return generate_uuid5(f"{document_id}\n{chunk_id}")


def connect() -> WeaviateClient:
    """Подключается к локальному Weaviate и проверяет, что он готов."""
    try:
        client = weaviate.connect_to_local(
            host=WEAVIATE_HOST,
            port=WEAVIATE_HTTP_PORT,
            grpc_port=WEAVIATE_GRPC_PORT,
        )
    except WeaviateConnectionError as exc:
        logger.exception(
            "не удалось подключиться к Weaviate: %s:%s",
            WEAVIATE_HOST,
            WEAVIATE_HTTP_PORT,
        )
        raise ConnectionError("не удалось подключиться к Weaviate") from exc

    if not client.is_ready():
        client.close()
        logger.error(
            "Weaviate не готов: %s:%s",
            WEAVIATE_HOST,
            WEAVIATE_HTTP_PORT,
        )
        raise ConnectionError("Weaviate не готов")
    return client


def replace_chunks(
    client: WeaviateClient,
    chunks: list[Chunk],
    vectors: list[list[float]],
    collection_name: str = COLLECTION_NAME,
) -> int:
    """Пересоздаёт коллекцию и записывает чанки с векторами. Возвращает число объектов."""
    if len(chunks) != len(vectors):
        raise ValueError(
            f"число чанков и эмбеддингов не совпадает: {len(chunks)} и {len(vectors)}"
        )
    _validate_vectors(vectors)

    if client.collections.exists(collection_name):
        client.collections.delete(collection_name)
        logger.info("удалена коллекция перед повторной загрузкой: %s", collection_name)

    collection = client.collections.create(
        name=collection_name,
        vector_config=Configure.Vectors.self_provided(
            vector_index_config=Configure.VectorIndex.hnsw(
                distance_metric=VectorDistances.COSINE,
            ),
        ),
        properties=[
            Property(name="document_id", data_type=DataType.TEXT),
            Property(name="source_name", data_type=DataType.TEXT),
            Property(name="chunk_id", data_type=DataType.TEXT),
            Property(name="text", data_type=DataType.TEXT),
        ],
    )
    logger.info("создана коллекция: %s", collection_name)

    if not chunks:
        return 0

    objects = [
        DataObject(
            properties={
                "document_id": chunk.document_id,
                "source_name": chunk.source_name,
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
            },
            uuid=chunk_uuid(chunk.document_id, chunk.chunk_id),
            vector=vector,
        )
        for chunk, vector in zip(chunks, vectors, strict=True)
    ]
    try:
        result = collection.data.insert_many(objects)
    except (
        WeaviateBatchError,
        WeaviateInsertInvalidPropertyError,
        WeaviateInsertManyAllFailedError,
    ):
        logger.exception("не удалось вставить объекты в %s", collection_name)
        raise
    if result.has_errors:
        for index, error in result.errors.items():
            message = getattr(error, "message", error)
            logger.error("ошибка вставки объекта %s: %s", index, message)
        raise RuntimeError("не удалось вставить часть объектов в Weaviate")

    stored = count_objects(client, collection_name)
    if stored != len(chunks):
        logger.error(
            "в коллекции %s объектов, ожидалось %s",
            stored,
            len(chunks),
        )
        raise RuntimeError("число объектов в Weaviate не совпало с числом чанков")
    logger.info("загружено объектов в %s: %s", collection_name, stored)
    return stored


def count_objects(client: WeaviateClient, collection_name: str = COLLECTION_NAME) -> int:
    """Возвращает число объектов в коллекции."""
    collection = client.collections.get(collection_name)
    aggregated = collection.aggregate.over_all(total_count=True)
    return int(aggregated.total_count or 0)


def fetch_chunk(
    client: WeaviateClient,
    chunk_id: str,
    collection_name: str = COLLECTION_NAME,
) -> Chunk | None:
    """Читает один чанк по ``chunk_id``. ``None``, если объекта нет."""
    collection = client.collections.get(collection_name)
    response = collection.query.fetch_objects(
        filters=Filter.by_property("chunk_id").equal(chunk_id),
        limit=1,
    )
    if not response.objects:
        return None
    properties = response.objects[0].properties
    return Chunk(
        document_id=str(properties["document_id"]),
        source_name=str(properties["source_name"]),
        chunk_id=str(properties["chunk_id"]),
        text=str(properties["text"]),
    )


def _validate_vectors(vectors: list[list[float]]) -> None:
    if not vectors:
        return
    width = len(vectors[0])
    if width == 0:
        raise ValueError("эмбеддинг пустой")
    for vector in vectors:
        if len(vector) != width:
            raise ValueError("эмбеддинги разной длины")

from pydantic import BaseModel


class Document(BaseModel):
    """Очищенный документ базы знаний, готовый к разбиению на чанки."""

    document_id: str
    source_name: str
    text: str


class Chunk(BaseModel):
    """Фрагмент документа с метаданными для дальнейшего поиска."""

    document_id: str
    source_name: str
    chunk_id: str
    text: str


class RetrievedChunk(BaseModel):
    """Чанк, найденный поиском, с дистанцией Weaviate и оценкой близости."""

    document_id: str
    source_name: str
    chunk_id: str
    text: str
    distance: float
    score: float


class SourceRef(BaseModel):
    """Источник, на который опирается ответ: файл и идентификатор чанка."""

    source_name: str
    chunk_id: str


class RagAnswer(BaseModel):
    """Ответ модели вместе с источниками и найденными чанками."""

    answer: str
    sources: list[SourceRef]
    is_grounded: bool
    chunks: list[RetrievedChunk]

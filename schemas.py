from pydantic import BaseModel


class Document(BaseModel):
    """Очищенный документ базы знаний, готовый к разбиению на чанки."""

    document_id: str
    source_name: str
    text: str

import logging
from typing import Protocol

from openai import APIConnectionError, APIStatusError, AuthenticationError, OpenAI
from weaviate.client import WeaviateClient

from config import COLLECTION_NAME, LLM_MODEL, OPENROUTER_API_KEY, OPENROUTER_BASE_URL, TOP_K
from prompts import SYSTEM_PROMPT, UNGROUNDED_SYSTEM_PROMPT, build_user_message
from retriever import search
from schemas import RagAnswer, RetrievedChunk, SourceRef

logger = logging.getLogger(__name__)

EMPTY_CONTEXT_ANSWER = "В базе знаний нет фрагментов по этому вопросу."


class ChatModel(Protocol):
    """Модель, которая отвечает текстом на системное и пользовательское сообщения."""

    def complete(self, system: str, user: str) -> str:
        """Возвращает текст ответа."""
        ...


class OpenRouterChat:
    """Клиент чата OpenRouter через совместимый OpenAI SDK."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ) -> None:
        key = (OPENROUTER_API_KEY if api_key is None else api_key).strip()
        if not key:
            logger.error("не задан OPENROUTER_API_KEY")
            raise RuntimeError("не задан OPENROUTER_API_KEY")
        self._model = LLM_MODEL if model is None else model
        self._client = OpenAI(api_key=key, base_url=base_url or OPENROUTER_BASE_URL)

    def complete(self, system: str, user: str) -> str:
        """Запрашивает ответ модели и возвращает текст сообщения."""
        logger.info("запрос к модели: %s", self._model)
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
        except AuthenticationError:
            logger.error("OpenRouter отклонил ключ доступа")
            raise RuntimeError("OpenRouter отклонил ключ доступа") from None
        except APIStatusError as exc:
            logger.error("OpenRouter вернул ошибку: %s", exc.status_code)
            raise RuntimeError(f"OpenRouter вернул ошибку: {exc.status_code}") from None
        except APIConnectionError:
            logger.error("не удалось связаться с OpenRouter")
            raise RuntimeError("не удалось связаться с OpenRouter") from None

        if not response.choices:
            logger.error("OpenRouter вернул пустой список ответов")
            raise RuntimeError("OpenRouter вернул пустой список ответов")
        content = response.choices[0].message.content
        if content is None or not content.strip():
            logger.error("OpenRouter вернул пустой текст ответа")
            raise RuntimeError("OpenRouter вернул пустой текст ответа")
        return content.strip()

    def close(self) -> None:
        """Закрывает HTTP-соединения клиента."""
        self._client.close()


def generate_answer(
    question: str,
    chunks: list[RetrievedChunk],
    chat: ChatModel | None = None,
) -> RagAnswer:
    """Строит ответ по уже найденным чанкам. Пустой список чанков модель не вызывает."""
    cleaned = question.strip()
    if not cleaned:
        raise ValueError("вопрос пустой")
    if not chunks:
        logger.info("контекст пуст, модель не вызывается")
        return RagAnswer(
            answer=EMPTY_CONTEXT_ANSWER,
            sources=[],
            is_grounded=False,
            chunks=[],
        )

    logger.info("контекст передан в модель, фрагментов: %s", len(chunks))
    text = _complete(chat, SYSTEM_PROMPT, build_user_message(cleaned, chunks))
    return RagAnswer(
        answer=text,
        sources=_sources(chunks),
        is_grounded=True,
        chunks=chunks,
    )


def answer(
    client: WeaviateClient,
    question: str,
    *,
    top_k: int = TOP_K,
    collection_name: str = COLLECTION_NAME,
    chat: ChatModel | None = None,
) -> RagAnswer:
    """Ищет чанки и строит ответ только по найденному контексту."""
    logger.info("запрос: %s, top_k=%s", question.strip(), top_k)
    chunks = search(client, question, top_k=top_k, collection_name=collection_name)
    for chunk in chunks:
        logger.info(
            "фрагмент: source_name=%s chunk_id=%s score=%.4f",
            chunk.source_name,
            chunk.chunk_id,
            chunk.score,
        )
    return generate_answer(question, chunks, chat)


def answer_without_retrieval(question: str, chat: ChatModel | None = None) -> str:
    """Отвечает без документов: только для сравнения с ответом по контексту."""
    cleaned = question.strip()
    if not cleaned:
        raise ValueError("вопрос пустой")
    logger.info("сравнение без поиска: %s", cleaned)
    return _complete(chat, UNGROUNDED_SYSTEM_PROMPT, cleaned)


def _complete(chat: ChatModel | None, system: str, user: str) -> str:
    """Вызывает модель. Клиент, созданный здесь, закрывается после ответа."""
    if chat is not None:
        return chat.complete(system, user)
    model = OpenRouterChat()
    try:
        return model.complete(system, user)
    finally:
        model.close()


def _sources(chunks: list[RetrievedChunk]) -> list[SourceRef]:
    return [SourceRef(source_name=chunk.source_name, chunk_id=chunk.chunk_id) for chunk in chunks]

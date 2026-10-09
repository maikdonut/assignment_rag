from typing import cast

import pytest
from weaviate.client import WeaviateClient

from prompts import SYSTEM_PROMPT, UNGROUNDED_SYSTEM_PROMPT, format_context
from rag_pipeline import (
    EMPTY_CONTEXT_ANSWER,
    OpenRouterChat,
    answer,
    answer_without_retrieval,
    generate_answer,
)
from schemas import RetrievedChunk


class _RecordingChat:
    def __init__(self, text: str = "Книги выдают на 21 день.") -> None:
        self.text = text
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return self.text


class _ForbiddenChat:
    def complete(self, system: str, user: str) -> str:
        raise AssertionError("модель не должна вызываться")


def _hit(
    source_name: str = "borrowing_rules.txt",
    chunk_id: str = "borrowing_rules.txt:0",
    text: str = "Обычные книги выдают на 21 день.",
    score: float = 0.8,
) -> RetrievedChunk:
    return RetrievedChunk(
        document_id="borrowing_rules",
        source_name=source_name,
        chunk_id=chunk_id,
        text=text,
        distance=1.0 - score,
        score=score,
    )


def test_format_context_numbers_sources_and_chunk_ids() -> None:
    chunks = [
        _hit(text="Срок выдачи — 21 день."),
        _hit(source_name="hours.txt", chunk_id="hours.txt:1", text="Открыто до 20:00."),
    ]
    context = format_context(chunks)
    assert context.startswith("[1] source_name=borrowing_rules.txt chunk_id=borrowing_rules.txt:0\n")
    assert "Срок выдачи — 21 день." in context
    assert "[2] source_name=hours.txt chunk_id=hours.txt:1\nОткрыто до 20:00." in context


def test_system_prompt_forbids_facts_outside_context() -> None:
    assert "только по нумерованным фрагментам" in SYSTEM_PROMPT
    assert "не знаешь" in SYSTEM_PROMPT
    assert "Не используй факты вне контекста." in SYSTEM_PROMPT
    assert SYSTEM_PROMPT != UNGROUNDED_SYSTEM_PROMPT


def test_generate_answer_passes_chunks_and_keeps_sources() -> None:
    chunks = [
        _hit(text="Обычные книги выдают на 21 день."),
        _hit(source_name="membership.txt", chunk_id="membership.txt:2", text="Гостевая карта стоит 5 долларов."),
    ]
    chat = _RecordingChat("На 21 день.")
    result = generate_answer("На сколько дней выдают книги?", chunks, chat)

    assert result.answer == "На 21 день."
    assert result.is_grounded is True
    assert [(source.source_name, source.chunk_id) for source in result.sources] == [
        ("borrowing_rules.txt", "borrowing_rules.txt:0"),
        ("membership.txt", "membership.txt:2"),
    ]
    assert result.chunks == chunks
    assert len(chat.calls) == 1
    system, user = chat.calls[0]
    assert system == SYSTEM_PROMPT
    assert "Обычные книги выдают на 21 день." in user
    assert "Гостевая карта стоит 5 долларов." in user
    assert "chunk_id=membership.txt:2" in user


def test_generate_answer_skips_model_when_no_chunks() -> None:
    result = generate_answer("Какой курс доллара?", [], _ForbiddenChat())
    assert result.answer == EMPTY_CONTEXT_ANSWER
    assert result.is_grounded is False
    assert result.sources == []
    assert result.chunks == []


def test_generate_answer_rejects_blank_question() -> None:
    with pytest.raises(ValueError, match="вопрос пустой"):
        generate_answer("  ", [_hit()], _RecordingChat())


def test_answer_searches_then_generates(monkeypatch: pytest.MonkeyPatch) -> None:
    chunks = [_hit()]
    seen: dict[str, object] = {}

    def _search(client: object, query: str, **kwargs: object) -> list[RetrievedChunk]:
        seen["query"] = query
        seen["top_k"] = kwargs["top_k"]
        return chunks

    monkeypatch.setattr("rag_pipeline.search", _search)
    chat = _RecordingChat()
    result = answer(cast(WeaviateClient, object()), "  срок выдачи  ", top_k=3, chat=chat)
    assert seen == {"query": "  срок выдачи  ", "top_k": 3}
    assert result.is_grounded is True
    assert result.sources[0].chunk_id == "borrowing_rules.txt:0"
    assert chat.calls[0][0] == SYSTEM_PROMPT


def test_answer_without_retrieval_uses_ungrounded_prompt() -> None:
    chat = _RecordingChat("Не знаю точный срок.")
    text = answer_without_retrieval("На сколько дней выдают книги?", chat)
    assert text == "Не знаю точный срок."
    system, user = chat.calls[0]
    assert system == UNGROUNDED_SYSTEM_PROMPT
    assert user == "На сколько дней выдают книги?"


def test_owned_client_is_closed_after_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []

    class _ClosableChat:
        def complete(self, system: str, user: str) -> str:
            events.append("complete")
            return "ответ"

        def close(self) -> None:
            events.append("close")

    monkeypatch.setattr("rag_pipeline.OpenRouterChat", _ClosableChat)
    generate_answer("срок выдачи", [_hit()])
    answer_without_retrieval("срок выдачи")
    assert events == ["complete", "close", "complete", "close"]


def test_owned_client_is_closed_when_completion_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []

    class _FailingChat:
        def complete(self, system: str, user: str) -> str:
            events.append("complete")
            raise RuntimeError("сбой модели")

        def close(self) -> None:
            events.append("close")

    monkeypatch.setattr("rag_pipeline.OpenRouterChat", _FailingChat)
    with pytest.raises(RuntimeError, match="сбой модели"):
        generate_answer("срок выдачи", [_hit()])
    assert events == ["complete", "close"]


def test_openrouter_chat_requires_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("rag_pipeline.OPENROUTER_API_KEY", "  ")
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        OpenRouterChat()

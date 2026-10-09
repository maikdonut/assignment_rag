from schemas import RetrievedChunk

SYSTEM_PROMPT = (
    "Отвечай только по нумерованным фрагментам контекста. "
    "Если контекста недостаточно, скажи, что не знаешь. "
    "Не используй факты вне контекста. "
    "В ответе укажи source_name и chunk_id фрагментов, на которые опираешься."
)

UNGROUNDED_SYSTEM_PROMPT = (
    "Ответь на вопрос своими знаниями, без документов и без базы знаний. "
    "Не ссылайся на источники."
)


def format_context(chunks: list[RetrievedChunk]) -> str:
    """Собирает нумерованный контекст: номер, источник, идентификатор чанка и текст."""
    blocks: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        blocks.append(
            f"[{index}] source_name={chunk.source_name} chunk_id={chunk.chunk_id}\n{chunk.text}"
        )
    return "\n\n".join(blocks)


def build_user_message(question: str, chunks: list[RetrievedChunk]) -> str:
    """Собирает пользовательское сообщение: вопрос и нумерованный контекст."""
    return f"Вопрос: {question.strip()}\n\nКонтекст:\n{format_context(chunks)}"

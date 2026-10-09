"""Оценка поиска: Hit@k, MRR и отказы по группам вопросов, без вызова LLM."""

import argparse
import json
import logging
from pathlib import Path

from pydantic import BaseModel, TypeAdapter, ValidationError
from weaviate.client import WeaviateClient
from weaviate.exceptions import WeaviateBaseError

from chunking import chunk_documents
from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    KNOWLEDGE_BASE_DIR,
    PROJECT_ROOT,
    RELEVANCE_THRESHOLD,
    TOP_K,
)
from embedder import embed_texts, resolve_device
from loader import load_documents
from rag_pipeline import is_context_sufficient
from retriever import search
from schemas import RetrievedChunk
from weaviate_store import connect, replace_chunks

logger = logging.getLogger(__name__)

EVAL_COLLECTION = "KnowledgeChunkEval"
GROUP_IN_SCOPE = "свои"
GROUP_PARAPHRASE = "свои, перефразированные"
GROUP_OUT_OF_SCOPE = "чужие"
GROUP_BORDERLINE = "пограничные"
SWEEP_THRESHOLDS: tuple[float, ...] = (0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75)

_DATA_DIR = PROJECT_ROOT / "tests" / "data"
_IN_SCOPE_PATH = _DATA_DIR / "retrieval_questions.json"
_PARAPHRASE_PATH = _DATA_DIR / "paraphrase_questions.json"
_OUT_OF_SCOPE_PATH = _DATA_DIR / "out_of_scope_questions.json"
_BORDERLINE_PATH = _DATA_DIR / "borderline_questions.json"
_EMPTY = "—"


class SourcedQuestion(BaseModel):
    """Вопрос, у которого в базе есть ожидаемый файл."""

    question: str
    source_name: str


class BareQuestion(BaseModel):
    """Вопрос без ожидаемого файла."""

    question: str


class EvalCase(BaseModel):
    """Один вопрос прогона с группой и, для своих, ожидаемым файлом."""

    group: str
    question: str
    source_name: str | None = None


class QuestionScore(BaseModel):
    """Результат одного поиска: вопрос и найденные чанки."""

    group: str
    question: str
    source_name: str | None
    chunks: list[RetrievedChunk]


class Ratio(BaseModel):
    """Сколько случаев из общего числа попали в долю."""

    count: int
    total: int

    @property
    def rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.count / self.total


class SweepRow(BaseModel):
    """Доли отказов на одном пороге по уже найденным чанкам."""

    threshold: float
    false_refusals: Ratio
    correct_refusals: Ratio
    borderline_refusals: Ratio


def collection_names_conflict(eval_name: str, working_name: str) -> bool:
    """Сообщает, что временная коллекция называется так же, как рабочая."""
    return eval_name == working_name


def source_rank(chunks: list[RetrievedChunk], source_name: str) -> int | None:
    """Возвращает позицию первого чанка с этим файлом, начиная с 1."""
    for index, chunk in enumerate(chunks, start=1):
        if chunk.source_name == source_name:
            return index
    return None


def is_hit(rank: int | None, top_k: int) -> bool:
    """Проверяет, что ожидаемый файл попал в первые ``top_k`` позиций."""
    return rank is not None and rank <= top_k


def mean_reciprocal_rank(ranks: list[int | None]) -> float:
    """Среднее обратных рангов. Промах даёт ноль."""
    if not ranks:
        return 0.0
    total = 0.0
    for rank in ranks:
        if rank is not None:
            total += 1.0 / rank
    return total / len(ranks)


def refusal_ratio(rows: list[QuestionScore], threshold: float) -> Ratio:
    """Считает, скольким вопросам контекста не хватило на этом пороге."""
    refused = sum(1 for row in rows if not is_context_sufficient(row.chunks, threshold))
    return Ratio(count=refused, total=len(rows))


def hit_ratio(rows: list[QuestionScore], top_k: int) -> Ratio:
    """Считает, скольким вопросам ожидаемый файл попал в top-k."""
    hits = 0
    for row in rows:
        if row.source_name is None:
            raise ValueError("для Hit@k нужен source_name")
        if is_hit(source_rank(row.chunks, row.source_name), top_k):
            hits += 1
    return Ratio(count=hits, total=len(rows))


def group_mrr(rows: list[QuestionScore]) -> float:
    """MRR группы, у каждого вопроса которой задан ожидаемый файл."""
    ranks: list[int | None] = []
    for row in rows:
        if row.source_name is None:
            raise ValueError("для MRR нужен source_name")
        ranks.append(source_rank(row.chunks, row.source_name))
    return mean_reciprocal_rank(ranks)


def rows_of(rows: list[QuestionScore], group: str) -> list[QuestionScore]:
    """Оставляет вопросы одной группы."""
    return [row for row in rows if row.group == group]


def sweep_rates(rows: list[QuestionScore]) -> list[SweepRow]:
    """Считает доли отказов на фиксированных порогах по уже найденным чанкам."""
    in_scope = rows_of(rows, GROUP_IN_SCOPE)
    out_of_scope = rows_of(rows, GROUP_OUT_OF_SCOPE)
    borderline = rows_of(rows, GROUP_BORDERLINE)
    return [
        SweepRow(
            threshold=threshold,
            false_refusals=refusal_ratio(in_scope, threshold),
            correct_refusals=refusal_ratio(out_of_scope, threshold),
            borderline_refusals=refusal_ratio(borderline, threshold),
        )
        for threshold in SWEEP_THRESHOLDS
    ]


def load_cases() -> list[EvalCase]:
    """Читает четыре файла вопросов. Перефразы и пограничные идут отдельными группами."""
    cases: list[EvalCase] = []
    cases.extend(_load_sourced(_IN_SCOPE_PATH, GROUP_IN_SCOPE))
    cases.extend(_load_sourced(_PARAPHRASE_PATH, GROUP_PARAPHRASE))
    cases.extend(_load_bare(_OUT_OF_SCOPE_PATH, GROUP_OUT_OF_SCOPE))
    cases.extend(_load_bare(_BORDERLINE_PATH, GROUP_BORDERLINE))
    return cases


def score_cases(
    client: WeaviateClient,
    collection_name: str,
    cases: list[EvalCase],
    top_k: int,
) -> list[QuestionScore]:
    """Ищет каждый вопрос один раз и сохраняет чанки для метрик и sweep."""
    rows: list[QuestionScore] = []
    for case in cases:
        chunks = search(client, case.question, top_k=top_k, collection_name=collection_name)
        rows.append(
            QuestionScore(
                group=case.group,
                question=case.question,
                source_name=case.source_name,
                chunks=chunks,
            )
        )
    return rows


def delete_eval_collection(
    client: WeaviateClient,
    eval_name: str,
    working_name: str,
) -> None:
    """Удаляет временную коллекцию. Рабочую коллекцию не трогает."""
    if collection_names_conflict(eval_name, working_name):
        raise RuntimeError("временная коллекция совпадает с рабочей")
    if client.collections.exists(eval_name):
        client.collections.delete(eval_name)
        logger.info("удалена временная коллекция: %s", eval_name)


def build_eval_index(client: WeaviateClient, chunk_size: int, overlap: int) -> None:
    """Нарезает базу заново и записывает эмбеддинги во временную коллекцию."""
    documents = load_documents(KNOWLEDGE_BASE_DIR)
    chunks = chunk_documents(documents, chunk_size=chunk_size, overlap=overlap)
    device = resolve_device()
    vectors = embed_texts([chunk.text for chunk in chunks], device=device)
    replace_chunks(client, chunks, vectors, EVAL_COLLECTION)


def main(argv: list[str] | None = None) -> None:
    """Печатает таблицу метрик поиска. Плохие метрики код выхода не меняют."""
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    for logger_name in ("httpx", "httpcore", "openai", "huggingface_hub", "sentence_transformers"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(description="Оценка поиска по базе знаний")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE, help="Размер чанка")
    parser.add_argument("--overlap", type=int, default=CHUNK_OVERLAP, help="Перекрытие чанков")
    parser.add_argument("--top-k", type=int, default=TOP_K, help="Сколько чанков брать из поиска")
    parser.add_argument("--threshold", type=float, default=RELEVANCE_THRESHOLD, help="Порог релевантности")
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Таблица порогов 0.40–0.75 по уже найденным чанкам",
    )
    args = parser.parse_args(argv)
    try:
        cases = load_cases()
    except ValueError as exc:
        print(exc)
        raise SystemExit(1) from None
    rows = _collect_rows(cases, args.chunk_size, args.overlap, args.top_k)
    _print_report(
        rows,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        top_k=args.top_k,
        threshold=args.threshold,
        sweep=args.sweep,
    )


def _collect_rows(
    cases: list[EvalCase],
    chunk_size: int,
    overlap: int,
    top_k: int,
) -> list[QuestionScore]:
    """Подключается к Weaviate, при других настройках строит временный индекс и ищет."""
    if collection_names_conflict(EVAL_COLLECTION, COLLECTION_NAME):
        print("Имя временной коллекции совпадает с рабочей.")
        raise SystemExit(1)
    try:
        client = connect()
    except (ConnectionError, OSError):
        raise SystemExit(1) from None
    cleanup_failed = False
    rows: list[QuestionScore] | None = None
    try:
        delete_eval_collection(client, EVAL_COLLECTION, COLLECTION_NAME)
        collection_name = COLLECTION_NAME
        if chunk_size != CHUNK_SIZE or overlap != CHUNK_OVERLAP:
            build_eval_index(client, chunk_size, overlap)
            collection_name = EVAL_COLLECTION
        rows = score_cases(client, collection_name, cases, top_k)
    except (ValueError, LookupError, RuntimeError, OSError, WeaviateBaseError) as exc:
        print(exc)
        raise SystemExit(1) from None
    finally:
        try:
            delete_eval_collection(client, EVAL_COLLECTION, COLLECTION_NAME)
        except (OSError, RuntimeError, WeaviateBaseError):
            logger.exception("не удалось удалить временную коллекцию: %s", EVAL_COLLECTION)
            cleanup_failed = True
        finally:
            client.close()
    if cleanup_failed:
        print("Не удалось удалить временную коллекцию.")
        raise SystemExit(1)
    if rows is None:
        raise SystemExit(1)
    return rows


def _print_report(
    rows: list[QuestionScore],
    *,
    chunk_size: int,
    overlap: int,
    top_k: int,
    threshold: float,
    sweep: bool,
) -> None:
    """Печатает параметры, таблицу вопросов, итог и при флаге таблицу порогов."""
    collection = (
        EVAL_COLLECTION
        if chunk_size != CHUNK_SIZE or overlap != CHUNK_OVERLAP
        else COLLECTION_NAME
    )
    print(
        f"Параметры: chunk_size={chunk_size} overlap={overlap} "
        f"top_k={top_k} порог={threshold:.2f} коллекция={collection}"
    )
    print(_question_table(rows, top_k, threshold))
    print("Итог")
    print(_summary(rows, top_k, threshold))
    if sweep:
        print("Пороги")
        print(_sweep_table(sweep_rates(rows)))


def _question_table(rows: list[QuestionScore], top_k: int, threshold: float) -> str:
    headers = ["группа", "вопрос", "файл", "ранг", "hit", "score", "лучший файл", "отказ"]
    body = [_question_cells(row, top_k, threshold) for row in rows]
    return _format_table(headers, body)


def _question_cells(row: QuestionScore, top_k: int, threshold: float) -> list[str]:
    best = max(row.chunks, key=lambda chunk: chunk.score) if row.chunks else None
    if row.source_name is None:
        file_cell = _EMPTY
        rank_cell = _EMPTY
        hit_cell = _EMPTY
    else:
        rank = source_rank(row.chunks, row.source_name)
        file_cell = row.source_name
        rank_cell = _EMPTY if rank is None else str(rank)
        hit_cell = "да" if is_hit(rank, top_k) else "нет"
    return [
        row.group,
        row.question,
        file_cell,
        rank_cell,
        hit_cell,
        _EMPTY if best is None else f"{best.score:.4f}",
        _EMPTY if best is None else best.source_name,
        "да" if not is_context_sufficient(row.chunks, threshold) else "нет",
    ]


def _summary(rows: list[QuestionScore], top_k: int, threshold: float) -> str:
    in_scope = rows_of(rows, GROUP_IN_SCOPE)
    paraphrases = rows_of(rows, GROUP_PARAPHRASE)
    lines = [
        _metric_line(f"Hit@{top_k} ({GROUP_IN_SCOPE})", hit_ratio(in_scope, top_k)),
        f"MRR ({GROUP_IN_SCOPE}): {group_mrr(in_scope):.3f}",
        _metric_line(f"Ложные отказы ({GROUP_IN_SCOPE})", refusal_ratio(in_scope, threshold)),
        _metric_line(f"Hit@{top_k} ({GROUP_PARAPHRASE})", hit_ratio(paraphrases, top_k)),
        f"MRR ({GROUP_PARAPHRASE}): {group_mrr(paraphrases):.3f}",
        _metric_line(
            f"Ложные отказы ({GROUP_PARAPHRASE})",
            refusal_ratio(paraphrases, threshold),
        ),
        _metric_line(
            f"Верные отказы ({GROUP_OUT_OF_SCOPE})",
            refusal_ratio(rows_of(rows, GROUP_OUT_OF_SCOPE), threshold),
        ),
        _metric_line(
            f"Отказы ({GROUP_BORDERLINE})",
            refusal_ratio(rows_of(rows, GROUP_BORDERLINE), threshold),
        ),
    ]
    return "\n".join(lines)


def _metric_line(label: str, ratio: Ratio) -> str:
    return f"{label}: {_format_ratio(ratio)}"


def _format_ratio(ratio: Ratio) -> str:
    return f"{ratio.rate:.3f} ({ratio.count}/{ratio.total})"


def _sweep_table(rates: list[SweepRow]) -> str:
    headers = [
        "порог",
        f"ложные отказы ({GROUP_IN_SCOPE})",
        f"верные отказы ({GROUP_OUT_OF_SCOPE})",
        f"отказы ({GROUP_BORDERLINE})",
    ]
    body = [
        [
            f"{row.threshold:.2f}",
            _format_ratio(row.false_refusals),
            _format_ratio(row.correct_refusals),
            _format_ratio(row.borderline_refusals),
        ]
        for row in rates
    ]
    return _format_table(headers, body)


def _format_table(headers: list[str], body: list[list[str]]) -> str:
    widths = [len(header) for header in headers]
    for row in body:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    def format_row(cells: list[str]) -> str:
        return "  ".join(cell.ljust(widths[index]) for index, cell in enumerate(cells))

    lines = [format_row(headers), format_row(["-" * width for width in widths])]
    lines.extend(format_row(row) for row in body)
    return "\n".join(lines)


def _load_sourced(path: Path, group: str) -> list[EvalCase]:
    try:
        items = TypeAdapter(list[SourcedQuestion]).validate_python(_read_json(path))
    except ValidationError as exc:
        raise ValueError(f"битый JSON: {path.name}") from exc
    cases: list[EvalCase] = []
    for item in items:
        question = item.question.strip()
        source_name = item.source_name.strip()
        if not question:
            raise ValueError("вопрос пустой")
        if not source_name:
            raise ValueError("не задан source_name")
        cases.append(EvalCase(group=group, question=question, source_name=source_name))
    return cases


def _load_bare(path: Path, group: str) -> list[EvalCase]:
    try:
        items = TypeAdapter(list[BareQuestion]).validate_python(_read_json(path))
    except ValidationError as exc:
        raise ValueError(f"битый JSON: {path.name}") from exc
    cases: list[EvalCase] = []
    for item in items:
        question = item.question.strip()
        if not question:
            raise ValueError("вопрос пустой")
        cases.append(EvalCase(group=group, question=question, source_name=None))
    return cases


def _read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"не найден файл вопросов: {path.name}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"битый JSON: {path.name}") from exc


if __name__ == "__main__":
    main()

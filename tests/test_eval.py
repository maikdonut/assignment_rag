from pathlib import Path
from typing import cast

import pytest
from weaviate.client import WeaviateClient

from config import COLLECTION_NAME
from eval import (
    EVAL_COLLECTION,
    GROUP_BORDERLINE,
    GROUP_IN_SCOPE,
    GROUP_OUT_OF_SCOPE,
    GROUP_PARAPHRASE,
    SWEEP_THRESHOLDS,
    EvalCase,
    QuestionScore,
    _load_bare,
    collection_names_conflict,
    delete_eval_collection,
    group_mrr,
    hit_ratio,
    is_hit,
    load_cases,
    main,
    mean_reciprocal_rank,
    refusal_ratio,
    rows_of,
    score_cases,
    source_rank,
    sweep_rates,
)
from schemas import RetrievedChunk


class _Collections:
    def __init__(self, present: set[str] | None = None) -> None:
        self.present = set(present or set())
        self.deleted: list[str] = []
        self.exists_calls: list[str] = []

    def exists(self, name: str) -> bool:
        self.exists_calls.append(name)
        return name in self.present

    def delete(self, name: str) -> None:
        self.deleted.append(name)
        self.present.discard(name)


class _Client:
    def __init__(self, present: set[str] | None = None) -> None:
        self.collections = _Collections(present)


def _chunk(source_name: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        document_id=source_name.removesuffix(".txt"),
        source_name=source_name,
        chunk_id=f"{source_name}:0",
        text="текст",
        distance=1.0 - score,
        score=score,
    )


def _row(
    group: str,
    source_name: str | None,
    *scores: float,
    sources: list[str] | None = None,
) -> QuestionScore:
    names = sources if sources is not None else [source_name or "other.txt"] * len(scores)
    chunks = [_chunk(name, score) for name, score in zip(names, scores, strict=True)]
    return QuestionScore(group=group, question="вопрос", source_name=source_name, chunks=chunks)


def test_source_rank_mrr_and_hit() -> None:
    chunks = [_chunk("other.txt", 0.9), _chunk("hours.txt", 0.8), _chunk("hours.txt", 0.7)]
    assert source_rank(chunks, "hours.txt") == 2
    assert source_rank(chunks, "missing.txt") is None
    assert is_hit(2, 5) is True
    assert is_hit(5, 5) is True
    assert is_hit(6, 5) is False
    assert is_hit(None, 5) is False
    assert mean_reciprocal_rank([1, 2, None]) == pytest.approx((1.0 + 0.5) / 3)
    assert mean_reciprocal_rank([]) == 0.0


def test_missed_source_has_zero_reciprocal_rank() -> None:
    row = _row(GROUP_IN_SCOPE, "hours.txt", 0.9, sources=["other.txt"])
    assert hit_ratio([row], 5).count == 0
    assert group_mrr([row]) == 0.0


def test_empty_retrieval_is_a_refusal() -> None:
    row = QuestionScore(group=GROUP_IN_SCOPE, question="вопрос", source_name="hours.txt", chunks=[])
    assert refusal_ratio([row], 0.57).count == 1
    assert hit_ratio([row], 5).count == 0


def test_refusal_requires_score_strictly_below_threshold() -> None:
    row = _row(GROUP_IN_SCOPE, "hours.txt", 0.50)
    assert refusal_ratio([row], 0.50).count == 0
    assert refusal_ratio([row], 0.51).count == 1


def test_group_rates_stay_separate() -> None:
    rows = [
        _row(GROUP_IN_SCOPE, "opening_hours.txt", 0.80),
        _row(GROUP_PARAPHRASE, "membership.txt", 0.10),
        _row(GROUP_OUT_OF_SCOPE, None, 0.20),
        _row(GROUP_BORDERLINE, None, 0.90),
    ]
    assert refusal_ratio(rows_of(rows, GROUP_IN_SCOPE), 0.57).count == 0
    assert refusal_ratio(rows_of(rows, GROUP_PARAPHRASE), 0.57).count == 1
    assert refusal_ratio(rows_of(rows, GROUP_OUT_OF_SCOPE), 0.57).count == 1
    assert refusal_ratio(rows_of(rows, GROUP_BORDERLINE), 0.57).count == 0
    assert hit_ratio(rows_of(rows, GROUP_IN_SCOPE), 5).count == 1
    paraphrases = rows_of(rows, GROUP_PARAPHRASE)
    assert hit_ratio(paraphrases, 5).count == 1
    assert group_mrr(paraphrases) == pytest.approx(1.0)


def test_sweep_reuses_saved_chunks() -> None:
    rows = [
        _row(GROUP_IN_SCOPE, "a.txt", 0.52),
        _row(GROUP_OUT_OF_SCOPE, None, 0.48),
        _row(GROUP_BORDERLINE, None, 0.70),
        _row(GROUP_PARAPHRASE, "b.txt", 0.10),
    ]
    table = sweep_rates(rows)
    assert [row.threshold for row in table] == list(SWEEP_THRESHOLDS)
    by_threshold = {row.threshold: row for row in table}
    assert by_threshold[0.40].false_refusals.count == 0
    assert by_threshold[0.40].correct_refusals.count == 0
    assert by_threshold[0.40].borderline_refusals.count == 0
    assert by_threshold[0.50].false_refusals.count == 0
    assert by_threshold[0.50].correct_refusals.count == 1
    assert by_threshold[0.50].borderline_refusals.count == 0
    assert by_threshold[0.55].false_refusals.count == 1
    assert by_threshold[0.55].correct_refusals.count == 1
    assert by_threshold[0.55].borderline_refusals.count == 0
    assert by_threshold[0.75].false_refusals.count == 1
    assert by_threshold[0.75].correct_refusals.count == 1
    assert by_threshold[0.75].borderline_refusals.count == 1
    assert by_threshold[0.40].false_refusals.total == 1


def test_score_cases_searches_once_per_question(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_search(client: object, query: str, **kwargs: object) -> list[RetrievedChunk]:
        calls.append(query)
        assert kwargs["top_k"] == 5
        assert kwargs["collection_name"] == COLLECTION_NAME
        return [_chunk("hours.txt", 0.8)]

    monkeypatch.setattr("eval.search", fake_search)
    cases = [
        EvalCase(group=GROUP_IN_SCOPE, question="один", source_name="hours.txt"),
        EvalCase(group=GROUP_OUT_OF_SCOPE, question="два"),
    ]
    rows = score_cases(cast(WeaviateClient, object()), COLLECTION_NAME, cases, top_k=5)
    assert calls == ["один", "два"]
    sweep_rates(rows)
    assert calls == ["один", "два"]


def test_question_files_split_groups() -> None:
    cases = load_cases()
    counts = {
        GROUP_IN_SCOPE: 6,
        GROUP_PARAPHRASE: 7,
        GROUP_OUT_OF_SCOPE: 5,
        GROUP_BORDERLINE: 5,
    }
    for group, expected in counts.items():
        assert sum(case.group == group for case in cases) == expected
    for case in cases:
        if case.group in {GROUP_IN_SCOPE, GROUP_PARAPHRASE}:
            assert case.source_name
        else:
            assert case.source_name is None


def test_load_rejects_broken_json(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="битый JSON"):
        _load_bare(path, GROUP_OUT_OF_SCOPE)


def test_load_rejects_blank_question(tmp_path: Path) -> None:
    path = tmp_path / "blank.json"
    path.write_text('[{"question": "  "}]', encoding="utf-8")
    with pytest.raises(ValueError, match="вопрос пустой"):
        _load_bare(path, GROUP_BORDERLINE)


def test_delete_eval_collection_removes_leftover() -> None:
    client = _Client({EVAL_COLLECTION})
    delete_eval_collection(cast(WeaviateClient, client), EVAL_COLLECTION, COLLECTION_NAME)
    assert client.collections.deleted == [EVAL_COLLECTION]


def test_delete_eval_collection_skips_missing() -> None:
    client = _Client()
    delete_eval_collection(cast(WeaviateClient, client), EVAL_COLLECTION, COLLECTION_NAME)
    assert client.collections.deleted == []


def test_delete_refuses_working_collection_name() -> None:
    client = _Client({COLLECTION_NAME})
    with pytest.raises(RuntimeError, match="совпадает"):
        delete_eval_collection(cast(WeaviateClient, client), COLLECTION_NAME, COLLECTION_NAME)
    assert client.collections.exists_calls == []


def test_conflicting_collection_name_is_detected() -> None:
    assert collection_names_conflict(COLLECTION_NAME, COLLECTION_NAME) is True
    assert collection_names_conflict(EVAL_COLLECTION, COLLECTION_NAME) is False


def test_main_exits_when_questions_are_invalid(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def _boom() -> list[EvalCase]:
        raise ValueError("битый JSON: x.json")

    monkeypatch.setattr("eval.load_cases", _boom)
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 1
    assert "битый JSON" in capsys.readouterr().out


def test_main_refuses_matching_collection_name(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr("eval.load_cases", lambda: [])

    def fail_connect() -> WeaviateClient:
        raise AssertionError("подключение не должно начинаться")

    monkeypatch.setattr("eval.EVAL_COLLECTION", COLLECTION_NAME)
    monkeypatch.setattr("eval.connect", fail_connect)
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 1
    assert "совпадает" in capsys.readouterr().out

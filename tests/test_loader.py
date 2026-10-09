from pathlib import Path

import pytest

from config import KNOWLEDGE_BASE_DIR
from loader import clean_text, load_documents


def test_clean_text_normalizes_whitespace_and_blank_lines() -> None:
    raw = "  Hello   world  \r\n\r\n\n  Second\tline  \r\n"
    assert clean_text(raw) == "Hello world\nSecond line"


def test_load_documents_sorts_and_cleans(tmp_path: Path) -> None:
    (tmp_path / "b_rules.txt").write_text("  Rule   one  \n\n  Rule two  \n", encoding="utf-8")
    (tmp_path / "a_hours.txt").write_text("Open  at   nine\n", encoding="utf-8")
    (tmp_path / "notes.md").write_text("ignore me\n", encoding="utf-8")

    documents = load_documents(tmp_path)

    assert [document.source_name for document in documents] == ["a_hours.txt", "b_rules.txt"]
    assert documents[0].document_id == "a_hours"
    assert documents[0].text == "Open at nine"
    assert documents[1].document_id == "b_rules"
    assert documents[1].text == "Rule one\nRule two"


def test_load_documents_skips_empty_after_cleaning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    (tmp_path / "empty.txt").write_text("   \n\n  \n", encoding="utf-8")
    (tmp_path / "kept.txt").write_text("Kept text\n", encoding="utf-8")

    with caplog.at_level("WARNING"):
        documents = load_documents(tmp_path)

    assert [document.source_name for document in documents] == ["kept.txt"]
    assert "empty.txt" in caplog.text


def test_knowledge_base_loads_at_least_five_documents() -> None:
    paths = sorted(KNOWLEDGE_BASE_DIR.glob("*.txt"))
    assert len(paths) >= 5

    documents = load_documents(KNOWLEDGE_BASE_DIR)

    assert len(documents) == len(paths)
    assert [document.source_name for document in documents] == [path.name for path in paths]
    assert all(document.text.strip() for document in documents)
    assert all(document.document_id == Path(document.source_name).stem for document in documents)

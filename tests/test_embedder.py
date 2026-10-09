import pytest
import torch

from embedder import embed_texts, resolve_device


class _FakeModel:
    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    def encode(self, texts: list[str], **kwargs: object) -> list[list[float]]:
        self.batches.append(list(texts))
        return [[float(len(text)), 0.0] for text in texts]


def test_resolve_device_uses_cpu_when_cuda_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device() == "cpu"


def test_resolve_device_uses_cuda_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device() == "cuda"


def test_embed_texts_preserves_order_and_batches() -> None:
    model = _FakeModel()
    texts = ["a", "bb", "ccc", "dddd"]

    vectors = embed_texts(texts, model=model, batch_size=3)

    assert vectors == [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]]
    assert model.batches == [["a", "bb", "ccc"], ["dddd"]]


def test_embed_texts_empty_skips_model() -> None:
    model = _FakeModel()

    assert embed_texts([], model=model, batch_size=2) == []
    assert model.batches == []


def test_embed_texts_rejects_non_positive_batch() -> None:
    with pytest.raises(ValueError):
        embed_texts(["текст"], model=_FakeModel(), batch_size=0)

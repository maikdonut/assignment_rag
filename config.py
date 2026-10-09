import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "knowledge_base"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

COLLECTION_NAME = "KnowledgeChunk"
EMBEDDING_MODEL = "deepvk/USER2-base"
EMBEDDING_BATCH_SIZE = 16


def _load_env_file(path: Path) -> None:
    """Подставляет переменные из файла ``.env``, не перезаписывая уже заданные."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped.removeprefix("export ").strip()
        if "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value


_load_env_file(PROJECT_ROOT / ".env")

WEAVIATE_HOST = os.environ.get("WEAVIATE_HOST", "localhost")
WEAVIATE_HTTP_PORT = int(os.environ.get("WEAVIATE_HTTP_PORT", "8080"))
WEAVIATE_GRPC_PORT = int(os.environ.get("WEAVIATE_GRPC_PORT", "50051"))

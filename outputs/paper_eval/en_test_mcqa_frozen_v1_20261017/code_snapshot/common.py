from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "paper_data"
CONDITIONS = ("base", "base_rag", "lora", "lora_rag")
CONDITION_NAMES = {"base": "基座", "base_rag": "基座＋RAG", "lora": "微调", "lora_rag": "微调＋RAG"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def digest_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def normalize(text: str) -> str:
    return re.sub(r"[^\w\u4e00-\u9fff]", "", text.casefold())


def load_protocol(path: Path | None = None) -> dict[str, Any]:
    return json.loads((path or ROOT / "paper_eval/protocol.json").read_text(encoding="utf-8"))


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def validate_questions(rows: list[dict[str, Any]]) -> None:
    ids, prompts = set(), set()
    for row in rows:
        for field in ("id", "source_group", "task", "mode", "question", "key_points", "evidence"):
            if not row.get(field):
                raise ValueError(f"Missing {field}: {row.get('id')}")
        if row["id"] in ids or normalize(row["question"]) in prompts:
            raise ValueError(f"Duplicate id/question: {row['id']}")
        if row["mode"] not in {"professional", "popular"}:
            raise ValueError("Unknown mode")
        if not all(isinstance(p, str) and p.strip() for p in row["key_points"]):
            raise ValueError(f"Key points need explicit reference statements: {row['id']}")
        ids.add(row["id"])
        prompts.add(normalize(row["question"]))


def artifact_signature(protocol: dict[str, Any], dataset: Path, corpus: Path, model_files: list[Path]) -> dict[str, Any]:
    signature = {
        "protocol": protocol,
        "dataset_sha256": digest_file(dataset),
        "corpus_sha256": digest_file(corpus),
        "model_files": {str(p.relative_to(ROOT)): digest_file(p) for p in model_files},
    }
    return {**signature, "signature_sha256": digest_json(signature)}

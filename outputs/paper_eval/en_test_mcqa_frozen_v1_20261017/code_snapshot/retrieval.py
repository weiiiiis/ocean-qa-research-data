from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from .common import digest_file, digest_json, read_jsonl


class Retriever:
    """Local-only multilingual retrieval. Never indexes questions or reference answers."""

    def __init__(self, corpus: Path, model_path: Path, cache_dir: Path):
        self.corpus_path, self.model_path = corpus, model_path
        self.chunks = read_jsonl(corpus)
        if not self.chunks or any(set(row) - {"id", "text", "source", "pdf_page", "section", "license", "url"} for row in self.chunks):
            raise ValueError("Corpus must contain source passages only, never evaluation labels")
        if not model_path.is_dir():
            raise FileNotFoundError(f"Local embedding model missing: {model_path}")
        from transformers import AutoModel, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
        self.model = AutoModel.from_pretrained(str(model_path), local_files_only=True).to("cpu").eval()
        fingerprint = digest_json({"corpus": digest_file(corpus), "model_files": {p.name: digest_file(p) for p in sorted(model_path.iterdir()) if p.suffix in {".safetensors", ".bin", ".json", ".txt"}}, "max_tokens": 384})
        cache_dir.mkdir(parents=True, exist_ok=True)
        target = cache_dir / f"embeddings_{fingerprint}.npz"
        if target.exists():
            self.matrix = np.load(target)["matrix"]
            if self.matrix.shape[0] != len(self.chunks):
                raise ValueError("Embedding cache/corpus mismatch")
        else:
            self.matrix = self.encode([row["text"] for row in self.chunks])
            np.savez_compressed(target, matrix=self.matrix)
        self.encode(["ocean science evaluation warmup"])

    def encode(self, texts: list[str], batch_size: int = 24) -> np.ndarray:
        import torch
        vectors = []
        for start in range(0, len(texts), batch_size):
            encoded = self.tokenizer(texts[start:start + batch_size], padding=True, truncation=True, max_length=384, return_tensors="pt")
            with torch.inference_mode():
                hidden = self.model(**encoded).last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            vectors.append(pooled.numpy().astype("float32"))
        return np.vstack(vectors)

    def search(self, question: str, top_k: int, min_score: float) -> tuple[list[dict[str, Any]], float]:
        started = time.perf_counter()
        scores = (self.matrix @ self.encode([question]).T).ravel()
        # Stable ties make cached and recomputed contexts reproducible.
        order = np.argsort(-scores, kind="stable")[:top_k]
        hits = [{**self.chunks[int(i)], "score": float(scores[int(i)])} for i in order if scores[int(i)] >= min_score]
        return hits, (time.perf_counter() - started) * 1000

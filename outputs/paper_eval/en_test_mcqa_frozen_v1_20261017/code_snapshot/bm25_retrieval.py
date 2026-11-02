"""Source-only English BM25 retrieval with deterministic source deduplication.

The configuration exactly matches the completed DEVELOPMENT diagnosis:
sklearn lowercase English word analysis and stop words, k1=1.5, b=0.75,
positive IDF, binary query term frequency, and highest-ranked chunk per source.
No model, embedding weights, evaluation labels or gold source IDs are read.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

from .common import digest_file, digest_json, read_jsonl


K1 = 1.5
B = 0.75
LEXICAL_SETTINGS = {"lowercase": True, "stop_words": "english", "token_pattern": r"(?u)\b\w\w+\b"}
ALLOWED_CORPUS_FIELDS = {"id", "text", "source", "pdf_page", "section", "license", "url"}


class Retriever:
    """Drop-in search API for the frozen English evaluator.

    ``model_path`` and ``cache_dir`` are accepted for call compatibility, but
    BM25 reads no embedding model and writes no embedding cache. Scores are BM25
    relevance scores, not cosine similarities; the recommended minimum is 0.0.
    Zero-score documents are always excluded because they match no query terms.
    """

    def __init__(self, corpus: Path, model_path: Path | None = None, cache_dir: Path | None = None):
        self.corpus_path = Path(corpus)
        self.model_path = model_path
        self.chunks = read_jsonl(self.corpus_path)
        if not self.chunks or any(set(row) - ALLOWED_CORPUS_FIELDS for row in self.chunks):
            raise ValueError("BM25 corpus must contain source passages only, never evaluation labels")
        required = ("id", "text", "section")
        if any(any(not isinstance(row.get(key), str) or not row[key].strip() for key in required) for row in self.chunks):
            raise ValueError("Every source chunk needs nonempty id, text and section")
        if len({row["id"] for row in self.chunks}) != len(self.chunks):
            raise ValueError("Duplicate source chunk IDs")
        self.vectorizer = CountVectorizer(**LEXICAL_SETTINGS)
        counts = self.vectorizer.fit_transform([row["text"] for row in self.chunks])
        counts = sparse.csr_matrix(counts, dtype=np.float64)
        lengths = np.asarray(counts.sum(axis=1)).ravel()
        self.average_document_length = float(np.mean(lengths))
        document_frequency = np.bincount(counts.indices, minlength=counts.shape[1])
        idf = np.log1p((counts.shape[0] - document_frequency + 0.5) / (document_frequency + 0.5))
        self.matrix = counts.copy()
        norms = K1 * (1 - B + B * lengths / max(self.average_document_length, 1e-12))
        for row in range(self.matrix.shape[0]):
            start, end = self.matrix.indptr[row:row + 2]
            frequency = self.matrix.data[start:end]
            self.matrix.data[start:end] = ((frequency * (K1 + 1) / (frequency + norms[row]))
                                          * idf[self.matrix.indices[start:end]])
        self.implementation_metadata = {
            "method": "bm25", "implementation_version": "ocean-bm25-source-unique-0.1",
            "implementation_sha256": digest_file(Path(__file__)),
            "corpus_sha256": digest_file(self.corpus_path),
            "lexical_settings": dict(LEXICAL_SETTINGS), "k1": K1, "b": B,
            "idf": "log(1+(N-df+0.5)/(df+0.5))", "query_term_frequency": "binary",
            "source_deduplication": "highest ranked chunk per exact section identity",
            "tie_breaking": "stable descending score then corpus order",
            "zero_score_documents": "excluded", "embedding_model_loaded": False,
            "average_document_length": self.average_document_length,
        }
        self.implementation_metadata["fingerprint_sha256"] = digest_json(self.implementation_metadata)

    def search(self, question: str, top_k: int, min_score: float) -> tuple[list[dict[str, Any]], float]:
        if not isinstance(question, str):
            raise TypeError("Retrieval query must be a question string")
        if not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        if not np.isfinite(min_score) or min_score < 0:
            raise ValueError("BM25 min_score must be finite and nonnegative")
        started = time.perf_counter()
        query = self.vectorizer.transform([question])
        query.data[:] = 1
        scores = (self.matrix @ query.T).toarray().ravel()
        order = np.argsort(-scores, kind="stable")
        selected, seen = [], set()
        for value in order:
            index = int(value)
            score = float(scores[index])
            if score <= 0 or score < min_score:
                # Sorted order guarantees no later candidate can pass.
                break
            chunk = self.chunks[index]
            if chunk["section"] in seen:
                continue
            selected.append({**chunk, "score": score})
            seen.add(chunk["section"])
            if len(selected) == top_k:
                break
        return selected, (time.perf_counter() - started) * 1000


BM25Retriever = Retriever

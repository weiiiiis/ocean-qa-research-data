"""CPU-only source-retrieval diagnosis on the English DEVELOPMENT pool.

No official test questions are loaded. Gold IDs are consulted only after the
question-only retrieval rankings and actual context budgets have been fixed.
This module never alters production retrieval or model evaluation settings.
"""
from __future__ import annotations

import os

# Set before importing NumPy / PyTorch; this can run beside GPU generation.
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "2"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import collections
import csv
import json
import math
from pathlib import Path

import numpy as np

from .common import ROOT, digest_file, read_jsonl, resolve, write_json, write_jsonl
from .english import fit_prompt
from .report import clustered_interval
from .retrieval import Retriever


METHODS = ("dense_current", "dense_source_unique", "dense_source_unique_no_floor", "word_tfidf_source_unique",
           "bm25_source_unique", "dense_bm25_rrf_source_unique")


def source_unique_order(order: list[int], chunks: list[dict], count: int) -> list[int]:
    selected, seen = [], set()
    for index in order:
        source = chunks[index]["section"]
        if source not in seen:
            selected.append(index)
            seen.add(source)
            if len(selected) == count:
                break
    return selected


def ranked(scores: np.ndarray, floor: float | None = None) -> list[int]:
    return [int(index) for index in np.argsort(-scores, kind="stable")
            if floor is None or scores[index] >= floor]


def bm25_matrix(counts, k1: float = 1.5, b: float = 0.75):
    """Positive-IDF BM25, binary query term frequency; source text only."""
    from scipy import sparse
    matrix = sparse.csr_matrix(counts, dtype=np.float64)
    document_lengths = np.asarray(matrix.sum(axis=1)).ravel()
    average_length = float(np.mean(document_lengths))
    df = np.bincount(matrix.indices, minlength=matrix.shape[1])
    idf = np.log1p((matrix.shape[0] - df + 0.5) / (df + 0.5))
    weights = matrix.copy()
    norms = k1 * (1 - b + b * document_lengths / max(average_length, 1e-12))
    for row in range(weights.shape[0]):
        start, end = weights.indptr[row:row + 2]
        frequency = weights.data[start:end]
        weights.data[start:end] = (frequency * (k1 + 1) / (frequency + norms[row])) * idf[weights.indices[start:end]]
    return weights, average_length


def label_group(question: dict) -> str:
    value = (question.get("original_assessments") or {}).get("answer_in_text")
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "answer_in_text_unlabeled"
    return "answer_in_text_1" if value == 1 else "answer_in_text_other"


def run(args: argparse.Namespace) -> None:
    import torch
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from threadpoolctl import threadpool_limits
    from transformers import AutoTokenizer

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    threadpool_limits(limits=2)
    dataset, corpus = resolve(args.dataset), resolve(args.corpus)
    questions, chunks = read_jsonl(dataset), read_jsonl(corpus)
    if not questions or any(question.get("split") not in {"dev", "development"} for question in questions):
        raise ValueError("This diagnostic accepts development records only, never official test")
    if any(not question.get("expected_source_id") for question in questions):
        raise ValueError("Missing gold source identity for offline recall evaluation")
    protocol = json.loads(resolve(args.protocol).read_text(encoding="utf-8"))
    out = resolve(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # Queries are extracted before reading any reference text/answer for scoring.
    query_texts = [question["question"] for question in questions]
    print(f"CPU-only retrieval development diagnosis: {len(questions)} questions, {len(chunks)} chunks", flush=True)
    retriever = Retriever(corpus, resolve(protocol["embedding_model"]), ROOT / "cache/paper_eval")
    query_vectors = retriever.encode(query_texts, batch_size=8)
    dense_scores = np.asarray(retriever.matrix @ query_vectors.T)
    texts = [chunk["text"] for chunk in chunks]
    lexical_settings = {"lowercase": True, "stop_words": "english", "token_pattern": r"(?u)\b\w\w+\b"}
    tfidf = TfidfVectorizer(**lexical_settings, sublinear_tf=True)
    tfidf_documents = tfidf.fit_transform(texts)
    tfidf_scores = (tfidf_documents @ tfidf.transform(query_texts).T).toarray()
    counts = CountVectorizer(**lexical_settings)
    count_documents = counts.fit_transform(texts)
    bm25, average_length = bm25_matrix(count_documents)
    count_queries = counts.transform(query_texts)
    count_queries.data[:] = 1
    bm25_scores = (bm25 @ count_queries.T).toarray()
    tokenizer = AutoTokenizer.from_pretrained(str(resolve(protocol["base_model"])), local_files_only=True)
    dense_floor = protocol["retrieval"]["min_score"]
    top_k = 3
    rrf_k, rrf_candidate_depth = 60, 100
    records = []
    for qi, question in enumerate(questions):
        dense = dense_scores[:, qi]
        lexical = tfidf_scores[:, qi]
        bm = bm25_scores[:, qi]
        dense_order = ranked(dense)
        dense_filtered = ranked(dense, dense_floor)
        # Zero lexical scores carry no evidence and are not eligible candidates.
        tfidf_order = [index for index in ranked(lexical) if lexical[index] > 0]
        bm_order = [index for index in ranked(bm) if bm[index] > 0]
        rrf = np.zeros(len(chunks), dtype=np.float64)
        for order in (dense_order[:rrf_candidate_depth], bm_order[:rrf_candidate_depth]):
            for rank, index in enumerate(order, 1):
                rrf[index] += 1 / (rrf_k + rank)
        rrf_order = [index for index in ranked(rrf) if rrf[index] > 0]
        plans = {
            "dense_current": (dense_filtered[:top_k], dense),
            "dense_source_unique": (source_unique_order(dense_filtered, chunks, top_k), dense),
            "dense_source_unique_no_floor": (source_unique_order(dense_order, chunks, top_k), dense),
            "word_tfidf_source_unique": (source_unique_order(tfidf_order, chunks, top_k), lexical),
            "bm25_source_unique": (source_unique_order(bm_order, chunks, top_k), bm),
            "dense_bm25_rrf_source_unique": (source_unique_order(rrf_order, chunks, top_k), rrf),
        }
        # At this point every ranking depends only on question text and corpus.
        gold_source = question["expected_source_id"]
        for method, (indices, scores) in plans.items():
            hits = [{**chunks[index], "score": float(scores[index])} for index in indices]
            prompt, shown = fit_prompt(tokenizer, question, hits, "openqa",
                protocol["retrieval"]["max_context_tokens"], protocol["generation"]["max_input_tokens"])
            sources = [hit["section"] for hit in hits]
            shown_sources = [hit["section"] for hit in shown if hit["text"].strip()]
            records.append({"id": question["id"], "source_group": question["source_group"], "label_group": label_group(question),
                "method": method, "expected_source_id_offline_only": gold_source,
                "source_recall_at_1": float(gold_source in sources[:1]), "source_recall_at_3": float(gold_source in sources[:3]),
                "source_recall_shown_1000_tokens": float(gold_source in shown_sources),
                "raw_hit_ids": [hit["id"] for hit in hits], "raw_source_ids": sources,
                "shown_hit_ids": [hit["id"] for hit in shown], "shown_source_ids": shown_sources,
                "selected_sources": len(set(sources)), "shown_sources": len(set(shown_sources)),
                "shown_context_tokens": sum(len(tokenizer.encode(hit["text"], add_special_tokens=False)) for hit in shown),
                "full_prompt_tokens": len(tokenizer.encode(prompt, add_special_tokens=False)),
                "any_shown_chunk_truncated": any(hit.get("context_truncated") for hit in shown)})
        if (qi + 1) % 20 == 0:
            print(f"Budgeted {qi + 1}/{len(questions)} question-only rankings", flush=True)
    write_jsonl(out / "per_question_retrieval.jsonl", records)
    metrics = ("source_recall_at_1", "source_recall_at_3", "source_recall_shown_1000_tokens")
    groups = ("all", "answer_in_text_1", "answer_in_text_other", "answer_in_text_unlabeled")
    summaries = []
    for method in METHODS:
        for group in groups:
            selected = [row for row in records if row["method"] == method and (group == "all" or row["label_group"] == group)]
            item = {"method": method, "group": group, "questions": len(selected)}
            for metric in metrics:
                item[metric] = float(np.mean([row[metric] for row in selected])) if selected else None
            item["shown_sources_mean"] = float(np.mean([row["shown_sources"] for row in selected])) if selected else None
            item["shown_context_tokens_mean"] = float(np.mean([row["shown_context_tokens"] for row in selected])) if selected else None
            summaries.append(item)
    with (out / "summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    indexed = {(row["method"], row["id"]): row for row in records}
    differences = {}
    for method in METHODS[1:]:
        differences[method + "_minus_dense_current"] = {metric: clustered_interval([
            (question["source_group"], indexed[(method, question["id"])][metric] - indexed[("dense_current", question["id"])][metric])
            for question in questions]) for metric in metrics}
    manifest = {"development_only": True, "dataset_path": str(dataset), "dataset_sha256": digest_file(dataset),
        "corpus_path": str(corpus), "corpus_sha256": digest_file(corpus), "code_sha256": digest_file(Path(__file__)),
        "protocol_sha256": digest_file(resolve(args.protocol)), "questions": len(questions), "corpus_chunks": len(chunks),
        "source_documents": len({chunk["section"] for chunk in chunks}), "cpu_threads_maximum": 2, "gpu_used": False,
        "query_policy": "Question stem only; never choices, answers, expected source identities or reference passages.",
        "gold_policy": "Gold source identity is used only for offline recall AFTER ranking and context budgets are fixed.",
        "lexical_preprocessing": lexical_settings, "word_tfidf_sublinear_tf": True,
        "bm25": {"k1": 1.5, "b": 0.75, "idf": "log(1+(N-df+0.5)/(df+0.5))", "query_term_frequency": "binary", "average_document_length": average_length},
        "rrf": {"k": rrf_k, "dense_weight": 1, "bm25_weight": 1, "candidate_depth_per_ranker": rrf_candidate_depth,
                "dense_minimum_score": None, "bm25_excludes_zero_scores": True, "deduplication": "highest ranked chunk per source after chunk-level fusion"},
        "context": {"top_k": top_k, "max_context_tokens_including_headers": protocol["retrieval"]["max_context_tokens"],
                    "max_input_tokens": protocol["generation"]["max_input_tokens"], "budgeting_code": "paper_eval.english.fit_prompt"},
        "timing_boundary": "CPU diagnosis alongside possible GPU workload; no timings are reported as formal performance.",
        "limitations": ["This tunes development retrieval, not the held-out official test.",
            "Source identity recall does not prove a relevant span survived truncation, entailment or correct generation.",
            "Source deduplication favors source coverage and may discard useful second windows from a long source.",
            "The six method configurations are fixed in this diagnostic; the best development result still needs independent test confirmation.",
            "Published answer_in_text labels are not new independent expert validation."]}
    write_json(out / "analysis.json", {"manifest": manifest, "summary": summaries, "paired_source_cluster_bootstrap_differences": differences})
    lines = ["# 英文开发集检索诊断", "", f"{len(questions)}道开发题；{len(chunks)}个原文窗口；{manifest['source_documents']}个支撑文本来源。所有检索只输入问题题干。", "",
        "Gold来源编号仅在排名和上下文预算确定后用于离线计分，不作为检索提示。未运行正式测试集、未改生产检索器。", "",
        "## 全部开发题", "", "| 方法 | 题数 | 来源Recall@1 | 来源Recall@3 | 1000-token预算实际显示来源Recall | 平均显示来源数 |", "|---|---:|---:|---:|---:|---:|"]
    for row in summaries:
        if row["group"] == "all":
            lines.append(f"| {row['method']} | {row['questions']} | {row['source_recall_at_1']:.1%} | {row['source_recall_at_3']:.1%} | {row['source_recall_shown_1000_tokens']:.1%} | {row['shown_sources_mean']:.2f} |")
    lines += ["", "## 按原始answer_in_text标注分组", "", "| 方法 | 原始标签组 | 题数 | 来源Recall@1 | 来源Recall@3 | 实际显示来源Recall |", "|---|---|---:|---:|---:|---:|"]
    for row in summaries:
        if row["group"] != "all" and row["questions"]:
            lines.append(f"| {row['method']} | {row['group']} | {row['questions']} | {row['source_recall_at_1']:.1%} | {row['source_recall_at_3']:.1%} | {row['source_recall_shown_1000_tokens']:.1%} |")
    overall = [row for row in summaries if row["group"] == "all"]
    best = sorted(overall, key=lambda row: (-row["source_recall_shown_1000_tokens"], -row["source_recall_at_1"], METHODS.index(row["method"])))[0]
    baseline = next(row for row in overall if row["method"] == "dense_current")
    improvement = best["source_recall_shown_1000_tokens"] - baseline["source_recall_shown_1000_tokens"]
    lines += ["", "## 开发阶段建议", "", f"按预先指定的实际显示来源Recall排序，最高为 **{best['method']}**：{best['source_recall_shown_1000_tokens']:.1%}；当前dense为{baseline['source_recall_shown_1000_tokens']:.1%}，差{improvement * 100:+.1f}个百分点。并列时依次比较Recall@1和固定方法顺序。", "",
        "这只推荐下一轮开发设置；需要在冻结设置后用独立测试确认收益，并检查开放回答质量。检索命中不能代替事实准确率。所有方案使用同一个Qwen tokenizer、1000-token证据预算和2048-token输入上限，按实际显示证据计来源命中。", "",
        "当前dense阈值为0.45；来源去重每个来源只取排名最高的一个窗口。另列无阈值dense以区分阈值和去重的影响。TF-IDF/BM25采用小写、sklearn英文停用词和两字符以上word token，不做词干提取。BM25 k1=1.5、b=0.75。RRF k=60、权重1:1，每个检索器取前100候选；BM25零分项不进入融合。", "",
        "完整分题检索结果见per_question_retrieval.jsonl，配对来源组bootstrap区间见analysis.json。本诊断只用CPU、最多2线程，与GPU推理并行时的时间不作正式性能数据。生产检索代码保持不变。", ""]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"output": str(out), "best_development_method": best["method"], "best_shown_recall": best["source_recall_shown_1000_tokens"], "dense_current_shown_recall": baseline["source_recall_shown_1000_tokens"]}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="paper_data/external_benchmarks/dev_openqa_en.jsonl")
    parser.add_argument("--corpus", default="paper_data/external_benchmarks/corpus_pira_chunks_en.jsonl")
    parser.add_argument("--protocol", default="paper_eval/protocol_en.json")
    parser.add_argument("--out", default="outputs/paper_eval/retrieval_dev_diagnosis_20261016")
    run(parser.parse_args())


if __name__ == "__main__":
    main()

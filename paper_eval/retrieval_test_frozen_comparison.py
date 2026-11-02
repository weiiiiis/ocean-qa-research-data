"""CPU-only official retrieval comparison of six previously fixed dev methods.

This supplemental analysis never selects a new model, retriever or parameter.
No LLM answer is generated or read. Frozen official BM25 remains unchanged.
"""
from __future__ import annotations

import os
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .common import ROOT, digest_file, digest_json, read_jsonl, resolve, write_json, write_jsonl
from .english import fit_prompt
from .official_english_report import interval
from .retrieval import Retriever
from .retrieval_diagnosis import METHODS, bm25_matrix, ranked, source_unique_order

METRICS = ("source_recall_at_1", "source_recall_at_3", "source_recall_shown_1000_tokens")
GROUPINGS = ("source_group", "source_group_conservative")
FROZEN_METHOD = "bm25_source_unique"


def file_record(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": digest_file(path), "bytes": path.stat().st_size}


def show(value: dict) -> str:
    mean, ci = value["mean"], value["ci95"]
    if mean is None:
        return "—"
    return f"{100 * mean:.1f}%" + (f" [{100 * ci[0]:.1f}, {100 * ci[1]:.1f}]" if ci else " [CI不可用]")


def run(args: argparse.Namespace) -> None:
    import torch
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from threadpoolctl import threadpool_limits
    from transformers import AutoTokenizer

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    pool_limits = threadpool_limits(limits=2)
    out = resolve(args.out).resolve()
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite retrieval results: {out}")
    frozen_dir = resolve(args.freeze).resolve()
    frozen_path, protocol_path = frozen_dir / "freeze_manifest.json", frozen_dir / "protocol.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if digest_file(protocol_path) != frozen["protocol_sha256"] or protocol["retrieval"]["method"] != "bm25":
        raise ValueError("Frozen official BM25 protocol changed")
    dataset = ROOT / "paper_data/external_benchmarks/official_openqa_en_227.jsonl"
    corpus = ROOT / "paper_data/external_benchmarks/corpus_pira_chunks_en.jsonl"
    for path in (dataset, corpus):
        label = str(path.relative_to(ROOT))
        if digest_file(path) != frozen["data_sha256"][label]:
            raise ValueError(f"Dataset/corpus differs from the formal freeze: {path}")
    dev_analysis_path = ROOT / "outputs/paper_eval/retrieval_dev_diagnosis_20261016/analysis.json"
    prior = json.loads(dev_analysis_path.read_text(encoding="utf-8"))["manifest"]
    dense_protocol_path = ROOT / "paper_eval/protocol_en.json"
    if digest_file(dense_protocol_path) != prior["protocol_sha256"]:
        raise ValueError("The original development-method protocol changed")
    if digest_file(ROOT / "paper_eval/retrieval_diagnosis.py") != prior["code_sha256"]:
        raise ValueError("Original six-method algorithm changed since development")
    dense_protocol = json.loads(dense_protocol_path.read_text(encoding="utf-8"))
    dense_floor = dense_protocol["retrieval"]["min_score"]
    if dense_floor != 0.45:
        raise ValueError("Do not change the previously fixed dense floor")
    questions, chunks = read_jsonl(dataset), read_jsonl(corpus)
    if len(questions) != 227 or len({q["id"] for q in questions}) != 227 or len(chunks) != 1507:
        raise ValueError("Expected official227 and unchanged1507 source windows")
    if any(q.get("split") != "test" or q.get("choices") for q in questions):
        raise ValueError("This wrapper accepts the prepared official open-QA stems only")
    if len({q["source_group"] for q in questions}) != 181 or len({q["source_group_conservative"] for q in questions}) != 144:
        raise ValueError("Canonical/conservative grouping differs from prepared official data")
    code_paths = [ROOT / "paper_eval" / name for name in
                  ("retrieval_diagnosis.py", "retrieval.py", "bm25_retrieval.py", "english.py", "common.py", "official_english_report.py")]
    code_paths.append(Path(__file__))
    source_paths = [dataset, corpus, frozen_path, protocol_path, dense_protocol_path, dev_analysis_path] + code_paths
    sources = {str(path): file_record(path) for path in source_paths}
    print("CPU-only fixed-method retrieval comparison: official227, source-only1507, max2 threads", flush=True)
    # The retrieval boundary contains only question stems and source passages.
    query_texts = [question["question"] for question in questions]
    retriever = Retriever(corpus, resolve(protocol["embedding_model"]), ROOT / "cache/paper_eval")
    query_vectors = retriever.encode(query_texts, batch_size=8)
    dense_scores = np.asarray(retriever.matrix @ query_vectors.T)
    texts = [chunk["text"] for chunk in chunks]
    lexical_settings = prior["lexical_preprocessing"]
    tfidf = TfidfVectorizer(**lexical_settings, sublinear_tf=prior["word_tfidf_sublinear_tf"])
    tfidf_documents = tfidf.fit_transform(texts)
    tfidf_scores = (tfidf_documents @ tfidf.transform(query_texts).T).toarray()
    counts = CountVectorizer(**lexical_settings)
    count_documents = counts.fit_transform(texts)
    bm25, average_length = bm25_matrix(count_documents, k1=prior["bm25"]["k1"], b=prior["bm25"]["b"])
    count_queries = counts.transform(query_texts)
    count_queries.data[:] = 1
    bm25_scores = (bm25 @ count_queries.T).toarray()
    tokenizer = AutoTokenizer.from_pretrained(str(resolve(protocol["base_model"])), local_files_only=True)
    top_k = prior["context"]["top_k"]
    rrf_k = prior["rrf"]["k"]
    rrf_depth = prior["rrf"]["candidate_depth_per_ranker"]
    rankings = []
    for qi, question_text in enumerate(query_texts):
        dense, lexical, bm = dense_scores[:, qi], tfidf_scores[:, qi], bm25_scores[:, qi]
        dense_order, filtered = ranked(dense), ranked(dense, dense_floor)
        tfidf_order = [index for index in ranked(lexical) if lexical[index] > 0]
        bm_order = [index for index in ranked(bm) if bm[index] > 0]
        rrf = np.zeros(len(chunks), dtype=np.float64)
        for order in (dense_order[:rrf_depth], bm_order[:rrf_depth]):
            for rank, index in enumerate(order, 1):
                rrf[index] += 1 / (rrf_k + rank)
        rrf_order = [index for index in ranked(rrf) if rrf[index] > 0]
        methods = {
            "dense_current": (filtered[:top_k], dense),
            "dense_source_unique": (source_unique_order(filtered, chunks, top_k), dense),
            "dense_source_unique_no_floor": (source_unique_order(dense_order, chunks, top_k), dense),
            "word_tfidf_source_unique": (source_unique_order(tfidf_order, chunks, top_k), lexical),
            "bm25_source_unique": (source_unique_order(bm_order, chunks, top_k), bm),
            "dense_bm25_rrf_source_unique": (source_unique_order(rrf_order, chunks, top_k), rrf),
        }
        for method, (indices, scores) in methods.items():
            hits = [{**chunks[index], "score": float(scores[index])} for index in indices]
            # No official answer, source label or human assessment is passed.
            prompt, shown = fit_prompt(tokenizer, {"question": question_text}, hits, "openqa",
                protocol["retrieval"]["max_context_tokens"], protocol["generation"]["max_input_tokens"],
                protocol["prompt_profile"])
            rankings.append({"question_index": qi, "method": method,
                "raw_hit_ids": [hit["id"] for hit in hits], "raw_source_ids": [hit["section"] for hit in hits],
                "shown_hit_ids": [hit["id"] for hit in shown], "shown_source_ids": [hit["section"] for hit in shown if hit["text"].strip()],
                "shown_context": shown, "full_prompt_tokens": len(tokenizer.encode(prompt, add_special_tokens=False)),
                "prompt_sha256": digest_json(prompt), "any_shown_chunk_truncated": any(hit.get("context_truncated") for hit in shown)})
        if (qi + 1) % 40 == 0:
            print(f"Fixed rankings and actual context budget {qi + 1}/227", flush=True)
    # Only now use gold source identity and assessment labels for offline scoring.
    records = []
    for ranking in rankings:
        question = questions[ranking.pop("question_index")]
        gold = question["expected_source_id"]
        records.append({**ranking, "id": question["id"], "original_id": question["original_id"],
            "source_group": question["source_group"], "source_group_conservative": question["source_group_conservative"],
            "corpus": question["corpus"], "original_answer_in_text": question["original_assessments"].get("answer_in_text"),
            "expected_source_id_offline_only": gold,
            "source_recall_at_1": float(gold in ranking["raw_source_ids"][:1]),
            "source_recall_at_3": float(gold in ranking["raw_source_ids"][:3]),
            "source_recall_shown_1000_tokens": float(gold in ranking["shown_source_ids"])})
    slices = {"all_227": {q["id"] for q in questions},
              "published_answer_in_text_1_206": {q["id"] for q in questions if q["original_assessments"].get("answer_in_text") == 1}}
    if len(slices["published_answer_in_text_1_206"]) != 206:
        raise ValueError("Published support label count changed")
    summaries, differences, csv_rows = {}, {}, []
    index = {(row["method"], row["id"]): row for row in records}
    for slice_name, ids in slices.items():
        summaries[slice_name], differences[slice_name] = {}, {}
        for grouping in GROUPINGS:
            summaries[slice_name][grouping], differences[slice_name][grouping] = {}, {}
            for method in METHODS:
                selected = [row for row in records if row["method"] == method and row["id"] in ids]
                values = {metric: interval([(row[grouping], row[metric], 1.0) for row in selected]) for metric in METRICS}
                summaries[slice_name][grouping][method] = values
                for metric in METRICS:
                    csv_rows.append({"slice": slice_name, "grouping": grouping, "method": method, "metric": metric, **values[metric]})
                if method != FROZEN_METHOD:
                    differences[slice_name][grouping][method + "_minus_frozen_bm25"] = {metric: interval([
                        (index[(method, qid)][grouping], index[(method, qid)][metric] - index[(FROZEN_METHOD, qid)][metric], 1.0)
                        for qid in sorted(ids)]) for metric in METRICS}
    for path in source_paths:
        if digest_file(path) != sources[str(path)]["sha256"]:
            raise ValueError(f"A frozen source changed during comparison: {path}")
    out.mkdir(parents=True, exist_ok=False)
    write_jsonl(out / "per_question_retrieval.jsonl", records)
    manifest = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "analysis_type": "supplemental_fixed_development_methods_on_official_test",
        "official_retrieval_selection_changed": False, "frozen_official_method": FROZEN_METHOD,
        "methods_fixed_in_development": list(METHODS), "method_selection_from_test": False,
        "source_files": sources, "freeze_manifest_sha256": sources[str(frozen_path)]["sha256"],
        "question_counts": {name: len(ids) for name, ids in slices.items()}, "canonical_source_groups": 181, "conservative_source_groups": 144,
        "corpus_chunks": 1507, "source_passages": len({row["section"] for row in chunks}),
        "cpu_threads_maximum": 2, "torch_threads": torch.get_num_threads(), "gpu_used": False, "llm_generation": False,
        "official_llm_outputs_read": False, "query_policy": "Stem only; references, choices, labels and gold source IDs never enter query or context construction",
        "gold_policy": "Gold source and answer_in_text read only after all rankings and actual context budgets were fixed",
        "fixed_hyperparameters": {"dense_floor": dense_floor, "top_k": top_k, "lexical": lexical_settings,
            "tfidf_sublinear_tf": True, "bm25": {**prior["bm25"], "average_document_length": average_length}, "rrf": prior["rrf"]},
        "actual_budget": {"context_tokens": 1000, "input_tokens": 2048, "prompt_profile": protocol["prompt_profile"],
            "code": "Frozen paper_eval.english.fit_prompt, including source headers and actual evidence_v2 postamble"},
        "uncertainty": {"replicates": 2000, "seed": 20261017, "groupings": list(GROUPINGS), "paired_difference_reference": FROZEN_METHOD},
        "limitations": ["The formal retriever remains frozen BM25 regardless of these results. Other methods are supplemental exploratory tests of fixed development configurations.",
            "No model/prompt/retrieval parameter may be selected or tuned from these official results.",
            "Source identity recall does not prove answer entailment, a relevant span surviving truncation or correct LLM generation.",
            "Main clusters merge identical known EID/exact passages. Conservative corpus2 grouping is an uncertainty assumption, not recovered true report lineage.",
            "Multiple supplemental methods/metrics are exploratory and intervals are not multiplicity-adjusted.",
            "CPU computation can overlap GPU inference; runtime here is not formal performance evidence. No expert scores are inferred."]}
    write_json(out / "analysis.json", {"manifest": manifest, "summary": summaries, "paired_source_cluster_differences_from_frozen_bm25": differences})
    with (out / "summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0])); writer.writeheader()
        writer.writerows({key: json.dumps(value) if isinstance(value, list) else value for key, value in row.items()} for row in csv_rows)
    lines = ["# 冻结后官方测试集检索补充比较", "",
        "**正式方案继续使用已冻结BM25，不因本报告结果改变。** 其余方法为开发阶段已固定配置的补充探索性测试，不排序推荐新方法，不据此调整参数、模型或提示。", "",
        "仅CPU、最多2线程；未生成或读取任何LLM回答。所有查询仅题干，gold仅在排名及实际预算固定后用于离线来源计分。", ""]
    for slice_name, ids in slices.items():
        lines += [f"## {slice_name}：{len(ids)}题", ""]
        for grouping in GROUPINGS:
            lines += [f"### {grouping}", "", "| 既定方法 | 来源Recall@1 [95% CI] | 来源Recall@3 [95% CI] | 实际1000-token显示Recall [95% CI] |", "|---|---:|---:|---:|"]
            for method in METHODS:
                values = summaries[slice_name][grouping][method]
                lines.append(f"| {method}{'（冻结方案）' if method == FROZEN_METHOD else ''} | " + " | ".join(show(values[metric]) for metric in METRICS) + " |")
            lines += ["", "| 方法 − 冻结BM25 | 配对题数 | Recall@1差值 | Recall@3差值 | 实际显示Recall差值 |", "|---|---:|---:|---:|---:|"]
            for pair, values in differences[slice_name][grouping].items():
                lines.append(f"| {pair} | {len(ids)} | " + " | ".join(show(values[metric]) for metric in METRICS) + " |")
            lines += ["", "差值单位为百分点，按同一来源组配对bootstrap；全部方法使用相同题目，CI未作多重比较校正。", ""]
    lines += ["## 可复核边界", "",
        "- 主集227题、原answer_in_text=1子组206题；canonical source_group共181组，保守分组144组。",
        "- six-method算法与参数复用原retrieval_diagnosis.py，dense阈值0.45、RRF k=60/深度100、BM25 k1=1.5/b=0.75不变。实际显示预算使用正式冻结evidence_v2提示与1000/2048-token上限。",
        "- per_question_retrieval.jsonl保存固定排名、实际显示原文及离线计分；原算法/预算源码与freeze_manifest SHA见analysis.json。",
        "- 来源命中不等于事实正确或证据支持；corpus2没有可靠报告ID，保守分组不能恢复真实全文来源。",
        "- 此补充比较不会修改正式BM25、冻结文件、训练样本、题目标签或任何正式LLM原始输出。", ""]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"out": str(out), "frozen_bm25_unchanged": True,
        "all_227": {method: {metric: values["mean"] for metric, values in summaries["all_227"]["source_group"][method].items()} for method in METHODS}}, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", default="paper_data/frozen_english_v1_20261017")
    parser.add_argument("--out", default="outputs/paper_eval/retrieval_test_frozen_comparison_20261017")
    run(parser.parse_args())


if __name__ == "__main__":
    main()

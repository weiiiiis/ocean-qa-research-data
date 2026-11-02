r"""CPU-only source-family overlap audit; never reads benchmark answer fields.

Run from the workspace:
  .venv\Scripts\python.exe paper_data/english_training_v2/overlap_audit/audit_sources.py

The 60 source facts are the unit of audit, rather than their 300 formatted SFT
examples. Similarity flags require human review and never change any dataset.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import re
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Process-local limits; other inference/training processes are unaffected.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

ROOT = Path(__file__).resolve().parents[3]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def normalized(text: str) -> str:
    return re.sub(r"[\W_]+", "", unicodedata.normalize("NFKC", text).casefold())


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, list) else value
                             for key, value in row.items()})


def benchmark_stems(path: Path) -> list[dict]:
    # Deliberately discard all other fields immediately: no reference answer,
    # answer letter, choice, support text or model output is used in this audit.
    rows = [{key: row.get(key, "") for key in ("id", "question", "source_group")}
            for row in read_jsonl(path)]
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError(f"Empty or duplicate benchmark IDs: {path}")
    if any(not row["question"].strip() for row in rows):
        raise ValueError(f"Empty benchmark question: {path}")
    return rows


def audit(args: argparse.Namespace) -> None:
    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer

    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    out = args.out.resolve()
    # Reproducibility must not erase any reviewer work added after this audit.
    for name in ("development_nearest_review.csv", "official_test_nearest_review.csv", "flagged_review.csv"):
        prior = out / name
        if prior.exists():
            with prior.open(encoding="utf-8-sig", newline="") as stream:
                if any(any(str(row.get(key, "")).strip() for key in
                           ("human_overlap_decision", "human_reviewer", "human_notes"))
                       for row in csv.DictReader(stream)):
                    raise ValueError(f"Reviewer work exists in {prior}; preserve it and select a new --out directory")
    started = time.perf_counter()
    paths = {"source_facts": args.facts.resolve(), "family_split": args.family_split.resolve(),
             "development": args.dev.resolve(), "official_test": args.test.resolve()}
    input_hashes = {key: digest(path) for key, path in paths.items()}
    facts = read_jsonl(paths["source_facts"])
    if len(facts) != 60 or len({fact["family_id"] for fact in facts}) != 60:
        raise ValueError("This audit expects the 60 unique native source facts, not 300 formatted examples")
    if any(not fact.get(field, "").strip() for fact in facts for field in ("question", "answer")):
        raise ValueError("Each native source fact requires a question and answer")
    partition = json.loads(paths["family_split"].read_text(encoding="utf-8"))
    train_ids, validation_ids = set(partition["train_families"]), set(partition["validation_families"])
    if train_ids & validation_ids or train_ids | validation_ids != {fact["family_id"] for fact in facts}:
        raise ValueError("Source-family partition must exactly cover the 60 facts without overlap")
    train_indices = [index for index, fact in enumerate(facts) if fact["family_id"] in train_ids]
    benchmarks = {"development": benchmark_stems(paths["development"]),
                  "official_test": benchmark_stems(paths["official_test"])}
    if len(benchmarks["development"]) != 101 or len(benchmarks["official_test"]) != 227:
        raise ValueError("Expected the selected Pira development101 and official test227 stems")
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
    model = AutoModel.from_pretrained(str(args.model), local_files_only=True).to("cpu").eval()

    def encode(texts: list[str]) -> np.ndarray:
        values = []
        for start in range(0, len(texts), args.batch_size):
            batch = tokenizer(texts[start:start + args.batch_size], padding=True, truncation=True,
                              max_length=384, return_tensors="pt")
            with torch.inference_mode():
                hidden = model(**batch).last_hidden_state
                mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                vector = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                vector = torch.nn.functional.normalize(vector, p=2, dim=1)
            values.append(vector.numpy().astype("float32"))
        return np.vstack(values)

    canonical = [fact["question"] for fact in facts]
    rewritten = [fact.get("question_rephrased") or fact["question"] for fact in facts]
    qa = [f"Question: {fact['question']}\nAnswer: {fact['answer']}" for fact in facts]
    print("Encoding 60 canonical questions, 60 existing rephrasings and 60 QA pairs on CPU...", flush=True)
    vectors = encode(canonical + rewritten + qa)
    q_vectors, rephrase_vectors, qa_vectors = np.split(vectors, 3)
    results, summaries = [], {}
    all_scores = {}
    for split, rows in benchmarks.items():
        print(f"Auditing {split}: {len(rows)} question stems; no reference answers or model results...", flush=True)
        stems = encode([row["question"] for row in rows])
        canonical_scores = stems @ q_vectors.T
        rephrase_scores = stems @ rephrase_vectors.T
        qa_scores = stems @ qa_vectors.T
        primary_scores = np.maximum(canonical_scores, rephrase_scores)
        split_results = []
        for index, row in enumerate(rows):
            nearest_index = int(np.argmax(primary_scores[index]))
            nearest = facts[nearest_index]
            canonical_index = int(np.argmax(canonical_scores[index]))
            qa_index = int(np.argmax(qa_scores[index]))
            train_nearest_index = train_indices[int(np.argmax(primary_scores[index, train_indices]))]
            train_nearest = facts[train_nearest_index]
            exact = [fact["family_id"] for fact, q1, q2 in zip(facts, canonical, rewritten)
                     if row["question"] == q1 or row["question"] == q2]
            norm = normalized(row["question"])
            normalized_matches = [fact["family_id"] for fact, q1, q2 in zip(facts, canonical, rewritten)
                                  if norm == normalized(q1) or norm == normalized(q2)]
            semantic = float(primary_scores[index, nearest_index]) >= args.threshold
            qa_flag = float(qa_scores[index, qa_index]) >= args.threshold
            matched_variant = ("canonical" if canonical_scores[index, nearest_index] >= rephrase_scores[index, nearest_index]
                               else "existing_rephrasing")
            item = {
                "benchmark_split": split, "id": row["id"], "question": row["question"], "source_group": row["source_group"],
                "exact_question_flag": bool(exact), "normalized_question_flag": bool(normalized_matches),
                "semantic_question_flag": semantic, "question_overlap_any_flag": bool(exact or normalized_matches or semantic),
                "exact_matching_families": exact, "normalized_matching_families": normalized_matches,
                "nearest_training_family_id": nearest["family_id"], "nearest_training_section": nearest["section"],
                "nearest_training_partition": "train" if nearest["family_id"] in train_ids else "validation",
                "nearest_training_pdf_page": nearest["pdf_page"], "nearest_training_question": nearest["question"],
                "nearest_training_question_rephrased": nearest.get("question_rephrased", ""),
                "nearest_training_answer": nearest["answer"], "nearest_matching_question_variant": matched_variant,
                "nearest_question_cosine": float(primary_scores[index, nearest_index]),
                "canonical_nearest_question_cosine": float(canonical_scores[index, canonical_index]),
                "canonical_nearest_family_id": facts[canonical_index]["family_id"],
                "rephrased_max_question_cosine": float(np.max(rephrase_scores[index])),
                "training48_exact_flag": bool(set(exact) & train_ids),
                "training48_normalized_flag": bool(set(normalized_matches) & train_ids),
                "training48_semantic_flag": float(primary_scores[index, train_nearest_index]) >= args.threshold,
                "training48_nearest_family_id": train_nearest["family_id"],
                "training48_nearest_question_cosine": float(primary_scores[index, train_nearest_index]),
                "training48_nearest_question": train_nearest["question"],
                "training48_nearest_answer": train_nearest["answer"],
                "training48_nearest_section": train_nearest["section"],
                "qa_auxiliary_semantic_flag": qa_flag, "qa_auxiliary_max_cosine": float(qa_scores[index, qa_index]),
                "qa_auxiliary_nearest_family_id": facts[qa_index]["family_id"],
                "human_overlap_decision": "", "human_reviewer": "", "human_notes": "",
            }
            split_results.append(item)
        count_fields = ("exact_question_flag", "normalized_question_flag", "semantic_question_flag", "question_overlap_any_flag",
                        "training48_exact_flag", "training48_normalized_flag", "training48_semantic_flag", "qa_auxiliary_semantic_flag")
        summaries[split] = {"benchmark_questions": len(rows), **{field: sum(row[field] for row in split_results) for field in count_fields},
                            "maximum_question_cosine": max(row["nearest_question_cosine"] for row in split_results),
                            "maximum_question_answer_auxiliary_cosine": max(row["qa_auxiliary_max_cosine"] for row in split_results),
                            "flagged_question_ids": [row["id"] for row in split_results if row["question_overlap_any_flag"]],
                            "auxiliary_only_flagged_ids": [row["id"] for row in split_results
                                                           if row["qa_auxiliary_semantic_flag"] and not row["question_overlap_any_flag"]]}
        results.extend(split_results)
        all_scores[split] = {"canonical": canonical_scores, "rephrased": rephrase_scores, "qa_auxiliary": qa_scores}
    # Ensure another process did not change source candidates during the audit.
    if any(digest(paths[key]) != value for key, value in input_hashes.items()):
        raise RuntimeError("An input changed while the CPU audit was running; no result will be published")
    out.mkdir(parents=True, exist_ok=True)
    fields = list(results[0])
    for split in benchmarks:
        write_csv(out / f"{split}_nearest_review.csv", [row for row in results if row["benchmark_split"] == split], fields)
    flagged = [row for row in results if row["question_overlap_any_flag"] or row["qa_auxiliary_semantic_flag"]]
    write_csv(out / "flagged_review.csv", flagged, fields)
    write_json(out / "audit.json", {"summary": summaries, "records": results})
    np.savez_compressed(out / "cosine_matrices.npz", **{
        f"{split}_{kind}": matrix for split, groups in all_scores.items() for kind, matrix in groups.items()})
    weights = {str(path.relative_to(args.model)): digest(path) for path in sorted(args.model.rglob("*"))
               if path.is_file() and path.suffix in {".bin", ".safetensors", ".json", ".txt", ".model"} and ".cache" not in path.parts}
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "audit_version": "native_source60_pira_question_overlap_v1",
        "script": str(Path(__file__).resolve()), "script_sha256": digest(Path(__file__)),
        "input_files": {key: {"path": str(path), "sha256": input_hashes[key]} for key, path in paths.items()},
        "native_source_families": len(facts), "training_partition_families": len(train_ids), "validation_partition_families": len(validation_ids),
        "formatted_sft_examples_not_audit_units": 300,
        "embedding": {"path": str(args.model.resolve()), "file_hashes": weights, "device": "cpu", "torch_threads": torch.get_num_threads(),
                      "torch_interop_threads": torch.get_num_interop_threads(), "pooling": "attention-mask mean; L2-normalized",
                      "max_tokens": 384, "batch_size": args.batch_size, "dtype": "float32"},
        "threshold": args.threshold, "exact_rule": "Raw case-sensitive equality against canonical or existing rephrased training question",
        "normalized_rule": "Unicode NFKC, casefold, remove non-word characters and underscores, against canonical or existing rephrased question",
        "primary_semantic_rule": "Maximum cosine against the 60 canonical questions and 60 already-authored rephrasings, grouped into 60 native source families",
        "canonical_only_score": "Reported separately; primary score includes existing rephrasing because it is used by candidate SFT",
        "qa_auxiliary_rule": "Compare benchmark question stem to canonical training Question/Answer concatenation; auxiliary only, not part of primary question flag",
        "evaluation_fields_used": ["id", "question", "source_group"], "benchmark_answers_or_choices_used": False,
        "model_generation_or_model_performance_used": False, "dataset_or_split_mutation": False,
        "source_family_partition": "All60 audit is conservative; training48 columns separately audit actual candidate training families, excluding 12 validation families",
        "human_review_status": "pending; all reviewer/decision/note cells blank",
        "interpretation": "Flags indicate candidates for manual review, not proven leakage. No automatic exclusion or relabeling. No automatic flag does not establish independence. This only audits the new source60, not all old adapter training or unknown base-model pretraining.",
        "official_test_boundary": "Official test was inspected solely as question-stem contamination audit before training; test answers and model performance were not used, and no test content was added to SFT or retrieval",
        "packages": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "numpy")},
        "elapsed_seconds": time.perf_counter() - started, "summary": summaries,
    }
    outputs = ["development_nearest_review.csv", "official_test_nearest_review.csv", "flagged_review.csv", "audit.json", "cosine_matrices.npz"]
    manifest["output_files_sha256"] = {name: digest(out / name) for name in outputs}
    write_json(out / "manifest.json", manifest)
    lines = ["# 新英文训练来源与Pirá题干相似性审计", "",
        "仅使用CPU，2个PyTorch线程；审计单位为60条原生来源事实，涵盖原问题与已有改写。没有按300条格式示例重复计数。", "",
        "## 结果", "", "| 范围 | 题数 | 原文精确相同 | 规范化相同 | 问题语义≥0.82 | 其中实际训练48族语义flag | QA辅助语义flag |", "|---|---:|---:|---:|---:|---:|---:|"]
    for split, summary in summaries.items():
        lines.append(f"| {split} | {summary['benchmark_questions']} | {summary['exact_question_flag']} | {summary['normalized_question_flag']} | {summary['semantic_question_flag']} | {summary['training48_semantic_flag']} | {summary['qa_auxiliary_semantic_flag']} |")
    lines += ["", "主语义分数是题干对原问题/已有改写的最大余弦，阈值0.82；原问题单独分数和Q+A辅助分数分开保存。Q+A辅助命中不自动并入主flag。规范化为Unicode NFKC、casefold后去非文字/数字及下划线。", "",
        "每题最相近的来源问题、答案、教材section和页码在两个nearest_review.csv中。training48列只看48个实际训练族；其余12族属于内部验证，未算作实际训练。", "",
        "**相似概念不是自动认定的泄漏。未触发阈值也不能证明完全独立。** 本审计只覆盖这60个新来源事实，不覆盖旧适配器全部训练历史或未知基座预训练。所有人工决定、姓名与备注栏为空；未删题、改标签、改训练内容或划分。", "",
        "正式test227只读取id、question、source_group作为训练前污染审计；未利用正式测试答案、选项或模型表现，也没有把测试内容写入训练集或检索语料。", "",
        "## 复现", "", "```powershell", ".venv\\Scripts\\python.exe paper_data/english_training_v2/overlap_audit/audit_sources.py", "```", "",
        "manifest.json记录脚本、模型和全部输入哈希、线程、阈值与方法；audit.json保留逐题记录，cosine_matrices.npz保留所有成对分数。", ""]
    (out / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summaries, ensure_ascii=False), flush=True)
    print(f"Audit complete in {time.perf_counter() - started:.1f}s: {out}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--facts", type=Path, default=ROOT / "paper_data/english_training_v2/source_facts_60.jsonl")
    parser.add_argument("--family-split", type=Path, default=ROOT / "paper_data/english_training_v2/family_split.json")
    parser.add_argument("--dev", type=Path, default=ROOT / "paper_data/external_benchmarks/dev_openqa_en.jsonl")
    parser.add_argument("--test", type=Path, default=ROOT / "paper_data/external_benchmarks/pira_test_en_227.jsonl")
    parser.add_argument("--model", type=Path, default=ROOT / "models/embedding_minilm")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--threshold", type=float, default=0.82)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1 or args.batch_size <= 0:
        parser.error("threshold must be in [0,1] and batch size must be positive")
    audit(args)


if __name__ == "__main__":
    main()

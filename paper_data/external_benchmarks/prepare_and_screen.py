"""Inspect pinned public files and screen against known local SFT data.

Outputs are candidates with original English references, not expert-approved labels.
No model answer generation, training or evaluation score is performed.
"""
from __future__ import annotations

import ast
import collections
import csv
import hashlib
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = Path(__file__).resolve().parent


def main():
    import numpy as np
    import pandas as pd
    import torch
    from sklearn.feature_extraction.text import TfidfVectorizer
    from paper_eval.common import normalize, write_json, write_jsonl, digest_file
    from paper_eval.data import training_inventory
    from paper_eval.retrieval import Retriever
    torch.set_num_threads(4)
    raw = OUT / "raw"
    frames = {split: pd.read_parquet(next((raw / "pira/data").glob(split + "-*.parquet"))) for split in ["train", "validation", "test"]}

    def clean(value):
        return None if isinstance(value, float) and math.isnan(value) else value

    def source_id(row):
        text = normalize(str(row["abstract"]))
        return "PIRA-DOC-" + hashlib.sha256(text.encode()).hexdigest()[:16]

    # Preserve supporting passages only in the retrieval corpus. Questions,
    # references, human assessments, and MC alternatives are never indexed.
    corpus = {}
    split_groups = {}
    for split, frame in frames.items():
        split_groups[split] = set()
        for row in frame.to_dict("records"):
            sid = source_id(row)
            split_groups[split].add(sid)
            corpus[sid] = {"id": sid, "text": row["abstract"], "corpus": int(row["corpus"]), "article_eid": clean(row["eid_article_scopus"]),
                           "source": "Pira 2.0 supplied supporting passage", "dataset_license": "CC BY 4.0; preserve original source rights"}
    write_jsonl(OUT / "pira_supporting_passages.jsonl", list(corpus.values()))
    pira = []
    for row in frames["test"].to_dict("records"):
        aliases = list(dict.fromkeys(str(row[k]).strip() for k in ["answer_en_origin", "answer_en_validate"] if clean(row[k]) is not None and str(row[k]).strip()))
        pira.append({"id": "PIRA-" + row["id_qa"], "original_id": row["id_qa"], "dataset": "pira_2_default_test", "language": "en",
                     "question": row["question_en_origin"], "reference_answers": aliases, "source_group": source_id(row),
                     "supporting_passage_id": source_id(row), "original_split": "test", "corpus": int(row["corpus"]),
                     "original_assessments": {k: clean(row[k]) for k in ["answer_in_text", "at_labels", "question_meaningful", "answer_equivalent", "answer_difficulty"]},
                     "review_status": "published references preserved; local suitability and overlap review pending"})
    write_jsonl(OUT / "pira_test_en_227.jsonl", pira)
    mcqa_frame = pd.read_parquet(next((raw / "pira/mcqa").glob("test-*.parquet")))
    pira_mc = [{"id": "PIRA-MC-" + r["id"], "original_id": r["id"], "dataset": "pira_2_mcqa_test", "language": "en", "question": r["question"],
                "choices": {k: r[k] for k in "ABCDE"}, "answer": r["alternative"], "reference_answer_text": r["correct"]} for r in mcqa_frame.to_dict("records")]
    for row in pira_mc:
        if row["choices"][row["answer"]].strip() != row["reference_answer_text"].strip():
            raise ValueError("Pira multiple-choice label mismatch")
    write_jsonl(OUT / "pira_mcqa_test_en_227.jsonl", pira_mc)
    science = pd.read_csv(raw / "oceanbenchmark_2026/Science_Text/data.csv")
    science_rows = [{"id": f"OCEANSCI-{int(r['id']):03d}", "dataset": "oceanbenchmark_2026_science_text", "language": "en", "question": r["question"],
                     "choices": ast.literal_eval(r["choices"]), "answer": r["answer"], "source_group": r["pdf_title"], "source_pdf_title": r["pdf_title"],
                     "category": clean(r["category"]), "review_status": "source-paper context / label ambiguity review pending"} for r in science.to_dict("records")]
    write_jsonl(OUT / "ocean_science_text_en_102.jsonl", science_rows)
    deduplicated, duplicate_map, seen = [], [], {}
    for row in science_rows:
        key = normalize(row["question"]) + json.dumps(row["choices"], sort_keys=True) + row["answer"]
        if key in seen:
            duplicate_map.append({"excluded_id": row["id"], "retained_id": seen[key], "reason": "same normalized question, choices and official label"})
        else:
            seen[key] = row["id"]
            deduplicated.append(row)
    write_jsonl(OUT / "ocean_science_text_en_dedup_100.jsonl", deduplicated)
    write_json(OUT / "exact_duplicate_exclusions.json", duplicate_map)
    legacy = [json.loads(line) for line in (raw / "oceanbench_2024/OceanBench.txt").read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    legacy_qa = [r for r in legacy if r["task_type"] == "Question-Answering"]
    write_jsonl(OUT / "oceanbench_legacy_qa_1181.jsonl", legacy_qa)
    ids_a, ids_b = {r["original_id"] for r in pira}, {r["original_id"] for r in pira_mc}
    summary = {"pira_total": sum(map(len, frames.values())), "pira_splits": {s: len(f) for s, f in frames.items()},
        "pira_test_source_passages": len(split_groups["test"]), "pira_all_available_supporting_passages": len(corpus),
        "pira_test_source_passages_also_in_original_train": len(split_groups["test"] & split_groups["train"]),
        "pira_test_source_passages_also_in_original_validation": len(split_groups["test"] & split_groups["validation"]),
        "pira_open_mcqa_shared_ids": len(ids_a & ids_b), "pira_answer_in_text": {str(k): int(v) for k, v in frames['test'].answer_in_text.fillna(-1).value_counts().items()},
        "pira_at_labels": {str(k): int(v) for k, v in frames['test'].at_labels.fillna(-1).value_counts().items()},
        "pira_at_labels_warning": "Do not equate at_labels with passage-answerability: 15 of 19 at_labels=0 rows have answer_in_text=1, and 4 at_labels=1 rows have answer_in_text=0.",
        "oceanbenchmark_text_count": len(science_rows), "oceanbenchmark_source_papers": science.pdf_title.nunique(),
        "oceanbenchmark_missing_category": int(science.category.isna().sum()), "oceanbenchmark_unique_after_exact_dedup": len(deduplicated),
        "legacy_records": len(legacy), "legacy_tasks": dict(collections.Counter(r["task_type"] for r in legacy)), "legacy_qa_records": len(legacy_qa),
        "candidate_main_questions": len(pira), "candidate_additional_mcq_raw": len(science_rows), "candidate_records_before_dedup": len(pira) + len(science_rows),
        "unique_normalized_question_candidates": len({normalize(r['question']) for r in pira + deduplicated})}
    write_json(OUT / "inspection_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    # Cross-language semantic audit: include Chinese training answers in windows,
    # not just exact English-versus-Chinese string comparison.
    inventory, training = training_inventory()
    windows = [(i, text[start:start + 700]) for i, r in enumerate(training) for text in [r["question"] + "\n" + r["answer"]] for start in range(0, len(text), 500)]
    candidates = pira + science_rows
    retriever = Retriever(ROOT / "paper_data/corpus.jsonl", ROOT / "models/embedding_minilm", ROOT / "cache/paper_eval")
    train_vectors = retriever.encode([t for _, t in windows])
    test_vectors = retriever.encode([q["question"] for q in candidates])
    scores = test_vectors @ train_vectors.T
    exact_questions = {normalize(r["question"]) for r in training}
    items = []
    for i, row in enumerate(candidates):
        j = int(np.argmax(scores[i]))
        nearest = training[windows[j][0]]
        score = float(scores[i, j])
        flags = []
        if normalize(row["question"]) in exact_questions:
            flags.append("exact_known_sft_question")
        if score >= 0.82:
            flags.append("cross_language_semantic_review")
        items.append({"id": row["id"], "dataset": row["dataset"], "question": row["question"], "source_group": row["source_group"], "flags": flags,
                      "similarity": score, "nearest_training_question": nearest["question"], "nearest_training_answer": nearest["answer"],
                      "training_path": nearest["path"], "training_line": nearest["line"], "human_decision": "", "reviewer": "", "notes": ""})
    write_json(OUT / "known_sft_overlap.json", {"training_inventory": inventory, "unique_training_pairs": len(training), "threshold": 0.82, "items": items,
        "warning": "Screening only, not proof of independence. English questions versus Chinese training require manual semantic review. Full historical SFT source lineage and base pretraining remain unknown."})
    with (OUT / "known_sft_overlap_review.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(items[0]))
        writer.writeheader()
        writer.writerows(items)
    summary["sft_semantic_flags"] = dict(collections.Counter(r["dataset"] for r in items if r["flags"]))
    summary["sft_audited_pairs"] = len(training)
    write_json(OUT / "inspection_summary.json", summary)
    print("Known SFT screening:", summary["sft_semantic_flags"], flush=True)


if __name__ == "__main__":
    main()

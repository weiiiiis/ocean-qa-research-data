"""Prepare one transparent AI pre-scoring job from anonymous review material."""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/paper_eval/en_test_blind40_frozen_v1_20261017/expert_review"
OUT = ROOT / "outputs/paper_eval/ai_prescore_teachercheck_v1_20261018"


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    def read(name):
        with (SOURCE / name).open(encoding="utf-8-sig", newline="") as stream:
            return list(csv.DictReader(stream))
    quality = read("reviewer_1_quality.csv")
    evidence = {row["item_id"]: row for row in read("reviewer_1_evidence.csv")}
    grouped = {}
    for row in quality:
        key = hashlib.sha256(json.dumps([row["question"], row["reference_answer_published_unverified"],
                                         row["supporting_passage_for_review"]], ensure_ascii=False).encode()).hexdigest()[:16]
        bundle = grouped.setdefault(key, {"bundle_id": key, "question": row["question"],
            "published_reference_unverified": row["reference_answer_published_unverified"],
            "supporting_passage_for_quality_review": row["supporting_passage_for_review"], "candidates": []})
        candidate = {"item_id": row["item_id"], "answer_for_quality": row["candidate_answer_citation_markers_removed"]}
        if row["item_id"] in evidence:
            record = evidence[row["item_id"]]
            candidate["evidence_review"] = {"answer_with_original_citations": record["candidate_answer"],
                "actual_evidence_provided": json.loads(record["actual_evidence_provided"])}
        bundle["candidates"].append(candidate)
    bundles = list(grouped.values())
    if len(quality) != 160 or len(evidence) != 80 or len(bundles) != 40 or any(len(b["candidates"]) != 4 for b in bundles):
        raise ValueError("Expected exactly 40 question-source bundles with four answers each")
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "inputs").mkdir()
    (OUT / "graded").mkdir()
    for shard, subset in enumerate((bundles[:14], bundles[14:27], bundles[27:]), 1):
        (OUT / "inputs" / f"shard_{shard}.json").write_bytes(json.dumps(subset, ensure_ascii=False, indent=2).encode("utf-8"))
    manifest = {"purpose": "One AI pre-score set for teacher verification, not two human reviewers",
                "quality_answers": 160, "evidence_answers": 80, "source_bundles": 40,
                "scores_status": "AI grading to be produced; teacher verification pending",
                "formal_human_templates_modified": False, "private_model_mapping_read": False,
                "input_sha256": {name: hashlib.sha256((SOURCE / name).read_bytes()).hexdigest()
                                 for name in ("reviewer_1_quality.csv", "reviewer_1_evidence.csv")}}
    (OUT / "input_manifest.json").write_bytes(json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"))
    print(f"Prepared 40 bundles: {OUT}")


if __name__ == "__main__":
    main()

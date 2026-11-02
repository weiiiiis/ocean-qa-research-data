"""Prepare a local pre-output expert-review sample using published labels only.

This standard-library script never reads model generations, training outputs or
development scores. It does not relabel official data or approve expert scores.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
import platform
import random
from pathlib import Path


HERE = Path(__file__).resolve().parent
EXTERNAL = HERE.parent
SEED = 20261017
SAMPLE_SIZE = 40
EXCLUDE_ORIGINAL_IDS = ("A1329", "B2585")
LEADING_MC_REGEX = r"^\s*([A-E])(?=$|[\s.)\]}:;,，。：、-])"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def dump_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def dump_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                         for key, value in row.items()} for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=EXTERNAL / "official_openqa_en_227.jsonl")
    parser.add_argument("--supporting-passages", type=Path, default=EXTERNAL / "pira_supporting_passages.jsonl")
    parser.add_argument("--corpus", type=Path, default=EXTERNAL / "corpus_pira_chunks_en.jsonl")
    parser.add_argument("--out", type=Path, default=HERE)
    args = parser.parse_args()
    out = args.out.resolve()
    if (out / "manifest.json").exists():
        raise SystemExit("A local plan already exists. Preserve it and use a new --out directory for any changed plan; expert work is never overwritten.")
    rows = read_jsonl(args.dataset)
    if len(rows) != 227 or any(row.get("split") != "test" or row.get("choices") for row in rows):
        raise ValueError("Expected exactly 227 official open-QA test records")
    eligible = [row for row in rows if (row.get("original_assessments") or {}).get("answer_in_text") == 1]
    unsuitable = [row for row in rows if (row.get("original_assessments") or {}).get("answer_in_text") != 1]
    if len(eligible) != 206 or len(unsuitable) != 21:
        raise ValueError("Published-label counts changed; investigate and prepare a new documented plan")
    supporting = {row["id"]: row for row in read_jsonl(args.supporting_passages)}
    chunks = collections.defaultdict(list)
    for chunk in read_jsonl(args.corpus):
        chunks[chunk["section"]].append(chunk["id"])
    strata = collections.defaultdict(list)
    for row in eligible:
        strata[int(row["corpus"])].append(row)
    counts = {corpus: len(pool) for corpus, pool in sorted(strata.items())}
    quotas = {corpus: math.floor(SAMPLE_SIZE * count / len(eligible)) for corpus, count in counts.items()}
    remaining = SAMPLE_SIZE - sum(quotas.values())
    priority = sorted(counts, key=lambda corpus: (-(SAMPLE_SIZE * counts[corpus] / len(eligible) - quotas[corpus]), corpus))
    for corpus in priority[:remaining]:
        quotas[corpus] += 1
    rng = random.Random(SEED)
    selected, sampling_details = [], []
    for corpus in sorted(strata):
        groups = collections.defaultdict(list)
        for row in strata[corpus]:
            groups[row["source_group"]].append(row)
        order = sorted(groups)
        rng.shuffle(order)
        if len(order) < quotas[corpus]:
            raise ValueError("Insufficient distinct sources for the fixed one-question-per-source allocation")
        chosen = order[:quotas[corpus]]
        for group in chosen:
            candidates = sorted(groups[group], key=lambda row: row["original_id"])
            selected.append(candidates[rng.randrange(len(candidates))])
        sampling_details.append({"corpus": corpus, "eligible_questions": counts[corpus],
            "eligible_distinct_source_groups": len(groups), "quota": quotas[corpus], "selected_source_groups": chosen})
    if len({row["source_group"] for row in selected}) != SAMPLE_SIZE:
        raise ValueError("The current plan requires all 40 sampled source_group IDs to differ")

    def review_record(row: dict) -> dict:
        source_id = row["expected_source_id"]
        source = supporting[source_id]
        return {"id": row["id"], "original_id": row["original_id"], "question": row["question"],
            "corpus": row["corpus"], "source_group": row["source_group"],
            "source_group_unit": row.get("source_group_unit"),
            "source_group_conservative": row.get("source_group_conservative", row["source_group"]),
            "expected_source_id_for_review_only": source_id, "source_article_eid": row.get("source_article_eid"),
            "reference_answer_origin_published_unverified": row["reference_answer_origin"],
            "reference_answer_origin_source": row.get("reference_answer_origin_source"),
            "published_assessments_unchanged": row.get("original_assessments", {}),
            "supporting_text_locator": {"file": str(args.supporting_passages.resolve()), "jsonl_id": source_id,
                "utf8_text_sha256": hashlib.sha256(source["text"].encode("utf-8")).hexdigest(),
                "chunk_corpus_file": str(args.corpus.resolve()), "chunk_ids": sorted(chunks[source_id]),
                "source": source.get("source"), "license": source.get("dataset_license")},
            "expert_review_status": "pending", "human_reference_suitable": "", "human_notes": ""}

    review_sample = [dict(review_position=index, **review_record(row)) for index, row in enumerate(selected, 1)]
    reference_review = [dict(review_reason="Published answer_in_text is 0 or missing; reference/evidence suitability needs human review, not a refusal gold label.",
                             **review_record(row)) for row in unsuitable]
    out.mkdir(parents=True, exist_ok=True)
    dump_jsonl(out / "selected_review_questions_40.jsonl", review_sample)
    dump_jsonl(out / "reference_suitability_review_21.jsonl", reference_review)
    dump_csv(out / "selected_review_ids_40.csv", [{key: row[key] for key in ("review_position", "id", "original_id", "corpus", "source_group", "source_group_conservative")} for row in review_sample])
    dump_csv(out / "reference_suitability_review_21.csv", [{"id": row["id"], "original_id": row["original_id"],
        "question": row["question"], "published_answer_in_text": row["published_assessments_unchanged"].get("answer_in_text"),
        "published_original_reference": row["reference_answer_origin_published_unverified"],
        "expected_source_id_for_review_only": row["expected_source_id_for_review_only"],
        "human_reference_suitable": "", "human_evidence_sufficient": "", "human_question_meaningful": "",
        "reviewer": "", "notes": "", "status": "pending"} for row in reference_review])
    input_files = [args.dataset, args.supporting_passages, args.corpus, Path(__file__)]
    previous_audit = EXTERNAL / "overlap_pretest_decisions_20261017.json"
    if previous_audit.exists():
        input_files.append(previous_audit)
    output_files = [out / name for name in ("selected_review_questions_40.jsonl", "selected_review_ids_40.csv",
                    "reference_suitability_review_21.jsonl", "reference_suitability_review_21.csv")]
    manifest = {
        "plan_date_local": "2026-10-17", "timezone": "Asia/Shanghai", "seed": SEED,
        "status": "Local pretest plan prepared; root formal freeze and independent expert approval remain separate.",
        "registration_status": "Local filesystem record only. Not an OSF registration or other external preregistration.",
        "model_answers_read_or_generated": False, "development_model_scores_read": False, "gpu_used": False,
        "expert_approval_obtained": False, "expert_scores": "pending; no scores supplied or inferred",
        "inputs_sha256": {str(path.resolve()): sha(path) for path in input_files},
        "outputs_sha256": {path.name: sha(path) for path in output_files},
        "python_version_for_rng_reproduction": platform.python_version(),
        "sample": {"published_answer_in_text_eligible": 206, "selected_questions": SAMPLE_SIZE,
            "distinct_original_source_groups": len({row["source_group"] for row in selected}),
            "selected_original_ids": [row["original_id"] for row in selected],
            "stratification": sampling_details,
            "algorithm": ["Keep only published answer_in_text == 1; do not use model outputs, EM, F1 or citation scores.",
                "Allocate 40 slots proportionally to corpus 1/2 question counts by largest remainders; remainder ties by ascending corpus ID.",
                "Use one random.Random(20261017) stream, process corpus strata in ascending order.",
                "Within each stratum, sort source_group strings then shuffle, and take the stratum quota of sources.",
                "Within each selected source, sort questions by original_id and choose one using rng.randrange.",
                "No source receives a second question; fail rather than silently relaxing this condition."],
            "inclusion_boundary": "Stratified source-balanced sampling, not an equal-probability simple random sample of questions. Corpus 2 passage IDs do not establish independent source documents."},
        "remaining_reference_review": {"questions": 21,
            "published_answer_in_text_zero": sum((row.get("original_assessments") or {}).get("answer_in_text") == 0 for row in unsuitable),
            "published_answer_in_text_missing": sum((row.get("original_assessments") or {}).get("answer_in_text") is None for row in unsuitable),
            "policy": "Human reference, question and evidence-sufficiency audit only; do not treat these as a refusal gold standard or relabel without a separate decision."},
        "automatic_analysis_rules": {
            "primary_question_set": {"name": "official_full_227", "questions": 227, "exclusions": []},
            "fixed_sensitivity_question_sets": [
                {"name": "legacy_flag_excluded_225", "questions": 225, "exclude_original_ids": list(EXCLUDE_ORIGINAL_IDS)},
                {"name": "published_answer_in_text_1_206", "questions": 206, "filter": "published original_assessments.answer_in_text == 1"}],
            "same_original_ids_across_tasks": "Use original_id to apply exactly the same subsets in open QA and MC, despite task-specific ID prefixes.",
            "mcqa": {"main_metric": "leading-option accuracy; invalid extraction counts incorrect", "leading_option_regex": LEADING_MC_REGEX,
                "extractor": "After optional leading whitespace, accept one capital A-E followed by end-of-string, whitespace or the explicitly listed punctuation. Never search inside an answer or mistake the A in Answer for option A.",
                "strict_format_secondary": "After strip(), exactly one capital A-E. Report strict accuracy and invalid-format rate separately; no chain-of-thought or explanatory suffix is accepted by this metric."},
            "openqa": {"primary_automatic": "SQuAD-style token F1 against reference_answer_origin only, after removing recognizable square-bracket citation ID markers from the model answer.",
                "raw_answer_sensitivity": "Same original reference and normalization, without removing model citation markers.",
                "reference_sensitivity": "Published alternate references are optional sensitivity only; equivalence has not been independently validated.",
                "interpretation": "All EM/F1 measures are lexical overlap, not factual accuracy, completeness or evidence support."},
            "uncertainty": {"main_clusters": "source_group (known article EID/known-EID-or-exact-passage component where available; exact supporting passage otherwise)",
                "corpus2_conservative_sensitivity": "Repeat paired bootstrap with source_group_conservative; all unresolved corpus-2 rows belong to PIRA-UNRESOLVED-CORPUS2. Point estimates stay unchanged; independent-source uncertainty can differ.",
                "resamples": 2000, "bootstrap_seed": 20261017, "primary_difference": "lora_rag minus base_rag", "others": "secondary/exploratory"}},
        "human_review_rules": {"sample_frozen_before_outputs": True, "independent_reviewers": 2,
            "quality_scope": "All four conditions for the same 40 preselected questions; randomize opaque answer IDs independently per reviewer after actual answers exist.",
            "factual_correctness": "Human 0-4; 0 incorrect/missing, 1 major errors, 2 mixed/partial, 3 mostly correct with minor issues, 4 correct for the question.",
            "essential_completeness": "Human 0-4, separate from correctness; 0 absent, 1 major omissions, 2 partial, 3 most essential points, 4 all essential requested points.",
            "evidence_support": "RAG answers: human 0-4 using actual provided evidence; no-RAG answers: not applicable, never zero-filled.",
            "abstention_and_task_completion": {
                "quality_packet": "Score whether the response supplies the facts and essential points requested by the question. The 40 sampled questions have published answer_in_text=1, but the actual retrieved windows can still be insufficient. A missing answer or only saying insufficient information is not a fully correct completed answer; pure abstention that supplies none of the requested facts has no essential-point coverage. Real experts apply the rubric; no automatic score is inserted.",
                "evidence_packet": "An appropriately qualified statement of insufficient evidence can be faithful if the ACTUAL supplied passages really are insufficient. This evidence-faithfulness judgment is separate from completing the question, factual task-answer quality and completeness. An unsupported blanket refusal is not automatically faithful when the needed evidence is present.",
                "aggregation": "Do not let evidence-faithfulness for abstention stand in for factual task-answer quality or completeness. Report these dimensions separately rather than rewarding an always-abstain system as high overall answer quality.",
                "other_21_questions": "These rules do not make the other 21 records a refusal gold standard. Their published 0/missing labels and original references remain unchanged; any reference-suitability judgment is a separate pending human review."},
            "citation_counts": ["citation_links_count", "supported_citation_links_count", "claims_requiring_citation", "claims_with_supported_citation"],
            "reviewer_independence": "Two real experts review independently before any discussion; preserve initial ratings and document later adjudication separately.",
            "blinding": "No model/condition names in expert packets. Private mapping remains separate. Quality packet may strip citation markers; evidence packet preserves actual answer and contexts. Wording can still reveal behavior, so blinding is imperfect.",
            "missing_scores": "Blank/pending, never imputed as zero. No automated F1 or citation-validity score becomes an expert score.",
            "reference_problems": "Flag doubtful references or ambiguous questions for separate review; do not silently repair official labels or select exclusions using observed model scores."},
        "limitations": ["The 40-question sample cannot certify the whole benchmark; report its selected scope and source-balanced sampling.",
            "Unflagged test questions are not proof of unseen pretraining or complete historical SFT independence.",
            "Corpus-2 passage-level groups are unresolved fragments of larger report sources; the conservative grouping is a sensitivity, not recovered true lineage.",
            "No model answers, expert approval, external registration or expert ratings exist in these plan artifacts."]}
    dump_json(out / "manifest.json", manifest)
    print(json.dumps({"out": str(out), "selected": len(selected), "selected_by_corpus": dict(collections.Counter(row["corpus"] for row in selected)),
                      "reference_review_pending": len(reference_review), "model_answers_read": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()

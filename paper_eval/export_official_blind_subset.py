"""Export real official Open-QA answers for the frozen 40-question review plan.

No generation or training is performed. This export is a REVIEW-ONLY SUBSET,
not a new 227-question benchmark run. Only expert_review/ is shareable; raw
condition identities, parent artifacts and anonymous-ID mappings stay outside it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from .common import digest_file, read_jsonl, resolve, write_json, write_jsonl
from .english_report import (
    blind_csv,
    fair_comparison,
    requested_conditions,
    review_questions_with_sources,
    score_record,
)


DEFAULT_PLAN = "paper_data/external_benchmarks/pretest_review_plan_20261017"


def snapshot(path: Path) -> tuple[bytes, str]:
    value = path.read_bytes()
    return value, hashlib.sha256(value).hexdigest()


def jsonl_snapshot(path: Path) -> tuple[list[dict], str]:
    value, digest = snapshot(path)
    try:
        rows = [json.loads(line) for line in value.decode("utf-8-sig").splitlines() if line.strip()]
    except json.JSONDecodeError as exc:
        raise ValueError(f"Incomplete/invalid JSONL snapshot: {path}; retry only after its writer has finished") from exc
    return rows, digest


def verify_plan_file(plan: Path, manifest: dict, name: str) -> Path:
    target = plan / name
    expected = manifest.get("outputs_sha256", {}).get(name)
    if not expected or not target.is_file() or digest_file(target) != expected:
        raise ValueError(f"Frozen plan artifact missing or changed: {target}")
    return target


def planned_dataset_hash(manifest: dict) -> str:
    matches = [value for key, value in manifest.get("inputs_sha256", {}).items()
               if Path(key).name == "official_openqa_en_227.jsonl"]
    if len(matches) != 1:
        raise ValueError("Plan must identify one frozen official_openqa_en_227.jsonl input")
    return matches[0]


def planned_corpus_hash(manifest: dict) -> str:
    matches = [value for key, value in manifest.get("inputs_sha256", {}).items()
               if Path(key).name == "corpus_pira_chunks_en.jsonl"]
    if len(matches) != 1:
        raise ValueError("Plan must identify one frozen source-window corpus")
    return matches[0]


def export(run: Path, plan: Path, out: Path) -> None:
    run, plan, out = run.resolve(), plan.resolve(), out.resolve()
    if out.exists():
        raise FileExistsError(f"Output already exists; preserve any expert work and choose a new directory: {out}")
    parent_manifest_bytes, parent_manifest_hash = snapshot(run / "run_manifest.json")
    parent_manifest = json.loads(parent_manifest_bytes.decode("utf-8-sig"))
    parent_signature = parent_manifest.get("signature", {})
    protocol = parent_signature.get("protocol", {})
    if protocol.get("task") not in {"openqa", "open_qa"}:
        raise ValueError("The parent must be an official open-QA run")
    conditions = requested_conditions(run)
    # requested_conditions reads the manifest too; reject an in-progress manifest mutation.
    if digest_file(run / "run_manifest.json") != parent_manifest_hash:
        raise ValueError("Parent manifest changed during export preparation")
    plan_manifest_bytes, plan_manifest_hash = snapshot(plan / "manifest.json")
    plan_manifest = json.loads(plan_manifest_bytes.decode("utf-8-sig"))
    sample_path = verify_plan_file(plan, plan_manifest, "selected_review_questions_40.jsonl")
    suitability_csv = verify_plan_file(plan, plan_manifest, "reference_suitability_review_21.csv")
    sample = read_jsonl(sample_path)
    if len(sample) != 40 or len({row["id"] for row in sample}) != 40:
        raise ValueError("The frozen plan must contain exactly 40 unique question IDs")
    if [row["original_id"] for row in sample] != plan_manifest["sample"]["selected_original_ids"]:
        raise ValueError("Plan sample IDs/order differ from its manifest")
    if any((row.get("published_assessments_unchanged") or {}).get("answer_in_text") != 1 for row in sample):
        raise ValueError("The fixed review sample must preserve its original answer_in_text==1 selection")
    if parent_signature.get("dataset_sha256") != planned_dataset_hash(plan_manifest):
        raise ValueError("Parent dataset fingerprint differs from the plan's frozen official dataset")
    if parent_signature.get("corpus_sha256") != planned_corpus_hash(plan_manifest):
        raise ValueError("Parent supporting source windows differ from the plan's frozen corpus")
    parent_selected, parent_selected_hash = jsonl_snapshot(run / "selected_questions.jsonl")
    questions = {question["id"]: question for question in parent_selected}
    if len(parent_selected) != 227 or len(questions) != 227:
        raise ValueError("Parent selected_questions must contain all 227 unique official questions")
    if any(question.get("split") != "test" or question.get("choices") for question in parent_selected):
        raise ValueError("Parent selected_questions is not the official open-QA test")
    review_questions = {}
    for expected in sample:
        qid = expected["id"]
        if qid not in questions:
            raise ValueError(f"Planned review question absent from parent: {qid}")
        observed = questions[qid]
        checks = {
            "original_id": expected["original_id"], "question": expected["question"],
            "source_group": expected["source_group"],
            "expected_source_id": expected["expected_source_id_for_review_only"],
            "reference_answer_origin": expected["reference_answer_origin_published_unverified"],
            "corpus": expected["corpus"],
            "source_group_conservative": expected["source_group_conservative"],
        }
        if any(observed.get(key) != value for key, value in checks.items()):
            raise ValueError(f"Question/source/original reference changed after sample selection: {qid}")
        if observed.get("original_assessments") != expected["published_assessments_unchanged"]:
            raise ValueError(f"Published assessments changed after sample selection: {qid}")
        review_questions[qid] = observed
    parent_answers, parent_answers_hash = jsonl_snapshot(run / "answers.jsonl")
    answers = {}
    for row in parent_answers:
        key = (row["condition"], row["id"])
        if key[0] not in conditions or key[1] not in questions or key in answers:
            raise ValueError(f"Unexpected/duplicate answer in parent run: {key}")
        if row.get("task", "openqa") not in {"openqa", "open_qa"}:
            raise ValueError("Parent saved answer task is not open QA")
        if row.get("question") != questions[key[1]]["question"]:
            raise ValueError(f"Saved answer question differs from its frozen question: {key}")
        if not isinstance(row.get("answer"), str):
            raise ValueError(f"Missing raw answer text: {key}")
        answers[key] = row
    needed = [(condition, question["id"]) for question in sample for condition in conditions]
    missing = [key for key in needed if key not in answers]
    if missing:
        raise ValueError(f"All requested conditions for the fixed 40 questions must be complete; missing {len(missing)} examples: {missing[:5]}")
    subset = [answers[key] for key in needed]
    fairness = fair_comparison(subset)
    # Gold-source lookup happens only HERE, after authentic saved model answers
    # have been selected. No source is retrieved or inserted into a model prompt.
    enriched_questions, source_status = review_questions_with_sources(run, review_questions)
    if source_status.get("questions_with_source") != 40:
        raise ValueError("All 40 gold supporting sources must be available for independent human review")
    for planned in sample:
        observed_chunks = enriched_questions[planned["id"]].get("review_gold_source_chunks", [])
        if set(observed_chunks) != set(planned["supporting_text_locator"]["chunk_ids"]):
            raise ValueError(f"Review source windows differ from the preselected plan: {planned['id']}")
    scored = [score_record(row, review_questions[row["id"]], "openqa") for row in subset]
    parent_coverage = {condition: {
        "saved_answers": sum(key[0] == condition for key in answers), "expected_questions": 227,
        "complete_227": {(condition, qid) for qid in questions}.issubset(answers),
    } for condition in conditions}
    out.mkdir(parents=True, exist_ok=False)
    write_jsonl(out / "answers.jsonl", subset)
    write_jsonl(out / "selected_questions.jsonl", [review_questions[row["id"]] for row in sample])
    # blind_csv supplies blank expert fields and independent opaque-ID ordering.
    # Only the expert_review subdirectory is distributable to human reviewers.
    blind_csv(out, enriched_questions, scored)
    shared = out / "expert_review"
    shutil.copyfile(suitability_csv, shared / "reference_suitability_review_21.csv")
    with (shared / "README.md").open("a", encoding="utf-8") as stream:
        stream.write("\n## Frozen official-review subset and task completion\n\n"
            "These questions were selected before official outputs using the frozen 40-question plan. This folder contains no model identities. "
            "Give each reviewer only their own quality/evidence files; keep ratings independent before discussion. "
            "The separate 21-question CSV is for teacher-led reference/evidence-suitability review, not a refusal gold standard. Its original labels remain unchanged.\n\n"
            "Quality: judge whether the answer supplies the facts and essential points requested. A missing answer or only saying information is insufficient "
            "does not complete the question and must not receive full correct/complete credit. Pure abstention that supplies none of the requested facts "
            "has no essential-point coverage. Apply the rubric independently; no score has been automatically filled.\n\n"
            "Evidence: if the ACTUAL supplied passages truly are insufficient, an appropriately qualified insufficiency statement may be faithful. "
            "That is separate from task-answer correctness and completeness. When the needed evidence is present, a blanket refusal is not automatically faithful. "
            "Report these dimensions separately so an always-abstain system is not rewarded as high overall answer quality.\n\n"
            "No expert approval or rating is implied. Leave unreviewed fields blank. Published references may be questioned in the notes rather than silently edited.\n")
    private = out / "private"
    (private / "parent_run_manifest.json").write_bytes(parent_manifest_bytes)
    (private / "pretest_plan_manifest.json").write_bytes(plan_manifest_bytes)
    shutil.copyfile(sample_path, private / "frozen_review_sample_40.jsonl")
    (out / "README.md").write_text(
        "# Official-answer expert review export — subset only\n\n"
        f"This is a review-only export of {len(sample)} preselected questions × {len(conditions)} requested conditions = {len(subset)} authentic saved raw answers. "
        "It is NOT a new 227-question evaluation run and does not replace the parent run or publish whole-test scores. "
        "No model generation/training or expert scoring was performed by this exporter.\n\n"
        "Share only the expert_review subdirectory, distributing each reviewer's own files separately. "
        "Do NOT share the export root, answers.jsonl, selected_questions.jsonl, review_export_manifest.json or private directory: "
        "these preserve condition identities or model provenance and could break blinding. "
        "Private model mappings are kept separately. Answer wording may still reveal behavior; blinding is imperfect.\n\n"
        "The 21-question CSV is a separate pending reference-suitability review; it is not a refusal gold standard. "
        "Quality/task completion, completeness and evidence faithfulness must be rated separately as explained in the shared README. "
        "Unreviewed scores stay blank. No external preregistration or expert approval is claimed.\n",
        encoding="utf-8")
    code_files = [Path(__file__), Path(__file__).with_name("english_report.py"), Path(__file__).with_name("common.py")]
    code_snapshot = private / "export_code_snapshot"
    code_snapshot.mkdir()
    for source_code in code_files:
        shutil.copyfile(source_code, code_snapshot / source_code.name)
    outputs = {str(path.relative_to(out)): digest_file(path) for path in sorted(out.rglob("*")) if path.is_file()}
    export_manifest = {
        "artifact_type": "review_only_official_openqa_preselected_subset", "is_official_227_test_run": False,
        "subset_questions": 40, "requested_conditions": list(conditions), "raw_answers_exported": len(subset),
        "expected_answers_in_subset": 40 * len(conditions), "subset_complete": True,
        "expert_quality_rows_per_reviewer": len(subset),
        "expert_evidence_rows_per_reviewer": sum(row["condition"].endswith("_rag") for row in subset),
        "reference_suitability_questions": 21, "expert_scores_status": "blank/pending", "expert_approval_claimed": False,
        "model_generation_performed": False, "training_performed": False,
        "parent": {"run_path": str(run), "manifest_sha256": parent_manifest_hash,
            "signature_sha256": parent_signature.get("signature_sha256"), "signature": parent_signature,
            "answers_snapshot_sha256": parent_answers_hash, "selected_questions_snapshot_sha256": parent_selected_hash,
            "full_run_coverage": parent_coverage,
            "boundary": "Coverage is recorded from the authentic parent snapshot; this 40-question export does not stand in for 227-question results."},
        "plan": {"path": str(plan), "manifest_sha256": plan_manifest_hash,
            "sample_sha256": digest_file(sample_path), "seed": plan_manifest.get("seed"),
            "selected_original_ids": [row["original_id"] for row in sample],
            "frozen_dataset_sha256": planned_dataset_hash(plan_manifest),
            "frozen_corpus_sha256": planned_corpus_hash(plan_manifest),
            "registration_status": plan_manifest.get("registration_status")},
        "source_review": source_status, "matched_prompt_and_evidence_check": fairness,
        "gold_source_boundary": "Only looked up during this post-generation review export. Never used to alter retrieval, prompts, answers or training.",
        "code_sha256": {path.name: digest_file(path) for path in code_files}, "output_files_sha256": outputs,
        "sharing_boundary": "Only expert_review/ is shareable; raw outputs, signatures and private mappings remain outside it.",
        "human_abstention_rule": plan_manifest.get("human_review_rules", {}).get("abstention_and_task_completion"),
        "limitations": ["An independently selected 40-question review sample is not an expert audit of all 227 questions.",
            "Expert correctness/completeness/support remain pending; automatic overlap scores are not exported as human scores.",
            "Appropriate abstention may be evidence-faithful but does not complete the requested factual answer.",
            "The 21 unsuitable-or-unannotated references are not relabeled unanswerable.",
            "Opaque labels cannot hide all condition clues in answer wording."]}
    write_json(out / "review_export_manifest.json", export_manifest)
    print(f"Review-only export saved: {out}; {len(subset)} real answers, two blank independent review packets")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--plan", default=DEFAULT_PLAN, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    export(resolve(str(args.run)), resolve(str(args.plan)), resolve(str(args.out)))


if __name__ == "__main__":
    main()

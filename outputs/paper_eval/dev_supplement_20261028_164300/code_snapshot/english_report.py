"""Descriptive, source-clustered reporting for the English development runs.

This module does not load a model and never assigns factual correctness from
string overlap. Human scores remain blank until independent reviewers supply them.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import random
import re
import string
from pathlib import Path
from typing import Any

import numpy as np

from .common import CONDITIONS, CONDITION_NAMES, digest_file, digest_json, read_jsonl, resolve, write_json
from .report import clustered_interval


CITATION = re.compile(r"\[([A-Za-z0-9][A-Za-z0-9_.:/-]*(?:\s*[,;]\s*[A-Za-z0-9][A-Za-z0-9_.:/-]*)*)\]")
COMPARISONS = (("lora", "base"), ("base_rag", "base"), ("lora_rag", "base_rag"), ("lora_rag", "lora"))
# Frozen development sensitivity rule, separate from the strict main score.
# Require a boundary so ordinary words such as Answer/Because are not options.
MC_PREFIX_PATTERN = r"^([A-E])(?=$|[\s.,:;)\]}-])"
MC_PREFIX_RE = re.compile(MC_PREFIX_PATTERN)


def requested_conditions(run: Path) -> tuple[str, ...]:
    manifest_path = run / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    conditions = manifest.get("conditions", list(CONDITIONS))
    if (not isinstance(conditions, list) or not conditions
            or any(not isinstance(condition, str) for condition in conditions)
            or len(set(conditions)) != len(conditions)
            or any(condition not in CONDITIONS for condition in conditions)):
        raise ValueError("Manifest conditions must be a nonempty unique list of known conditions")
    return tuple(conditions)


def squad_normalize(value: str) -> str:
    """Official SQuAD-style English normalization: lower, punctuation, articles."""
    text = "".join(ch for ch in str(value).lower() if ch not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def squad_scores(answer: str, reference: str) -> tuple[float, float]:
    prediction, gold = squad_normalize(answer), squad_normalize(reference)
    exact = float(prediction == gold)
    predicted_tokens, gold_tokens = prediction.split(), gold.split()
    if not predicted_tokens or not gold_tokens:
        return exact, float(predicted_tokens == gold_tokens)
    common = sum((collections.Counter(predicted_tokens) & collections.Counter(gold_tokens)).values())
    if not common:
        return exact, 0.0
    precision, recall = common / len(predicted_tokens), common / len(gold_tokens)
    return exact, 2 * precision * recall / (precision + recall)


def citations(answer: str) -> list[str]:
    return [piece.strip() for match in CITATION.finditer(answer) for piece in re.split(r"[,;]", match.group(1))]


def remove_citations(answer: str) -> str:
    return CITATION.sub("", answer).strip()


def reference_strings(question: dict) -> tuple[str, list[str]]:
    primary = question.get("reference_answer_origin", question.get("reference_answer", ""))
    if not isinstance(primary, str) or not primary.strip():
        raise ValueError(f"Missing original answer reference: {question['id']}")
    secondary: list[str] = [primary]
    validation = question.get("reference_answer_validate")
    if isinstance(validation, str) and validation.strip():
        secondary.append(validation)
    released = question.get("reference_answers", [])
    if isinstance(released, str):
        released = [released]
    if isinstance(released, dict):
        released = released.get("text", list(released.values()))
    for value in released:
        if isinstance(value, str) and value.strip():
            secondary.append(value)
    return primary, list(dict.fromkeys(secondary))


def reference_source(question: dict) -> str | None:
    value = question.get("expected_source_id") or question.get("source_article_eid")
    return str(value) if value is not None and str(value).strip() else None


def source_hit(question: dict, context: list[dict]) -> bool | None:
    expected = reference_source(question)
    if expected is None:
        return None
    for chunk in context:
        # section is the runner's canonical supporting-passage identity.
        observed = chunk.get("section")
        if observed is None:
            observed = chunk.get("article_eid", chunk.get("source_article_eid"))
        if observed is not None and str(observed) == expected:
            return True
    return False


def review_questions_with_sources(run: Path, questions: dict) -> tuple[dict, dict]:
    """Read gold source identities AFTER generation, solely for human packets.

    Does not modify selected_questions.jsonl, retrieval, prompts or model inputs.
    Adjacent corpus windows are rejoined at their longest exact character overlap.
    """
    enriched = {qid: dict(question) for qid, question in questions.items()}
    manifest_path = run / "run_manifest.json"
    if not manifest_path.exists():
        return enriched, {"status": "unavailable_no_run_manifest"}
    signature = json.loads(manifest_path.read_text(encoding="utf-8")).get("signature", {})
    corpus_label = signature.get("corpus_path")
    if not corpus_label:
        return enriched, {"status": "unavailable_no_corpus_path"}
    corpus_path = resolve(corpus_label)
    if not corpus_path.exists():
        return enriched, {"status": "unavailable_corpus_missing", "corpus_path": str(corpus_path)}
    current_hash = digest_file(corpus_path)
    if signature.get("corpus_sha256") and signature["corpus_sha256"] != current_hash:
        raise ValueError("Review corpus differs from the frozen generation corpus")
    by_source: dict[str, list[dict]] = collections.defaultdict(list)
    for chunk in read_jsonl(corpus_path):
        by_source[str(chunk.get("section", ""))].append(chunk)
    found = 0
    for question in enriched.values():
        expected = reference_source(question)
        chunks = sorted(by_source.get(expected or "", []), key=lambda chunk: chunk["id"])
        if not chunks:
            continue
        text = ""
        used = []
        for chunk in chunks:
            value = chunk["text"]
            if value in text:
                used.append(chunk["id"])
                continue
            overlap = 0
            for size in range(min(len(text), len(value)), 15, -1):
                if text[-size:] == value[:size]:
                    overlap = size
                    break
            text += value[overlap:] if overlap else ("\n\n" if text else "") + value
            used.append(chunk["id"])
        question["supporting_text"] = text
        question["review_gold_source_chunks"] = used
        found += 1
    return enriched, {"status": "gold_sources_joined_for_human_review_only", "questions_with_source": found,
        "corpus_path": str(corpus_path), "corpus_sha256": current_hash,
        "boundary": "Gold source lookup occurs only in this post-generation report. Never used to retrieve or construct model prompts."}


def score_record(row: dict, question: dict, task: str) -> dict:
    answer = row.get("answer", "")
    if not isinstance(answer, str):
        raise ValueError(f"Non-text answer: {row['id']}")
    context = row.get("retrieved_context") or []
    markers = citations(answer)
    available = {str(chunk["id"]) for chunk in context}
    chinese = len(re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff]", answer))
    latin = len(re.findall(r"[A-Za-z]", answer))
    letter_denominator = chinese + latin
    fraction = chinese / letter_denominator if letter_denominator else 0.0
    scored = {"id": row["id"], "condition": row["condition"], "source_group": question["source_group"],
        "answer": answer, "citation_markers": len(markers), "citation_markers_unique": len(set(markers)),
        "citation_markers_resolved": sum(marker in available for marker in markers),
        "citation_markers_unresolved": [marker for marker in markers if marker not in available],
        "cjk_characters": chinese, "cjk_fraction_among_latin_and_cjk": fraction,
        "contains_cjk": chinese > 0, "obvious_cjk_output_flag": chinese >= 8 or fraction > 0.05,
        "empty_answer": not answer.strip(),
        "gold_source_recall_at_k": source_hit(question, context) if row["condition"].endswith("_rag") else None,
        "retrieved_chunks_shown": len(context), "retrieved_context": context}
    if task == "mcqa":
        choices = question["choices"]
        permitted = set(choices) if isinstance(choices, dict) else set("ABCDE"[:len(choices)])
        prediction = answer.strip()
        valid = bool(re.fullmatch(r"[A-E]", prediction)) and prediction in permitted
        gold = question.get("answer")
        if gold not in permitted:
            raise ValueError(f"Invalid gold MC letter: {question['id']}: {gold}")
        prefix_match = MC_PREFIX_RE.match(prediction)
        prefix = prefix_match.group(1) if prefix_match and prefix_match.group(1) in permitted else None
        prefix_correct = bool(prefix == gold)
        scored.update(mc_valid_format=valid, mc_prediction=prediction if valid else None,
                      accuracy=float(valid and prediction == gold), gold_letter=gold,
                      mc_prefix_parseable=prefix is not None, mc_prefix_prediction=prefix,
                      mc_prefix_sensitivity_accuracy=float(prefix_correct),
                      mc_prefix_format_only_rescued=bool(prefix_correct and not (valid and prediction == gold)))
    else:
        primary, references = reference_strings(question)
        em, f1 = squad_scores(answer, primary)
        stripped_em, stripped_f1 = squad_scores(remove_citations(answer), primary)
        multi = [squad_scores(answer, reference) for reference in references]
        scored.update(exact_match=em, token_f1=f1, citation_stripped_exact_match=stripped_em,
            citation_stripped_token_f1=stripped_f1,
            published_reference_sensitivity_exact_match=max(v[0] for v in multi),
            published_reference_sensitivity_token_f1=max(v[1] for v in multi),
            published_references_count=len(references), reference_answer_origin=primary)
        if scored["gold_source_recall_at_k"] is False:
            scored["diagnostic_bucket"] = "gold_source_not_retrieved"
        elif scored["gold_source_recall_at_k"] is True and stripped_f1 < 0.5:
            scored["diagnostic_bucket"] = "gold_source_retrieved_low_overlap_proxy"
        elif scored["gold_source_recall_at_k"] is True:
            scored["diagnostic_bucket"] = "gold_source_retrieved_other"
        else:
            scored["diagnostic_bucket"] = "no_retrieval_or_gold_source_unavailable"
    return scored


def describe(values: list[Any]) -> dict:
    usable = [float(v) for v in values if v is not None and isinstance(v, (int, float)) and np.isfinite(v)]
    if not usable:
        return {"n": 0, "mean": None, "p50": None, "p95": None, "max": None}
    return {"n": len(usable), "mean": float(np.mean(usable)), "p50": float(np.median(usable)),
            "p95": float(np.percentile(usable, 95)), "max": max(usable)}


def performance(rows: list[dict], conditions: tuple[str, ...] = CONDITIONS) -> dict:
    summary = {}
    top_level = ("pipeline_ms", "pipeline_ttft_ms", "retrieval_ms")
    metric_keys = ("generation_ms", "model_ttft_ms", "decode_tokens_per_second", "input_tokens",
        "output_tokens_including_eos", "output_characters", "resident_allocated_mib", "peak_allocated_mib",
        "peak_reserved_mib", "incremental_peak_allocated_mib", "process_rss_mib")
    for condition in conditions:
        selected = [row for row in rows if row["condition"] == condition]
        values = {key: describe([row.get(key) for row in selected]) for key in top_level}
        values.update({key: describe([(row.get("metrics") or {}).get(key) for row in selected]) for key in metric_keys})
        truncations = [(row.get("metrics") or {}).get("hit_token_limit") for row in selected]
        values["hit_token_limit_fraction"] = float(np.mean([v for v in truncations if v is not None])) if any(v is not None for v in truncations) else None
        values["records"] = len(selected)
        summary[condition] = values
    return summary


def fair_comparison(rows: list[dict]) -> dict:
    by_key = {(row["condition"], row["id"]): row for row in rows}
    checked, missing_hashes = [], []
    for left, right in (("base", "lora"), ("base_rag", "lora_rag")):
        ids = sorted(set(qid for condition, qid in by_key if condition == left) & set(qid for condition, qid in by_key if condition == right))
        for qid in ids:
            a, b = by_key[(left, qid)], by_key[(right, qid)]
            # Full selected chunk dictionaries include scores/timing-independent metadata.
            # The actual ordered IDs and text define the evidence supplied to generation.
            if left.endswith("_rag"):
                context_a = [(chunk.get("id"), chunk.get("text")) for chunk in a.get("retrieved_context", [])]
                context_b = [(chunk.get("id"), chunk.get("text")) for chunk in b.get("retrieved_context", [])]
                if context_a != context_b:
                    raise ValueError(f"RAG evidence differs between models: {qid}")
            for field in ("context_sha256", "prompt_sha256"):
                if a.get(field) is not None and b.get(field) is not None:
                    if a[field] != b[field]:
                        raise ValueError(f"Unfair {field}: {left}/{right}/{qid}")
                else:
                    missing_hashes.append({"id": qid, "pair": f"{left}/{right}", "field": field})
            checked.append({"id": qid, "pair": f"{left}/{right}"})
    return {"checked_pairs": len(checked), "missing_hash_checks": missing_hashes,
            "status": "hashes_and_actual_context_consistent" if not missing_hashes else "actual_context_consistent_some_hash_checks_unavailable"}


def metric_intervals(scored: list[dict], task: str, conditions: tuple[str, ...] = CONDITIONS,
                     expected_ids: set[str] | None = None) -> tuple[dict, dict]:
    metrics = ["accuracy", "mc_prefix_sensitivity_accuracy"] if task == "mcqa" else ["exact_match", "token_f1", "citation_stripped_exact_match", "citation_stripped_token_f1",
        "published_reference_sensitivity_exact_match", "published_reference_sensitivity_token_f1"]
    summaries = {}
    for condition in conditions:
        values = [row for row in scored if row["condition"] == condition]
        summaries[condition] = {metric: clustered_interval([(row["source_group"], row[metric]) for row in values]) for metric in metrics}
        summaries[condition].update(records=len(values),
            obvious_cjk_output_fraction=float(np.mean([row["obvious_cjk_output_flag"] for row in values])) if values else None,
            contains_cjk_fraction=float(np.mean([row["contains_cjk"] for row in values])) if values else None,
            empty_answer_fraction=float(np.mean([row["empty_answer"] for row in values])) if values else None)
        if task == "mcqa":
            summaries[condition]["invalid_format_fraction"] = float(np.mean([not row["mc_valid_format"] for row in values])) if values else None
            full_set = expected_ids is None or {row["id"] for row in values} == expected_ids
            summaries[condition].update(
                mc_prefix_parseable_records=sum(row["mc_prefix_parseable"] for row in values),
                mc_prefix_correct_records=int(sum(row["mc_prefix_sensitivity_accuracy"] for row in values)),
                mc_prefix_format_only_rescued_records=sum(row["mc_prefix_format_only_rescued"] for row in values),
                mc_prefix_sensitivity_denominator=len(expected_ids) if expected_ids is not None else len(values),
                mc_prefix_sensitivity_status="complete_all_selected_questions" if full_set else "withheld_until_condition_complete")
            if not full_set:
                # Do not turn missing generations into wrong answers or silently
                # substitute an easier/compliant/finished subset denominator.
                summaries[condition]["mc_prefix_sensitivity_accuracy"] = clustered_interval([])
        markers = sum(row["citation_markers"] for row in values)
        resolved = sum(row["citation_markers_resolved"] for row in values)
        summaries[condition]["citation_diagnostics"] = {"markers": markers, "resolved_markers": resolved,
            "resolvable_marker_fraction": resolved / markers if markers else None,
            "answers_with_any_marker_fraction": float(np.mean([row["citation_markers"] > 0 for row in values])) if values else None}
        recalled = [(row["source_group"], float(row["gold_source_recall_at_k"])) for row in values if row["gold_source_recall_at_k"] is not None]
        summaries[condition]["gold_source_recall_at_k"] = clustered_interval(recalled)
        summaries[condition]["diagnostic_buckets"] = dict(collections.Counter(row.get("diagnostic_bucket", "mcqa") for row in values))
    indexed = {(row["condition"], row["id"]): row for row in scored}
    differences = {}
    for left, right in COMPARISONS:
        if left not in conditions or right not in conditions:
            continue
        paired_ids = sorted(set(qid for condition, qid in indexed if condition == left) & set(qid for condition, qid in indexed if condition == right))
        differences[f"{left}_minus_{right}"] = {metric: clustered_interval([
            (indexed[(left, qid)]["source_group"], indexed[(left, qid)][metric] - indexed[(right, qid)][metric])
            for qid in paired_ids]) for metric in metrics}
    return summaries, differences


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]) if rows else fields or ["status"])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value for key, value in row.items()})


def failure_checklist(run: Path, questions: dict, scored: list[dict], task: str,
                      conditions: tuple[str, ...] = CONDITIONS) -> None:
    grouped = {(row["condition"], row["id"]): row for row in scored}
    checklist = []
    for qid, question in questions.items():
        item = {"id": qid, "source_group": question["source_group"], "question": question["question"],
            "reference_answer_origin": question.get("reference_answer_origin", question.get("reference_answer_text", "")),
            "choices": question.get("choices", ""), "gold_letter": question.get("answer", ""),
            "expected_source_id": reference_source(question), "supporting_text": question.get("supporting_text", ""),
            "human_reference_verified": "", "human_failure_cause": "", "human_notes": ""}
        for condition in conditions:
            row = grouped.get((condition, qid))
            item.update({f"{condition}_answer": row["answer"] if row else "",
                f"{condition}_gold_source_hit": row["gold_source_recall_at_k"] if row else "",
                f"{condition}_citation_markers": row["citation_markers"] if row else "",
                f"{condition}_citation_resolved": row["citation_markers_resolved"] if row else "",
                f"{condition}_automatic_{'accuracy' if task == 'mcqa' else 'token_f1'}": row["accuracy" if task == "mcqa" else "token_f1"] if row else "",
                f"{condition}_diagnostic_bucket_proxy": row.get("diagnostic_bucket", "") if row else "",
                f"{condition}_actual_context_shown": row["retrieved_context"] if row else ""})
            if task == "mcqa":
                item.update({f"{condition}_prefix_sensitivity_prediction": row["mc_prefix_prediction"] if row else "",
                    f"{condition}_prefix_sensitivity_accuracy": row["mc_prefix_sensitivity_accuracy"] if row else "",
                    f"{condition}_prefix_format_only_rescued": row["mc_prefix_format_only_rescued"] if row else ""})
        checklist.append(item)
    write_csv(run / "failure_checklist.csv", checklist)


def blind_csv(run: Path, questions: dict, scored: list[dict]) -> None:
    """Two independent randomized packets. Model mapping stays outside shared files."""
    target = run / "expert_review"
    target.mkdir(exist_ok=True)
    private = run / "private"
    private.mkdir(exist_ok=True)
    answer_hash = digest_file(run / "answers.jsonl")
    mapping_path = private / "english_blind_mapping.json"
    if mapping_path.exists():
        prior = json.loads(mapping_path.read_text(encoding="utf-8"))
        if prior.get("answers_sha256") == answer_hash:
            # Preserve human-entered work if a report is regenerated.
            return
        raise ValueError("Answers changed after blind packets were created; use a fresh run directory")
    mapping: dict = {"answers_sha256": answer_hash, "review_status": "pending", "packets": {}}
    for reviewer in (1, 2):
        rng = random.Random(int(answer_hash[:16], 16) + reviewer)
        ordered = list(scored)
        rng.shuffle(ordered)
        quality_rows, evidence_rows, links = [], [], {}
        for index, row in enumerate(ordered, 1):
            opaque = digest_json({"hash": answer_hash, "reviewer": reviewer, "index": index})[:16]
            question = questions[row["id"]]
            links[opaque] = {"id": row["id"], "condition": row["condition"]}
            quality_rows.append({"item_id": opaque, "question": question["question"],
                "reference_answer_published_unverified": question.get("reference_answer_origin", question.get("reference_answer_text", "")),
                "supporting_passage_for_review": question.get("supporting_text", ""),
                "candidate_answer_citation_markers_removed": remove_citations(row["answer"]),
                "factual_correctness_0_to_4": "", "essential_completeness_0_to_4": "",
                "reference_problem_or_ambiguity": "", "notes": ""})
            if row["condition"].endswith("_rag"):
                evidence_rows.append({"item_id": opaque, "question": question["question"], "candidate_answer": row["answer"],
                    "actual_evidence_provided": row["retrieved_context"], "evidence_faithfulness_0_to_4": "",
                    "citation_links_count": "", "supported_citation_links_count": "",
                    "claims_requiring_citation": "", "claims_with_supported_citation": "", "notes": ""})
        write_csv(target / f"reviewer_{reviewer}_quality.csv", quality_rows)
        write_csv(target / f"reviewer_{reviewer}_evidence.csv", evidence_rows)
        mapping["packets"][f"reviewer_{reviewer}"] = links
    write_json(mapping_path, mapping)
    (target / "README.md").write_text("# Independent expert review — scores pending\n\n"
        "Each reviewer completes only their own quality and evidence CSVs independently. Do not share the private directory. "
        "The item identifiers are randomized and model identities are omitted. Answer wording may still reveal system behavior; blinding is imperfect.\n\n"
        "Quality scores: 0 entirely incorrect/missing, 1 mostly incorrect or major omissions, 2 mixed/partial, 3 mostly correct with minor limitations, "
        "4 correct and complete for the question. Score factual correctness and essential completeness separately; flag dubious published references "
        "instead of treating them as unquestionable truth. Citation markers are removed from quality-only answers; originals are preserved in answers.jsonl.\n\n"
        "Evidence scores: 0 unsupported/contradicted, 1 mostly unsupported, 2 mixed, 3 mostly supported, 4 fully supported by supplied evidence. "
        "Count actual citation links and claims rather than copying the automatic marker count. Valid source IDs alone do not demonstrate support. "
        "Leave unreviewed items blank. Do not enter 0 for missing reviews. These templates have no imported human ratings.\n", encoding="utf-8")


def fmt(value: Any, percentage: bool = False) -> str:
    if value is None:
        return "—"
    return f"{100 * value:.1f}%" if percentage else f"{value:.2f}"


def interval_text(value: dict) -> str:
    ci = value.get("ci95")
    return fmt(value["mean"], True) + (f" [{100 * ci[0]:.1f}, {100 * ci[1]:.1f}]" if ci else " [CI unavailable]")


def report(run: Path) -> None:
    run = run.resolve()
    conditions = requested_conditions(run)
    rows = read_jsonl(run / "answers.jsonl")
    selected = read_jsonl(run / "selected_questions.jsonl")
    if not selected:
        raise ValueError("No selected development questions")
    questions = {q["id"]: q for q in selected}
    if len(questions) != len(selected):
        raise ValueError("Duplicate selected question IDs")
    tasks = {"mcqa" if question.get("choices") else "openqa" for question in selected}
    if len(tasks) != 1:
        raise ValueError("Mixed tasks must be reported in separate run directories")
    task = tasks.pop()
    if len({(r["condition"], r["id"]) for r in rows}) != len(rows):
        raise ValueError("Duplicate condition/question results")
    for row in rows:
        if row["id"] not in questions or row["condition"] not in conditions:
            raise ValueError("Unknown result question or a condition not requested by the run manifest")
    expected_ids = set(questions)
    coverage = {}
    for condition in conditions:
        observed_ids = {row["id"] for row in rows if row["condition"] == condition}
        coverage[condition] = {"completed_questions": len(observed_ids), "selected_questions": len(expected_ids),
            "complete_identical_question_set": observed_ids == expected_ids,
            "missing_question_ids": sorted(expected_ids - observed_ids)}
    expected = len(questions) * len(conditions)
    complete = len(rows) == expected and all(value["complete_identical_question_set"] for value in coverage.values())
    fairness = fair_comparison(rows)
    scored = [score_record(row, questions[row["id"]], task) for row in rows]
    review_questions, review_source_status = review_questions_with_sources(run, questions)
    summary, differences = metric_intervals(scored, task, conditions, expected_ids)
    costs = performance(rows, conditions)
    splits = sorted({str(q.get("split", "development_unspecified")) for q in selected})
    warnings = ["This is an English DEVELOPMENT diagnosis, not the held-out official test result.",
        "SQuAD EM/token F1 measure string overlap, not factual accuracy, semantic equivalence, completeness or citation entailment.",
        "Primary open-QA scores use only reference_answer_origin and the original generated answer, including citation markers.",
        "Citation-stripped metrics are separate secondary diagnostics; published-reference sensitivity takes maxima across released references whose equivalence has not been independently validated.",
        "Gold-source recall checks source identity among actual shown contexts; it is not an answer factuality score and does not guarantee relevant text survived truncation.",
        "CJK flags are simple script diagnostics, not a validated language identification metric.",
        "Costs are descriptive natural-answer measurements on one device. Output length, input length, retrieval and model order confound latency; no equal-work probe means no inference-speed improvement claim.",
        "CUDA allocator measurements are not total GPU memory. p95 from a small sample is unstable.",
        "Human correctness, completeness and evidence support ratings remain pending; no missing rating is converted to zero."]
    if task == "mcqa":
        warnings += ["Strict single-uppercase-letter accuracy remains the main MC metric. Prefix parsing is a separate development sensitivity rule applied identically to every requested condition.",
            "Prefix sensitivity uses all selected questions once a condition is complete; unparseable generated answers count as incorrect. Its aggregate is withheld for incomplete conditions, with missing generations never converted to zero. Paired sensitivity differences use only questions completed in both conditions.",
            "The prefix rule was fixed after inspecting the first development outputs; it is a post-hoc development sensitivity, not a newly verified factual or semantic accuracy measure."]
    analysis = {"task": task, "split_labels": splits, "development_only": True,
        "conditions": list(conditions), "condition_coverage": coverage,
        "selected_questions": len(questions), "source_groups": len({q["source_group"] for q in selected}),
        "expected_answers": expected, "completed_answers": len(rows), "complete": complete,
        "artifacts": {"answers_sha256": digest_file(run / "answers.jsonl"), "selected_questions_sha256": digest_file(run / "selected_questions.jsonl")},
        "fairness": fairness, "review_material_source": review_source_status,
        "automatic_overlap_or_mc_metrics": summary, "paired_source_cluster_bootstrap_differences": differences,
        "cost_descriptive": costs, "expert_review": {"status": "pending", "factual_correctness": None, "completeness": None, "evidence_support": None},
        "confidence_interval_method": "2000 paired source_group cluster bootstrap resamples; question-weighted means; seed 20261016; undefined with fewer than two source groups.",
        "warnings": warnings, "per_answer_automatic_diagnostics": scored}
    if task == "mcqa":
        analysis["mc_format_sensitivity_method"] = {"pattern": MC_PREFIX_PATTERN,
            "preprocessing": "Strip leading and trailing whitespace; case-sensitive anchored match; require a provided option letter.",
            "strict_main_metric_unchanged": True, "development_post_hoc": True,
            "unparseable_generated_answers": "incorrect", "denominator": "all selected questions per complete condition",
            "incomplete_condition": "aggregate withheld; missing generations remain missing",
            "paired_differences": "only shared completed question IDs"}
    write_json(run / "analysis.json", analysis)
    failure_checklist(run, review_questions, scored, task, conditions)
    if task == "openqa" and scored and complete:
        blind_csv(run, review_questions, scored)
    if task == "mcqa":
        sensitivity_rows = [{"id": row["id"], "condition": row["condition"], "source_group": row["source_group"],
            "raw_answer": row["answer"], "gold_letter": row["gold_letter"],
            "strict_valid_format": row["mc_valid_format"], "strict_prediction": row["mc_prediction"],
            "strict_accuracy": row["accuracy"], "sensitivity_rule": MC_PREFIX_PATTERN,
            "prefix_parseable": row["mc_prefix_parseable"], "prefix_prediction": row["mc_prefix_prediction"],
            "prefix_sensitivity_accuracy": row["mc_prefix_sensitivity_accuracy"],
            "prefix_format_only_rescued": row["mc_prefix_format_only_rescued"]} for row in scored]
        write_csv(run / "mc_format_sensitivity.csv", sensitivity_rows)
    flat_costs = []
    for condition, values in costs.items():
        flat = {"condition": condition, "records": values["records"], "hit_token_limit_fraction": values["hit_token_limit_fraction"]}
        flat.update({f"{metric}_{stat}": number for metric, item in values.items() if isinstance(item, dict) for stat, number in item.items()})
        flat_costs.append(flat)
    write_csv(run / "performance_summary.csv", flat_costs)
    lines = ["# 英文开发集诊断报告", "", f"任务：{task}；开发题 {len(questions)}；来源组 {analysis['source_groups']}；已完成 {len(rows)}/{expected} 条回答。", "",
        "本次请求条件：" + "、".join(CONDITION_NAMES[condition] for condition in conditions) + f"；请求范围内完整：{'是' if complete else '否'}。", "",
        "本报告用于选择训练和检索设置，不是Pirá正式测试集成绩。专家事实、完整性和引用支持评分仍为pending。", ""]
    if task == "openqa":
        lines += ["## 原始发布答案的自动重合度", "", "主指标只使用原始参考答案，保留模型答案中的引用标记。分数是文本重合度，不是事实准确率。区间为按来源组bootstrap的95%区间，单位为百分比。", "",
            "| 条件 | 完成数 | EM [95% CI] | Token F1 [95% CI] | 去引用F1（次要） | 公开多参考F1（敏感性） |", "|---|---:|---:|---:|---:|---:|"]
        for condition in conditions:
            data = summary[condition]
            lines.append(f"| {CONDITION_NAMES[condition]} | {data['records']} | {interval_text(data['exact_match'])} | {interval_text(data['token_f1'])} | {fmt(data['citation_stripped_token_f1']['mean'], True)} | {fmt(data['published_reference_sensitivity_token_f1']['mean'], True)} |")
        lines += ["", "多参考敏感性结果取发布参考答案的最高重合度，标签等价性尚未独立审核，不能替代原始答案主结果或专家正确性。", ""]
        main_metric = "token_f1"
    else:
        lines += ["## 严格选择题", "", "仅剥除首尾空白后单个大写A—E且属于该题选项的答案有效。解释、括号、小写或额外内容均计格式无效并计错。", "",
            "| 条件 | 完成数 | 准确率 [95% CI] | 无效格式率 |", "|---|---:|---:|---:|"]
        for condition in conditions:
            data = summary[condition]
            lines.append(f"| {CONDITION_NAMES[condition]} | {data['records']} | {interval_text(data['accuracy'])} | {fmt(data['invalid_format_fraction'], True)} |")
        lines += ["", "### 格式敏感性（次要，严格主评分保持原样）", "",
            "四组或本次请求的各组均用同一开头字母规则：剥除首尾空白，匹配一个大写A—E，后面必须为结束、空白或规定标点，并确认属于题目选项。无法解析的已生成回答计错。", "",
            "```regex", MC_PREFIX_PATTERN, "```", "",
            "此规则不会把Answer、Because等普通英文词的首字母当作选项，也不提取位于解释中间的字母。每个完整条件以全部选中题为分母；不筛选格式合规题。尚未完整的条件暂不发布该整题均值，缺失回答不计0分。", "",
            "| 条件 | 已解析 / 已完成 | 敏感性准确率 [95% CI] | 全题目标分母 | 放宽格式后新增正确 | 状态 |", "|---|---:|---:|---:|---:|---|"]
        for condition in conditions:
            data = summary[condition]
            lines.append(f"| {CONDITION_NAMES[condition]} | {data['mc_prefix_parseable_records']}/{data['records']} | {interval_text(data['mc_prefix_sensitivity_accuracy'])} | {data['mc_prefix_sensitivity_denominator']} | {data['mc_prefix_format_only_rescued_records']} | {data['mc_prefix_sensitivity_status']} |")
        lines += ["", "该规则在第一版开发输出检查后确定，属于事后敏感性分析。它反映格式罚分对结果的影响，不评价解释正文是否正确，也不修正公开标签。不能替代严格主结果或独立人工审核；逐条结果见mc_format_sensitivity.csv。", ""]
        main_metric = "accuracy"
    lines += ["", "## 配对差异", "", "各差值只用两个条件均有回答的题，按source_group一起重采样；属于开发集探索性分析。", "", "| 比较 | 配对题数 | 差值及95% CI（百分点） |", "|---|---:|---:|"]
    for pair, values in differences.items():
        left, right = pair.split("_minus_")
        data = values[main_metric]
        lines.append(f"| {CONDITION_NAMES[left]} − {CONDITION_NAMES[right]} | {data['questions']} | {interval_text(data)} |")
    if task == "mcqa":
        lines += ["", "### 格式敏感性的配对差异", "",
            "仅使用两个请求条件均有回答的共有题，分母为配对题数；与上面的完整条件全题均值分开解释。", "",
            "| 比较 | 配对题数 | 敏感性差值及95% CI（百分点） |", "|---|---:|---:|"]
        for pair, values in differences.items():
            left, right = pair.split("_minus_")
            data = values["mc_prefix_sensitivity_accuracy"]
            lines.append(f"| {CONDITION_NAMES[left]} − {CONDITION_NAMES[right]} | {data['questions']} | {interval_text(data)} |")
    lines += ["", "## 检索、引用和语言诊断", "", "| 条件 | Gold来源召回@实际显示k | 引用数 | 可解析引用比例 | 明显含中文比例 |", "|---|---:|---:|---:|---:|"]
    for condition in conditions:
        data = summary[condition]
        lines.append(f"| {CONDITION_NAMES[condition]} | {fmt(data['gold_source_recall_at_k']['mean'], True)} | {data['citation_diagnostics']['markers']} | {fmt(data['citation_diagnostics']['resolvable_marker_fraction'], True)} | {fmt(data['obvious_cjk_output_fraction'], True)} |")
    lines += ["", "来源命中按chunk.section与expected_source_id相等计算，只是检索诊断。资料编号可解析也不代表它支持陈述。实际截断后送入模型的原文保存在answers.jsonl和failure_checklist.csv。", "",
        "## 自然回答实测成本", "", "| 条件 | 输出token均值 | 端到端ms均值 / p50 / p95 | 首tokenms均值 / p50 / p95 | 解码token/s均值 | PyTorch峰值MiB最大值 | 截断率 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for condition in conditions:
        data = costs[condition]
        latency = " / ".join(fmt(data["pipeline_ms"][key]) for key in ("mean", "p50", "p95"))
        ttft = " / ".join(fmt(data["pipeline_ttft_ms"][key]) for key in ("mean", "p50", "p95"))
        lines.append(f"| {CONDITION_NAMES[condition]} | {fmt(data['output_tokens_including_eos']['mean'])} | {latency} | {ttft} | {fmt(data['decode_tokens_per_second']['mean'])} | {fmt(data['peak_allocated_mib']['max'])} | {fmt(data['hit_token_limit_fraction'], True)} |")
    lines += ["", "完整成本的均值、p50、p95和样本量见performance_summary.csv；显存仅为PyTorch分配器，不是整卡占用。未使用固定工作量探针，不能从回答耗时下降推断推理效率改善。", "",
        "## 复核材料与边界", "", f"提示词/证据一致性检查：{fairness['status']}；已检查配对 {fairness['checked_pairs']} 个。", "",
        "failure_checklist.csv含各条件原始答案、参考答案、实际证据和自动诊断，人工结论栏留空。完整开放问答运行结束后才生成expert_review目录中的两位评审独立匿名CSV；private目录的模型映射不能发给评审。", "",
        "“gold来源未检索到”和“已命中但F1低”只是失败定位线索；低F1可能来自措辞差异或参考标签问题。开发集参考答案、来源标注和适用性仍需人工审核。", "",
        "缺失回答不当作0分；各条件均值基于实际完成题，只有完整且相同题集合时才可直接横比。配对差异只使用共有题。当前人工评分未导入。", ""]
    (run / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"English {task} development report: {run / 'report.md'} ({len(rows)}/{expected} answers); expert review pending")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    report(args.run)


if __name__ == "__main__":
    main()

"""Compare existing English OPEN-QA development runs without generating answers.

All tables use the intersection of selected question IDs. Missing condition
answers stay missing. Automatic overlap, citation syntax and script diagnostics
do not supply expert factual, completeness or evidence-support scores.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np

from .common import CONDITIONS, CONDITION_NAMES, digest_file, digest_json, read_jsonl, resolve, write_json, write_jsonl
from .english_report import score_record, write_csv
from .report import clustered_interval


def artifact_inventory(signature: dict, label: str | None) -> dict[str, str]:
    if not label:
        return {}
    folder = str(resolve(label).resolve()).replace("\\", "/").rstrip("/").casefold()
    selected = {}
    for key, value in signature.get("model_files", {}).items():
        absolute = str(resolve(key).resolve()).replace("\\", "/")
        if absolute.casefold().startswith(folder + "/"):
            selected[absolute[len(folder) + 1:]] = value
    return dict(sorted(selected.items()))


def load_run(path: Path) -> dict:
    path = path.resolve()
    manifest = json.loads((path / "run_manifest.json").read_text(encoding="utf-8-sig"))
    signature = manifest.get("signature", {})
    protocol = signature.get("protocol", {})
    selected = read_jsonl(path / "selected_questions.jsonl")
    questions = {question["id"]: question for question in selected}
    if len(questions) != len(selected) or not selected:
        raise ValueError(f"Empty or duplicate selected question IDs: {path}")
    for question in selected:
        if question.get("choices"):
            raise ValueError(f"Only open QA is accepted, not MC: {path}")
        if question.get("split") not in {"dev", "development"}:
            raise ValueError(f"Only DEVELOPMENT records are accepted: {question['id']} in {path}")
        if not question.get("source_group"):
            raise ValueError(f"Missing source_group: {question['id']}")
    # Read once so a concurrently appended run cannot change the hash after the
    # rows used in this comparison have already been captured.
    answer_bytes = (path / "answers.jsonl").read_bytes()
    try:
        answers = [json.loads(line) for line in answer_bytes.decode("utf-8-sig").splitlines() if line.strip()]
    except json.JSONDecodeError as exc:
        raise ValueError(f"Answers snapshot contains an incomplete JSONL line; retry after the active write finishes: {path}") from exc
    indexed = {}
    for row in answers:
        key = (row["condition"], row["id"])
        if key[0] not in CONDITIONS or key[1] not in questions or key in indexed:
            raise ValueError(f"Unexpected/duplicate saved condition/question: {key} in {path}")
        if row.get("task", "openqa") not in {"openqa", "open_qa"}:
            raise ValueError(f"Saved task is not open QA: {path}")
        indexed[key] = row
    adapter_files = artifact_inventory(signature, protocol.get("adapter"))
    base_files = artifact_inventory(signature, protocol.get("base_model"))
    weights = {key: value for key, value in adapter_files.items() if key.endswith((".safetensors", ".bin"))}
    metadata = {
        "run": path.name, "path": str(path), "selected_questions": len(questions), "saved_answers": len(answers),
        "source_groups_selected": len({question["source_group"] for question in selected}),
        "actual_conditions": [condition for condition in CONDITIONS if any(key[0] == condition for key in indexed)],
        "adapter_path_from_frozen_protocol": protocol.get("adapter"),
        "adapter_artifact_inventory_sha256": digest_json(adapter_files) if adapter_files else None,
        "adapter_weight_files_sha256": weights, "adapter_artifact_files_sha256": adapter_files,
        "base_model_path": protocol.get("base_model"),
        "base_artifact_inventory_sha256": digest_json(base_files) if base_files else None,
        "prompt_profile": protocol.get("prompt_profile", "original"),
        "retrieval_method": (protocol.get("retrieval") or {}).get("method", "dense"),
        "retrieval_configuration": protocol.get("retrieval", {}), "generation_configuration": protocol.get("generation", {}),
        "corpus_sha256": signature.get("corpus_sha256"), "dataset_sha256": signature.get("dataset_sha256"),
        "signature_sha256": signature.get("signature_sha256"), "quantization": manifest.get("quantization"),
        "packages": manifest.get("packages"), "hardware": manifest.get("hardware"),
        "code_files": signature.get("code_files", {}),
        "answers_sha256": hashlib.sha256(answer_bytes).hexdigest(),
        "answers_snapshot_bytes": len(answer_bytes),
        "answers_snapshot_boundary": "One byte snapshot is parsed and hashed. A source run may append further answers later; rerun the comparison to refresh incomplete conditions.",
        "selected_questions_sha256": digest_file(path / "selected_questions.jsonl"),
        "manifest_sha256": digest_file(path / "run_manifest.json"),
    }
    return {"path": path, "manifest": manifest, "metadata": metadata, "questions": questions, "answers": indexed}


def mean(values: list[Any]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return float(np.mean(present)) if present else None


def question_identity(question: dict) -> dict:
    # A common ID alone is insufficient if a dataset's questions or references changed.
    return {key: question.get(key) for key in ("question", "reference_answer_origin", "source_group", "expected_source_id")}


def summarize(run: dict, common_ids: list[str]) -> tuple[list[dict], dict]:
    results, scored = [], {}
    for condition in run["metadata"]["actual_conditions"]:
        available = [qid for qid in common_ids if (condition, qid) in run["answers"]]
        records = [score_record(run["answers"][(condition, qid)], run["questions"][qid], "openqa") for qid in available]
        for record in records:
            scored[(condition, record["id"])] = record
        marker_total = sum(record["citation_markers"] for record in records)
        resolved_total = sum(record["citation_markers_resolved"] for record in records)
        is_rag = condition.endswith("_rag")
        strict_cited = [bool(record["citation_markers"] and not record["citation_markers_unresolved"]) for record in records]
        raw = [run["answers"][(condition, qid)] for qid in available]
        row = {"run": run["metadata"]["run"], "condition": condition,
            "adapter_used": condition.startswith("lora"),
            "common_selected_questions": len(common_ids), "answers_available": len(records),
            "missing_common_answers": len(common_ids) - len(records),
            "source_groups_available": len({record["source_group"] for record in records}),
            "citation_stripped_token_f1_text_overlap": mean([record["citation_stripped_token_f1"] for record in records]),
            "original_token_f1_text_overlap": mean([record["token_f1"] for record in records]),
            "citation_appearance_answer_fraction": mean([record["citation_markers"] > 0 for record in records]),
            "all_citation_ids_resolvable_answer_fraction_requires_marker": mean(strict_cited),
            "all_citation_ids_resolvable_answers": sum(strict_cited),
            "citation_markers": marker_total, "resolvable_citation_markers": resolved_total,
            "marker_level_resolvable_fraction": resolved_total / marker_total if marker_total else None,
            "no_rag_unsupported_marker_answer_fraction": mean([record["citation_markers"] > 0 for record in records]) if not is_rag else None,
            "gold_source_recall_shown": mean([record["gold_source_recall_at_k"] for record in records]),
            "gold_source_recall_denominator": sum(record["gold_source_recall_at_k"] is not None for record in records),
            "hit_token_limit_fraction": mean([(record.get("metrics") or {}).get("hit_token_limit") for record in raw]),
            "truncation_measurement_denominator": sum((record.get("metrics") or {}).get("hit_token_limit") is not None for record in raw),
            "contains_cjk_answer_fraction": mean([record["contains_cjk"] for record in records]),
            "obvious_cjk_output_fraction": mean([record["obvious_cjk_output_flag"] for record in records]),
            "output_tokens_including_eos_mean": mean([(record.get("metrics") or {}).get("output_tokens_including_eos") for record in raw]),
            "expert_factual_status": "pending", "expert_completeness_status": "pending", "expert_evidence_support_status": "pending"}
        results.append(row)
    return results, scored


def comparable_pair(left: dict, right: dict, condition: str, ids: list[str]) -> dict:
    a, b = left["metadata"], right["metadata"]
    reasons = []
    for field in ("prompt_profile", "retrieval_method", "retrieval_configuration", "corpus_sha256", "generation_configuration", "quantization", "base_artifact_inventory_sha256"):
        if a.get(field) != b.get(field):
            reasons.append(field + " differs")
    if not a.get("base_artifact_inventory_sha256") or not b.get("base_artifact_inventory_sha256"):
        reasons.append("frozen base artifact inventory unavailable")
    if not a.get("corpus_sha256") or not b.get("corpus_sha256"):
        reasons.append("frozen corpus fingerprint unavailable")
    prompt_mismatch, context_mismatch, missing_hash = [], [], []
    for qid in ids:
        left_row, right_row = left["answers"][(condition, qid)], right["answers"][(condition, qid)]
        for field, mismatches in (("prompt_sha256", prompt_mismatch), ("context_sha256", context_mismatch)):
            if not left_row.get(field) or not right_row.get(field):
                missing_hash.append({"id": qid, "field": field})
            elif left_row[field] != right_row[field]:
                mismatches.append(qid)
        actual_a = [(hit.get("id"), hit.get("text")) for hit in left_row.get("retrieved_context", [])]
        actual_b = [(hit.get("id"), hit.get("text")) for hit in right_row.get("retrieved_context", [])]
        if actual_a != actual_b:
            context_mismatch.append(qid)
    if prompt_mismatch:
        reasons.append("rendered prompt hashes differ")
    if context_mismatch:
        reasons.append("actual supplied context or hashes differ")
    if missing_hash:
        reasons.append("some rendered prompt/context hashes unavailable")
    if not ids:
        reasons.append("no paired answers")
    adapter_changed = a["adapter_artifact_inventory_sha256"] != b["adapter_artifact_inventory_sha256"]
    return {"eligible_for_paired_overlap_delta": not reasons, "reasons_if_skipped": reasons,
        "same_condition": condition, "same_profile": a["prompt_profile"] == b["prompt_profile"],
        "same_retrieval_method": a["retrieval_method"] == b["retrieval_method"],
        "paired_answers": len(ids), "prompt_mismatch_ids": sorted(set(prompt_mismatch)),
        "context_mismatch_ids": sorted(set(context_mismatch)), "missing_hash_checks": missing_hash,
        "adapter_inventory_changed": adapter_changed,
        "interpretation": "Matched prompts, contexts, base artifacts and decoding; adapter comparison candidate, still exploratory and not a factuality score."
            if not reasons and condition.startswith("lora") and adapter_changed
            else "Descriptive repeated configuration or multi-setting comparison; no isolated fine-tuning claim."}


def format_pct(value: float | None) -> str:
    return "—" if value is None else f"{100 * value:.1f}%"


def compare(paths: list[Path], out: Path) -> None:
    runs = [load_run(path) for path in paths]
    names = [run["metadata"]["run"] for run in runs]
    if len(set(names)) != len(names):
        raise ValueError("Run directory names must be unique")
    out = out.resolve()
    if out in {run["path"] for run in runs}:
        raise ValueError("Comparison output must be separate from every source run")
    intersection = set.intersection(*(set(run["questions"]) for run in runs))
    common_ids = [qid for qid in runs[0]["questions"] if qid in intersection]
    if not common_ids:
        raise ValueError("No selected question IDs are shared by all supplied runs")
    for qid in common_ids:
        identity = question_identity(runs[0]["questions"][qid])
        if any(question_identity(run["questions"][qid]) != identity for run in runs[1:]):
            raise ValueError(f"Same question ID has changed source/question/original answer: {qid}")
    source_groups = {runs[0]["questions"][qid]["source_group"] for qid in common_ids}
    table, scores = [], {}
    for run in runs:
        rows, indexed = summarize(run, common_ids)
        table.extend(rows)
        scores[run["metadata"]["run"]] = indexed
        run["metadata"]["shared_selected_question_count"] = len(common_ids)
        run["metadata"]["shared_source_groups"] = len(source_groups)
    paired = []
    for left, right in itertools.combinations(runs, 2):
        left_name, right_name = left["metadata"]["run"], right["metadata"]["run"]
        shared_conditions = [condition for condition in CONDITIONS if condition in left["metadata"]["actual_conditions"] and condition in right["metadata"]["actual_conditions"]]
        for condition in shared_conditions:
            ids = [qid for qid in common_ids if (condition, qid) in left["answers"] and (condition, qid) in right["answers"]]
            check = comparable_pair(left, right, condition, ids)
            entry = {"left_run": left_name, "right_run": right_name, "difference_direction": "right minus left",
                "condition": condition, "comparability": check, "source_groups": len({left["questions"][qid]["source_group"] for qid in ids})}
            if check["eligible_for_paired_overlap_delta"]:
                entry["paired_source_cluster_bootstrap"] = {metric: clustered_interval([
                    (left["questions"][qid]["source_group"], scores[right_name][(condition, qid)][metric] - scores[left_name][(condition, qid)][metric])
                    for qid in ids]) for metric in ("citation_stripped_token_f1", "token_f1")}
            else:
                entry["paired_source_cluster_bootstrap"] = None
            paired.append(entry)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "summary.csv", table)
    write_jsonl(out / "shared_questions.jsonl", [runs[0]["questions"][qid] for qid in common_ids])
    write_jsonl(out / "per_answer_automatic_metrics.jsonl", [{"run": name, **record} for name, indexed in scores.items() for record in indexed.values()])
    warnings = ["English DEVELOPMENT comparison only; the shared subset can be small, follows supplied run selections, is not random sampling and is not an official held-out test result.",
        "Citation-stripped and original token F1 are lexical overlap, not factual correctness or semantic equivalence.",
        "At least one citation marker is required for the all-IDs-resolvable answer fraction; zero citations never count as successful citation.",
        "Resolvable IDs establish syntax/link availability, not evidence support. The no-RAG marker rate indicates unsupplied citation IDs, not all unsupported factual claims.",
        "The CJK script diagnostic is not validated English language identification.",
        "Different prompt profiles or retrieval settings are multi-setting comparisons and must not be attributed solely to fine-tuning.",
        "Only matched same-condition prompts, actual contexts, retrieval, frozen base artifacts and generation settings receive a paired overlap difference.",
        "Tables use actual available answers within the shared selected question IDs; missing conditions are omitted and missing answers are not scored as zero.",
        "No automatic best model is declared, and all expert factual/completeness/evidence ratings remain pending."]
    analysis = {"development_only": True, "shared_question_ids": common_ids, "shared_question_count": len(common_ids),
        "shared_source_group_count": len(source_groups), "source_groups": sorted(source_groups),
        "metadata": [run["metadata"] for run in runs], "summary": table, "paired_overlap_comparisons": paired,
        "confidence_interval_method": "2000 source_group-clustered paired bootstrap resamples; question-weighted means; seed 20261016; no CI with fewer than two sources.",
        "expert_review": {"status": "pending", "factual": None, "completeness": None, "evidence_support": None},
        "warnings": warnings, "comparison_code_sha256": digest_file(Path(__file__))}
    write_json(out / "analysis.json", analysis)
    lines = ["# 英文开发运行的共同题目比较", "", f"所有输入运行共同选择 **{len(common_ids)}题、{len(source_groups)}个source_group**。各行只统计该运行实际已有的条件和回答；缺失答案不按0分处理。", "",
        "这是开发阶段探索性比较。引用去除后的Token F1与原始Token F1均为文本重合度，不能写成事实准确率；没有导入专家评分，也未自动选出最佳模型。", "",
        "## 运行元数据", "", "| 运行 | 原选题数 | 实际条件 | Prompt profile | 检索 | adapter目录（LoRA条件使用） | adapter权重SHA256 |", "|---|---:|---|---|---|---|---|"]
    for run in runs:
        meta = run["metadata"]
        hashes = "; ".join(f"{key}: {value}" for key, value in meta["adapter_weight_files_sha256"].items()) or "冻结记录中缺失"
        lines.append(f"| {meta['run']} | {meta['selected_questions']} | {', '.join(meta['actual_conditions'])} | {meta['prompt_profile']} | {meta['retrieval_method']} | {meta['adapter_path_from_frozen_protocol']} | {hashes} |")
    lines += ["", "完整模型清单指纹、语料哈希、生成设置、提示词与实现哈希见analysis.json。不同提示词或检索的差异是多项设置共同变化，不能归因于单独微调。", "",
        "## 共同题目上的自动诊断", "", "| 运行 | 条件 | 已有／共同题数 | 来源组 | 去引用F1（重合度） | 原F1（重合度） | 有引用答案率 | 全编号可解析答案率＊ | 编号级可解析率 | 无RAG无来源编号率 | Gold来源召回 | 截断率 | 明显中文率 |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in table:
        values = [format_pct(row[field]) for field in ("citation_stripped_token_f1_text_overlap", "original_token_f1_text_overlap",
            "citation_appearance_answer_fraction", "all_citation_ids_resolvable_answer_fraction_requires_marker", "marker_level_resolvable_fraction",
            "no_rag_unsupported_marker_answer_fraction", "gold_source_recall_shown", "hit_token_limit_fraction", "obvious_cjk_output_fraction")]
        lines.append(f"| {row['run']} | {CONDITION_NAMES[row['condition']]} | {row['answers_available']}/{row['common_selected_questions']} | {row['source_groups_available']} | " + " | ".join(values) + " |")
    lines += ["", "＊全编号可解析答案率：至少出现1个引用标记，且该答案每个标记都能对应实际提供的资料ID，分母为本行实际回答数；无引用不能算成功。编号级比例以标记出现次数为分母。编号正确不代表原文支持陈述。Gold来源召回仅适用于RAG；无RAG出现编号意味着该编号没有提供来源。截断依据hit_token_limit记录，中文率是字符脚本诊断，不是英文质量评分。", "",
        "## 同条件的配对文本重合度差", "", "只对相同prompt_profile、检索配置、语料、生成参数、基座权重，以及逐题实际提示词／证据哈希一致的配对给出区间。方向均为后列运行减前列运行。", "",
        "| 前列运行 → 后列运行 | 条件 | 配对题数／来源组 | 去引用F1差（百分点，95% CI） | 状态 |", "|---|---|---:|---:|---|"]
    for entry in paired:
        check = entry["comparability"]
        if entry["paired_source_cluster_bootstrap"] is not None:
            data = entry["paired_source_cluster_bootstrap"]["citation_stripped_token_f1"]
            ci = data["ci95"]
            delta = "—" if data["mean"] is None else f"{100 * data['mean']:+.1f}"
            if ci:
                delta += f" [{100 * ci[0]:+.1f}, {100 * ci[1]:+.1f}]"
            else:
                delta += " [CI unavailable]"
            status = "匹配设置的探索性重合度差；待专家验证"
        else:
            delta = "不作配对效应解释"
            status = "; ".join(check["reasons_if_skipped"])
        lines.append(f"| {entry['left_run']} → {entry['right_run']} | {CONDITION_NAMES[entry['condition']]} | {check['paired_answers']}/{entry['source_groups']} | {delta} | {status} |")
    lines += ["", "同一adapter在不同提示词或检索下的差异也不能描述为训练收益。即使配对条件匹配，短开发子集上的区间仍只描述文本重合度；不得代替独立测试与双人事实评价。", "",
        "## 文件", "", "- summary.csv：所有实际条件的共同题目表及明确分母。", "- shared_questions.jsonl：本次共同题目与source_group。", "- per_answer_automatic_metrics.jsonl：逐条原始回答及自动诊断。", "- analysis.json：冻结元数据、全部哈希、可比性检查和配对区间。", ""]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Compared {len(runs)} actual development runs on {len(common_ids)} common questions/{len(source_groups)} sources: {out / 'report.md'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    compare([resolve(str(path)) for path in args.runs], resolve(str(args.out)))


if __name__ == "__main__":
    main()

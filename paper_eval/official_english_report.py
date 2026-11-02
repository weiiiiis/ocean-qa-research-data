"""Report one official English run without generating or modifying any answer.

Run only after the protocol is frozen and official answers have been generated:
    python -m paper_eval.official_english_report --run RUN_DIR --out NEW_REPORT_DIR

MC prefix accuracy evaluates option selection; strict accuracy separately
evaluates the exact requested format. Both frozen rules are reported.
Open-QA EM/F1 are lexical-overlap metrics, never expert factual correctness.
"""
from __future__ import annotations

import argparse
import collections
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .common import CONDITION_NAMES, digest_file, read_jsonl, resolve, write_json
from .english_report import (
    COMPARISONS,
    MC_PREFIX_PATTERN,
    fair_comparison,
    performance,
    requested_conditions,
    score_record,
)

RESAMPLES = 2000
SEED = 20261017
FIXED_HISTORICAL_REVIEW_IDS = frozenset({"A1329", "B2585"})
GROUPINGS = ("source_group", "source_group_conservative")
SLICE_NAMES = {
    "all_227": "主集：全部227题",
    "historical_review_excluded_225": "敏感性：排除两个历史复核标记题（225题）",
    "published_answer_in_text_1_206": "敏感性：官方answer_in_text=1（206题）",
    "published_answer_in_text_0_or_missing_21": "诊断：answer_in_text=0或缺失（21题）",
}


def interval(items: list[tuple[str, float, float]]) -> dict:
    """Cluster bootstrap of a question-weighted mean or pooled count ratio.

    Each tuple contains (source group, numerator, denominator). For a mean,
    each question has denominator one. Marker resolution uses marker counts,
    not an average of per-answer ratios. Both groupings use identical methods.
    """
    totals: dict[str, list[float]] = {}
    for group, numerator, denominator in items:
        if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator < 0:
            raise ValueError("Nonfinite or invalid metric count")
        target = totals.setdefault(group, [0.0, 0.0])
        target[0] += float(numerator)
        target[1] += float(denominator)
    if not totals:
        return {"mean": None, "ci95": None, "questions": 0, "source_groups": 0,
                "informative_source_groups": 0, "numerator_total": 0.0, "denominator_total": 0.0,
                "bootstrap_valid_replicates": 0, "bootstrap_zero_denominator_replicates": 0}
    values = np.asarray(list(totals.values()), dtype=np.float64)
    numerator, denominator = values[:, 0], values[:, 1]
    denominator_total = float(denominator.sum())
    informative = int(np.count_nonzero(denominator))
    mean = float(numerator.sum() / denominator_total) if denominator_total else None
    ci = None
    valid_count = zero_count = 0
    if denominator_total and len(totals) > 1 and informative > 1:
        rng = np.random.default_rng(SEED)
        draws = rng.integers(0, len(totals), size=(RESAMPLES, len(totals)))
        boot_denominator = denominator[draws].sum(axis=1)
        usable = boot_denominator > 0
        valid_count = int(usable.sum())
        zero_count = RESAMPLES - valid_count
        if valid_count:
            replicates = numerator[draws].sum(axis=1)[usable] / boot_denominator[usable]
            ci = np.percentile(replicates, [2.5, 97.5]).tolist()
    return {"mean": mean, "ci95": ci, "questions": len(items), "source_groups": len(totals),
            "informative_source_groups": informative, "numerator_total": float(numerator.sum()),
            "denominator_total": denominator_total, "bootstrap_valid_replicates": valid_count,
            "bootstrap_zero_denominator_replicates": zero_count}


def withheld_metric() -> dict:
    result = interval([])
    result["status"] = "withheld_until_condition_has_all_227_answers"
    return result


def question_metric(rows: list[dict], key: str, grouping: str) -> dict:
    eligible = [row for row in rows if row.get(key) is not None]
    result = interval([(row[grouping], float(row[key]), 1.0) for row in eligible])
    result["missing_or_not_applicable_records"] = len(rows) - len(eligible)
    return result


def slice_masks(questions: dict[str, dict]) -> dict[str, set[str]]:
    all_ids = set(questions)
    overlap_ids = {qid for qid, question in questions.items() if question["original_id"] in FIXED_HISTORICAL_REVIEW_IDS}
    support_ids = {qid for qid, question in questions.items()
                   if question["original_assessments"].get("answer_in_text") == 1}
    masks = {"all_227": all_ids, "historical_review_excluded_225": all_ids - overlap_ids,
             "published_answer_in_text_1_206": support_ids,
             "published_answer_in_text_0_or_missing_21": all_ids - support_ids}
    expected = dict(zip(SLICE_NAMES, (227, 225, 206, 21)))
    if {key: len(value) for key, value in masks.items()} != expected:
        raise ValueError("Official slice sizes changed; do not silently select a different denominator")
    if {questions[qid]["original_id"] for qid in overlap_ids} != FIXED_HISTORICAL_REVIEW_IDS:
        raise ValueError("Historical review mask must match original_id A1329/B2585, regardless of ID prefix")
    return masks


def metric_names(task: str) -> list[str]:
    answer_metrics = (["accuracy", "mc_prefix_sensitivity_accuracy", "mc_format_invalid"] if task == "mcqa"
                      else ["citation_stripped_exact_match", "citation_stripped_token_f1", "exact_match", "token_f1"])
    return answer_metrics + ["answer_has_citation", "answer_has_at_least_one_and_all_markers_resolved",
                             "gold_source_recall_at_k", "hit_token_limit"]


def summarize(rows: list[dict], grouping: str, metrics: list[str], complete: bool, target_size: int) -> dict:
    result = {"actual_answers": len(rows), "target_questions": target_size,
              "condition_full_227_complete": complete,
              "status": "published_full_fixed_denominator" if complete and len(rows) == target_size else "withheld_incomplete_condition",
              "metrics": {}, "counts": {
                  "citation_markers": sum(row["citation_markers"] for row in rows),
                  "resolved_citation_markers": sum(row["citation_markers_resolved"] for row in rows)}}
    if not complete or len(rows) != target_size:
        result["metrics"] = {key: withheld_metric() for key in metrics + ["marker_resolution_ratio"]}
        return result
    result["metrics"] = {key: question_metric(rows, key, grouping) for key in metrics}
    result["metrics"]["marker_resolution_ratio"] = interval([
        (row[grouping], float(row["citation_markers_resolved"]), float(row["citation_markers"])) for row in rows])
    return result


def paired_differences(scored: list[dict], mask: set[str], conditions: tuple[str, ...], grouping: str,
                       metrics: list[str], condition_complete: dict[str, bool]) -> dict:
    index = {(row["condition"], row["id"]): row for row in scored if row["id"] in mask}
    result = {}
    for left, right in COMPARISONS:
        if left not in conditions or right not in conditions:
            continue
        left_ids = {qid for condition, qid in index if condition == left}
        right_ids = {qid for condition, qid in index if condition == right}
        paired = sorted(left_ids & right_ids)
        comparisons = {}
        for metric in metrics:
            usable = [qid for qid in paired if index[(left, qid)].get(metric) is not None
                      and index[(right, qid)].get(metric) is not None]
            comparisons[metric] = interval([
                (index[(left, qid)][grouping], float(index[(left, qid)][metric]) - float(index[(right, qid)][metric]), 1.0)
                for qid in usable])
            comparisons[metric]["paired_question_denominator"] = len(usable)
            comparisons[metric]["not_applicable_or_missing_pairs"] = len(paired) - len(usable)
        result[f"{left}_minus_{right}"] = {"shared_completed_questions": len(paired),
            "target_slice_questions": len(mask), "both_conditions_full_227_complete": condition_complete[left] and condition_complete[right],
            "denominator_policy": "Only questions generated in both conditions; metric-NA pairs excluded and counted separately",
            "metrics": comparisons}
    return result


def fmt(value: Any, percent: bool = True) -> str:
    if value is None:
        return "—"
    return f"{100 * value:.1f}%" if percent else f"{value:.2f}"


def fmt_interval(value: dict) -> str:
    if value.get("status", "").startswith("withheld"):
        return "未完整，暂不发布"
    text = fmt(value.get("mean"))
    ci = value.get("ci95")
    return text + (f" [{100 * ci[0]:.1f}, {100 * ci[1]:.1f}]" if ci is not None else " [CI不可用]")


def report(run: Path, out: Path) -> None:
    run, out = resolve(str(run)).resolve(), resolve(str(out)).resolve()
    if out.exists():
        raise ValueError("--out must be a new directory; existing reports/reviewer work will not be overwritten")
    if out == run or run in out.parents:
        raise ValueError("Use a separate report directory outside the frozen inference run")
    manifest_path, answers_path, selected_path = (run / name for name in
                                                 ("run_manifest.json", "answers.jsonl", "selected_questions.jsonl"))
    if not manifest_path.exists():
        raise FileNotFoundError("Official reporting requires the frozen run_manifest.json")
    source_hashes = {path.name: digest_file(path) for path in (manifest_path, answers_path, selected_path)}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    conditions = requested_conditions(run)
    selected, rows = read_jsonl(selected_path), read_jsonl(answers_path)
    questions = {question["id"]: question for question in selected}
    if len(selected) != 227 or len(questions) != 227 or len({q["original_id"] for q in selected}) != 227:
        raise ValueError("Official reporting requires all 227 unique selected original IDs")
    for question in selected:
        if question.get("split") != "test" or question.get("original_split") != "test":
            raise ValueError("This report cannot re-label a development set as official test")
        if not question.get("source_group") or not question.get("source_group_conservative"):
            raise ValueError("Prepared official source_group and conservative grouping are both required")
        if not isinstance(question.get("reference_answer_origin"), str) or not question["reference_answer_origin"].strip():
            raise ValueError("Missing original official reference answer")
    tasks = {"mcqa" if question.get("choices") else "openqa" for question in selected}
    if len(tasks) != 1:
        raise ValueError("Report open-QA and MCQA in separate directories; do not add their sample sizes")
    task = tasks.pop()
    expected_ids = set(questions)
    if len({(row["condition"], row["id"]) for row in rows}) != len(rows):
        raise ValueError("Duplicate condition/question output")
    for row in rows:
        if row["condition"] not in conditions or row["id"] not in questions:
            raise ValueError("Answer condition/question lies outside the frozen requested selection")
        if row.get("task") != task or row.get("phase", "quality") != "quality":
            raise ValueError("Only natural-answer records for this one task may be reported")
        if row.get("question") != questions[row["id"]]["question"]:
            raise ValueError("Saved answer stem differs from the frozen selected question")
    masks = slice_masks(questions)
    label_counts = collections.Counter("missing" if q["original_assessments"].get("answer_in_text") is None
                                      else str(int(q["original_assessments"]["answer_in_text"])) for q in selected)
    if dict(label_counts) != {"1": 206, "0": 10, "missing": 11}:
        raise ValueError("Official answer_in_text labels must remain 206/10/11")
    coverage = {}
    for condition in conditions:
        observed = {row["id"] for row in rows if row["condition"] == condition}
        coverage[condition] = {"completed_answers": len(observed), "expected_answers": 227,
                               "complete": observed == expected_ids, "missing_ids": sorted(expected_ids - observed)}
    complete_flags = {condition: coverage[condition]["complete"] for condition in conditions}
    scored = []
    for raw in rows:
        question = questions[raw["id"]]
        item = score_record(raw, question, task)
        # Alternate-reference maxima are not part of this official report.
        for key in list(item):
            if key.startswith("published_reference_sensitivity_"):
                item.pop(key)
        item.update(original_id=question["original_id"], corpus=question["corpus"],
                    source_group_conservative=question["source_group_conservative"],
                    source_group_original=question.get("source_group_original"),
                    original_assessments=question["original_assessments"],
                    historical_training_overlap_review_flag=question["original_id"] in FIXED_HISTORICAL_REVIEW_IDS,
                    answer_has_citation=bool(item["citation_markers"]),
                    answer_has_at_least_one_and_all_markers_resolved=bool(item["citation_markers"] > 0
                          and item["citation_markers_resolved"] == item["citation_markers"]),
                    hit_token_limit=(raw.get("metrics") or {}).get("hit_token_limit"))
        if task == "mcqa":
            item["mc_format_invalid"] = not item["mc_valid_format"]
        scored.append(item)
    metrics = metric_names(task)
    slices = {}
    for name, mask in masks.items():
        subset = [row for row in scored if row["id"] in mask]
        slices[name] = {"description": SLICE_NAMES[name], "target_questions": len(mask),
            "selected_original_ids": sorted(questions[qid]["original_id"] for qid in mask), "groupings": {}}
        for grouping in GROUPINGS:
            slices[name]["groupings"][grouping] = {
                "selected_source_groups": len({questions[qid][grouping] for qid in mask}),
                "by_condition": {condition: summarize([row for row in subset if row["condition"] == condition],
                                  grouping, metrics, complete_flags[condition], len(mask)) for condition in conditions},
                "paired_differences": paired_differences(subset, mask, conditions, grouping, metrics, complete_flags)}
    fairness = fair_comparison(rows)
    analysis = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "report_type": "official_pira_test", "task": task,
        "run_directory": str(run), "requested_conditions": list(conditions), "selected_questions": 227,
        "expected_answers": len(conditions) * 227, "completed_answers": len(rows),
        "complete": all(complete_flags.values()), "condition_coverage": coverage,
        "source_artifacts_sha256": source_hashes,
        "reporting_code_sha256": {path.name: digest_file(path) for path in (Path(__file__), Path(__file__).with_name("english_report.py"))},
        "frozen_run_signature_sha256": manifest.get("signature", {}).get("signature_sha256"),
        "source_group_counts": {grouping: len({q[grouping] for q in selected}) for grouping in GROUPINGS},
        "fixed_slice_definitions": {
            "all_227": "All official questions, no model-based filtering",
            "historical_review_excluded_225": "original_id not in {A1329,B2585}; prior flags are review candidates, not confirmed leakage",
            "published_answer_in_text_1_206": "Released original_assessments.answer_in_text == 1",
            "published_answer_in_text_0_or_missing_21": "Released answer_in_text == 0 (10) or missing (11); retain original golds, not relabeled unanswerable",
        },
        "original_annotation_counts": {"answer_in_text": dict(label_counts),
            "at_labels": dict(collections.Counter("missing" if q["original_assessments"].get("at_labels") is None
                         else str(int(q["original_assessments"]["at_labels"])) for q in selected))},
        "mc_option_parsing_rule": {"pattern": MC_PREFIX_PATTERN, "strict_score_rule_unchanged": True,
            "option_selection_metric_key": "mc_prefix_sensitivity_accuracy", "strict_format_metric_key": "accuracy",
            "fixed_before_official_outputs": True, "unparseable_generated_answer": "incorrect"} if task == "mcqa" else None,
        "openqa_metric_policy": "Original reference only. Citation-stripped EM/F1 first, raw EM/F1 also reported. All are lexical overlap, not factual accuracy.",
        "citation_metric_policy": "Appearance and at-least-one/all-resolved are fractions of all answers. Marker ratio pools resolved markers / all markers, not mean per-answer ratios. No citations -> all-resolved answer indicator false, marker ratio undefined if its denominator is zero. Resolved IDs do not establish evidence support.",
        "bootstrap": {"resamples": RESAMPLES, "seed": SEED, "method": "Percentile source-cluster bootstrap with replacement; question-weighted means and pooled marker-count ratios",
            "groupings": list(GROUPINGS), "undefined_ci": "Fewer than two informative clusters or zero denominator; zero-denominator bootstrap replicates excluded and counted",
            "paired_policy": "Resample the same source clusters for within-question differences; include only common completed IDs, with metric-specific NA pairs omitted and counted"},
        "grouping_limitation": "Main groups merge same known EID or exact normalized passage; missing source/report identity remains unresolved. Conservative sensitivity puts all corpus2 questions in one unresolved group; it is a dependence assumption, not proof of one actual report or complete-document independence.",
        "fairness": fairness, "slices": slices, "natural_answer_cost_descriptive": performance(rows, conditions),
        "expert_review": {"status": "pending", "factual_correctness": None, "completeness": None, "citation_evidence_support": None},
        "warnings": ["The two task variants share 227 original IDs. Never sum open-QA and MCQA as 454 independent questions.",
            "No selected reference answer, annotation or dataset split is modified by this report.",
            "No fixed-denominator per-condition mean is published unless that condition has every one of the 227 actual answers. Missing outputs are not zero scores.",
            "Paired differences can use fewer questions than a full slice; their explicit shared denominator must accompany the result.",
            "Original at_labels is not used as an abstention/refusal label. The 21 unsupported-or-unannotated records are retained with published answers.",
            "Source-ID resolution and source recall are automatic diagnostics, not factual correctness, citation entailment or expert judgment.",
            "Source clusters are not guaranteed independent full reports, particularly corpus2. Conservative grouping is a sensitivity, not a guarantee.",
            "Natural-answer latency depends on input/output length, retrieval and model order; no equal-work speed claim follows from this report.",
            "No expert ratings have been entered, imported or inferred."],
        "per_answer_automatic_scores": scored,
    }
    lines = ["# Pirá官方英文测试报告", "",
        f"任务：{task}；官方问题227道；请求条件：{'、'.join(CONDITION_NAMES[c] for c in conditions)}；实际回答{len(rows)}/{len(conditions) * 227}条。", "",
        f"请求范围完整：{'是' if analysis['complete'] else '否'}。任何未完成227题的条件，固定分母均值暂不发布。缺失回答不计0。", "",
        "开放问答与选择题分别报告，两者共享227个原始问题ID，不能合计为454道独立题。人工事实、完整性及引用支持评分尚未完成。", ""]
    if task == "openqa":
        lines += ["开放题仅使用reference_answer_origin；去引用EM/F1与原文EM/F1都是文本重合度，不能称为事实准确率。", ""]
    else:
        lines += ["选择题同时报告统一开头字母解析的选项准确率和严格单字母准确率；前者用于选项判断，后者同时反映格式遵循。解析规则在正式输出前固定、四组一致。原始严格分数单列保留。", "",
                  "```regex", MC_PREFIX_PATTERN, "```", ""]
    for name, mask in masks.items():
        lines += [f"## {SLICE_NAMES[name]}", ""]
        for grouping in GROUPINGS:
            data = slices[name]["groupings"][grouping]
            label = "主来源分组" if grouping == "source_group" else "保守分组敏感性（corpus2合为一个未解析组）"
            lines += [f"### {label}：{data['selected_source_groups']}组", "",
                      "下列值为百分比，方括号为来源组bootstrap的95%区间。", ""]
            if task == "mcqa":
                lines += ["| 条件 | 实际 / 目标题数 | 严格单字母准确率 | 开头字母选项准确率 | 无效格式率 |",
                          "|---|---:|---:|---:|---:|"]
                answer_keys = ("accuracy", "mc_prefix_sensitivity_accuracy", "mc_format_invalid")
            else:
                lines += ["| 条件 | 实际 / 目标题数 | 去引用EM | 去引用F1 | 原EM | 原F1 |",
                          "|---|---:|---:|---:|---:|---:|"]
                answer_keys = ("citation_stripped_exact_match", "citation_stripped_token_f1", "exact_match", "token_f1")
            for condition, item in data["by_condition"].items():
                lines.append(f"| {CONDITION_NAMES[condition]} | {item['actual_answers']}/{item['target_questions']} | " +
                             " | ".join(fmt_interval(item["metrics"][key]) for key in answer_keys) + " |")
            lines += ["", "| 条件 | 有引用回答比例 | 至少1个引用且全部编号可解析 / 全部回答 | 可解析marker / 全部marker | Gold来源召回 | 长度截断 |",
                      "|---|---:|---:|---:|---:|---:|"]
            for condition, item in data["by_condition"].items():
                keys = ("answer_has_citation", "answer_has_at_least_one_and_all_markers_resolved",
                        "marker_resolution_ratio", "gold_source_recall_at_k", "hit_token_limit")
                lines.append(f"| {CONDITION_NAMES[condition]} | " + " | ".join(fmt_interval(item["metrics"][key]) for key in keys) + " |")
            lines += ["", "引用编号可解析不代表原文支持答案。没有引用的回答不会算作“全部可解析”；全组无marker时marker比例未定义。无RAG条件的来源召回不适用。", ""]
            if task == "mcqa":
                paired_keys = ("accuracy", "mc_prefix_sensitivity_accuracy")
                lines += ["| 配对比较 | 共有已回答 / 目标题数 | 严格单字母差值 [95% CI] | 选项准确率差值 [95% CI] |",
                          "|---|---:|---:|---:|"]
            else:
                paired_keys = ("citation_stripped_token_f1", "token_f1")
                lines += ["| 配对比较 | 共有已回答 / 目标题数 | 去引用F1差值 [95% CI] | 原F1差值 [95% CI] |",
                          "|---|---:|---:|---:|"]
            for pair, item in data["paired_differences"].items():
                left, right = pair.split("_minus_")
                lines.append(f"| {CONDITION_NAMES[left]} − {CONDITION_NAMES[right]} | {item['shared_completed_questions']}/{item['target_slice_questions']} | " +
                             " | ".join(fmt_interval(item["metrics"][key]) for key in paired_keys) + " |")
            lines += ["", "配对差值单位为百分点，只使用两个条件共有的实际回答；完整分母与配对分母分开。其他自动指标的配对区间、NA数和marker计数详见analysis.json。", ""]
    lines += ["## 标签与统计边界", "",
        "- 主集227；历史复核敏感性固定按original_id排除A1329/B2585后225；两题仍保留在主集，不自动认定泄漏。",
        "- 原始answer_in_text=1共206；=0共10；缺失11。后21条保留原答案作诊断；at_labels另行保留，不解释为拒答标签。",
        f"- 主来源分组{analysis['source_group_counts']['source_group']}组；保守分组{analysis['source_group_counts']['source_group_conservative']}组。缺报告ID及缺EID的记录不能证明整文档独立。",
        f"- 来源组bootstrap：{RESAMPLES}次，seed={SEED}；问题均值按题加权，marker比例按marker总数。少于两个有效来源组或分母为0时CI不可用。",
        "- 当前自动指标和引用检查均不代替专家事实、完整性、证据支持评价。人工分全部pending。",
        "- 自然回答成本仅作描述，完整值在analysis.json；没有固定工作量复测，不据此声称推理效率提升。", "",
        "## 可追溯性", "",
        f"原始回答SHA256：`{source_hashes['answers.jsonl']}`。", "",
        f"问题清单SHA256：`{source_hashes['selected_questions.jsonl']}`。", "",
        "本报告只读取原始回答和冻结问题，不生成模型答案，不修改原始标签、参考答案或切分。", ""]
    # Refuse to publish if an active process appended data during analysis.
    for path in (manifest_path, answers_path, selected_path):
        if digest_file(path) != source_hashes[path.name]:
            raise ValueError("Inference input changed while reporting; wait until the run is stable")
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "analysis.json", analysis)
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Official {task} report saved to {out}; complete={analysis['complete']}; expert scores pending")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True, help="Frozen official inference run")
    parser.add_argument("--out", type=Path, required=True, help="New report directory outside the frozen run")
    args = parser.parse_args()
    report(args.run, args.out)


if __name__ == "__main__":
    main()

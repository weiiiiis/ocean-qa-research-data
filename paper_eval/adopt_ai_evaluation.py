"""Report the user's adopted AI evaluations separately from human ratings."""
from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/paper_eval/ai_prescore_teachercheck_v1_20261018"
REVIEW = ROOT / "outputs/paper_eval/en_test_blind40_frozen_v1_20261017"
OUT = ROOT / "outputs/paper_eval/ai_evaluation_adopted_20261018"
CONDITIONS = ("base", "base_rag", "lora", "lora_rag")
NAMES = dict(zip(CONDITIONS, ("基座", "基座＋RAG", "微调", "微调＋RAG")))
METRICS = ("factual_quality", "completeness", "faithfulness", "citation_support", "citation_coverage")
REPLICATES, SEED = 2000, 20261018


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_bytes((json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))


def terms(row, metric):
    if metric == "factual_quality":
        return row["ai_factual_correctness"], 1
    if metric == "completeness":
        return row["ai_essential_completeness"], 1
    evidence = row.get("ai_evidence")
    if evidence is None:
        return 0, 0
    if metric == "faithfulness":
        return evidence["faithfulness"], 1
    if metric == "citation_support":
        return evidence["supported_citation_links_count"], evidence["citation_links_count"]
    if metric == "citation_coverage":
        return evidence["claims_with_supported_citation"], evidence["claims_requiring_citation"]
    raise KeyError(metric)


def estimate(rows, metric, grouping, groups, draws):
    totals = np.zeros((len(groups), 2), dtype=float)
    locations = {group: i for i, group in enumerate(groups)}
    for row in rows:
        totals[locations[row[grouping]]] += terms(row, metric)
    numerator, denominator = totals.sum(axis=0)
    sampled = totals[draws].sum(axis=1)
    valid = sampled[:, 1] > 0
    values = np.full(REPLICATES, np.nan)
    values[valid] = sampled[valid, 0] / sampled[valid, 1]
    interval = np.quantile(values[valid], [.025, .975]).tolist() if valid.sum() and len(groups) > 1 else None
    return {"mean": float(numerator / denominator) if denominator else None,
            "ci95": interval, "numerator": float(numerator), "denominator": float(denominator),
            "valid_bootstrap_replicates": int(valid.sum())}, values


def show(metric, ratio=False):
    if metric["mean"] is None:
        return "不适用"
    factor, suffix = (100, "%") if ratio else (1, "")
    value = f"{metric['mean'] * factor:.2f}{suffix}"
    ci = metric["ci95"]
    return value + (f" [{ci[0] * factor:.2f}, {ci[1] * factor:.2f}]" if ci else "")


def main():
    if OUT.exists():
        raise FileExistsError(f"Preserve the adopted report; output already exists: {OUT}")
    prior = read_json(SOURCE / "ai_prescore_manifest.json")
    for rel, expected in prior["source_hashes"].items():
        if digest(ROOT / rel) != expected:
            raise ValueError(f"AI score file changed after its recorded calibration: {rel}")
    mapping_data = read_json(REVIEW / "private/english_blind_mapping.json")
    if digest(REVIEW / "answers.jsonl") != mapping_data["answers_sha256"]:
        raise ValueError("Anonymous mapping no longer identifies the preserved raw review answers")
    mapping = mapping_data["packets"]["reviewer_1"]
    questions = {q["id"]: q for q in read_jsonl(REVIEW / "selected_questions.jsonl")}
    rows = []
    for shard in (1, 2, 3):
        for grade in read_jsonl(SOURCE / "graded" / f"shard_{shard}.jsonl"):
            link = mapping[grade["item_id"]]
            q = questions[link["id"]]
            rows.append({**grade, **link, "source_group": q["source_group"],
                         "source_group_conservative": q["source_group_conservative"], "corpus": q["corpus"],
                         "rating_origin": "AI", "independent_human_rating": False})
    if len(rows) != 160 or len({r["item_id"] for r in rows}) != 160 or len(questions) != 40:
        raise ValueError("The adopted AI evaluation must contain exactly the fixed 40 x 4 answers")
    selected = set(questions)
    if any({r["id"] for r in rows if r["condition"] == c} != selected for c in CONDITIONS):
        raise ValueError("Each condition must cover the same 40 questions")
    if sum(r["ai_evidence"] is not None for r in rows) != 80:
        raise ValueError("Expected 40 actual-evidence ratings per RAG condition")
    result = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "adoption_date_local": "2026-10-18",
        "adoption_basis": "User reports that their teacher agreed to directly use the AI suggestions.",
        "status": "adopted_as_AI_evaluation", "rating_origin": "AI", "ai_evaluation_workflows": 1,
        "parallel_scoring_shards": 3, "independent_human_raters": 0, "teacher_itemwise_scores_entered": False,
        "questions": 40, "quality_answers": 160, "evidence_answers": 80, "separate_reference_reviews": 21,
        "scope": "The preselected 40-question subset of 206 published answer_in_text=1 questions; not all 227 test questions.",
        "bootstrap": {"replicates": REPLICATES, "seed": SEED,
                      "method": "Paired source-cluster bootstrap; question-weighted score means and pooled count ratios. Same cluster draws for all conditions and within-question contrasts.",
                      "interpretation": "Exploratory intervals conditional on the selected subset and grouping assumptions; not an estimate of human-rater validity or population representativeness."},
        "groupings": {}, "score_distribution": {},
        "inputs_sha256": {str(p.relative_to(ROOT)): digest(p) for p in
                          (SOURCE / "ai_prescore_manifest.json", REVIEW / "private/english_blind_mapping.json",
                           REVIEW / "selected_questions.jsonl", *(SOURCE / "graded" / f"shard_{i}.jsonl" for i in (1, 2, 3)))},
        "analysis_script_sha256": digest(Path(__file__)),
        "limitations": ["Single AI evaluation workflow; no independent human ratings or inter-rater agreement.",
            "Exact judge model revision and decoding parameters were not independently logged in this interactive workflow.",
            "Three parallel grading shards and AI internal calibration are not independent expert ratings.",
            "AI reference judgments have not replaced the official references or frozen automatic metrics.",
            "AI citation-link support counts use semantic citation scopes and may differ from strict automatic ID-format parsing.",
            "Source availability and wording can reveal configuration despite omitted explicit condition labels.",
            "Score intervals quantify question/source sampling variation, not AI-judge error or repeated-judge variability."]}
    for grouping in ("source_group", "source_group_conservative"):
        groups = sorted({row[grouping] for row in rows})
        rng = np.random.default_rng(SEED)
        draws = rng.integers(0, len(groups), (REPLICATES, len(groups)))
        summary, samples = {}, {}
        for condition in CONDITIONS:
            subset = [r for r in rows if r["condition"] == condition]
            summary[condition], samples[condition] = {}, {}
            for metric in METRICS:
                summary[condition][metric], samples[condition][metric] = estimate(subset, metric, grouping, groups, draws)
        paired = {}
        for left, right in (("lora", "base"), ("base_rag", "base"), ("lora_rag", "base_rag"), ("lora_rag", "lora")):
            values = {}
            for metric in METRICS:
                a, b = summary[left][metric], summary[right][metric]
                if a["mean"] is None or b["mean"] is None:
                    continue
                differences = samples[left][metric] - samples[right][metric]
                valid = np.isfinite(differences)
                values[metric] = {"mean": a["mean"] - b["mean"],
                    "ci95": np.quantile(differences[valid], [.025, .975]).tolist() if valid.sum() else None,
                    "paired_questions": 40, "valid_bootstrap_replicates": int(valid.sum())}
            paired[f"{left}_minus_{right}"] = values
        result["groupings"][grouping] = {"source_groups": len(groups), "by_condition": summary, "paired_differences": paired}
    for condition in CONDITIONS:
        subset = [r for r in rows if r["condition"] == condition]
        result["score_distribution"][condition] = {field: {str(score): sum(r[field] == score for r in subset) for score in range(5)}
            for field in ("ai_factual_correctness", "ai_essential_completeness")}
    OUT.mkdir(parents=True, exist_ok=False)
    write_json(OUT / "analysis.json", result)
    with (OUT / "per_answer_ai_scores.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        fields = ("item_id", "id", "condition", "source_group", "source_group_conservative", "rating_origin", "ai_factual_correctness", "ai_essential_completeness", "ai_quality_reason_zh", "ai_reference_problem_zh", "ai_uncertainty_reason_zh", "ai_evidence")
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "ai_evidence": json.dumps(row["ai_evidence"], ensure_ascii=False) if row["ai_evidence"] else ""})
    write_json(OUT / "per_answer_ai_scores.json", rows)
    primary = result["groupings"]["source_group"]
    lines = ["# 已采纳的AI评价结果", "", "采用日期：2026-10-18。用户转述老师同意直接使用AI建议分；本报告将它们作为AI评价采用。没有新增或代填真人逐条评分，没有把AI分写成双人人评。", "",
        "## 范围与评分来源", "", "正式生成前固定抽取40题，每题4条回答，共160条质量评分；其中两个RAG条件共80条证据评分。该子集来自公开answer_in_text=1的206题，不是227题全量人工／AI逐条评价。31题来自corpus1、9题来自corpus2。", "",
        "三份分片由同一AI评价流程阅读，另有AI内部一致性校对；这不是3位独立评审。原文、逐陈述理由及最终评分已归档。精确评审模型构建版本与解码参数没有独立记录，不能声称已完成可重复的跨模型评审验证。", "",
        "## 四组AI评分", "", "均分范围0—4。方括号为按40个来源组进行2000次配对bootstrap的95%区间（seed=20261018），只反映该样本的来源抽样变动，不包含AI判断误差。", "",
        "| 条件 | AI事实答题质量 | AI关键要点完整性 | AI证据忠实性 |", "|---|---:|---:|---:|"]
    for c in CONDITIONS:
        m = primary["by_condition"][c]
        lines.append(f"| {NAMES[c]} | {show(m['factual_quality'])} | {show(m['completeness'])} | {show(m['faithfulness'])} |")
    lines += ["", "纯拒答没有给出所求事实时，答题质量和完整性为0；若实际给定资料确实不足，忠实性仍可高分。三个维度分别解释，不能以高忠实性替代完成题目。0—4均分不转换成事实准确率。", "",
        "## AI核查的引用支持与覆盖", "", "按陈述—引用关联及需引用陈述的合计数计算，不对逐回答百分比简单取平均；分母0时不适用。", "",
        "| 条件 | 受支持关联／全部关联 | AI引用支持率 [95% CI] | 有支持引用陈述／需引用陈述 | AI引用覆盖率 [95% CI] |", "|---|---:|---:|---:|---:|"]
    for c in ("base_rag", "lora_rag"):
        m = primary["by_condition"][c]
        s, v = m["citation_support"], m["citation_coverage"]
        lines.append(f"| {NAMES[c]} | {int(s['numerator'])}/{int(s['denominator'])} | {show(s,True)} | {int(v['numerator'])}/{int(v['denominator'])} | {show(v,True)} |")
    lines += ["", "## 微调＋RAG相对基座＋RAG", "", "| AI指标 | 配对差值 [95% CI] |", "|---|---:|"]
    difference = primary["paired_differences"]["lora_rag_minus_base_rag"]
    for metric, label in (("factual_quality", "事实答题质量（0—4分）"), ("completeness", "完整性（0—4分）"),
                          ("faithfulness", "忠实性（0—4分）"), ("citation_support", "引用支持率（百分点）"), ("citation_coverage", "引用覆盖率（百分点）")):
        lines.append(f"| {label} | {show(difference[metric],metric.startswith('citation_'))} |")
    lines += ["", "差值用未四舍五入数值计算。引用支持的点估计上升，不代表所有维度同时改善；需结合差值区间和引用覆盖率。区间不用于宣称真人认可或AI评审有效性。", "",
        f"保守来源分组将corpus2合并为一个未解析组，共{result['groupings']['source_group_conservative']['source_groups']}组；其完整区间与配对结果保存于analysis.json。该处理是来源依赖敏感性假设，不证明这些资料来自同一篇文献。", "",
        "## 与正式227题自动结果的关系", "", "正式选择题89.9%与开放问答去引用F1 35.7%仍是全227题对应自动指标。这里40题AI评分单列报告，不替换公开标签、不重写正式分数。21题参考适用性AI预审仅用于指出疑问，未据此事后更改主结果。", "",
        "## 当前采用方式", "", "本部分可表述为AI-based evaluation或AI-judged answer quality。当前独立人类评分人数为0，因此不报告专家盲评结果、人工事实准确率、Cohen κ或双人一致性。原空白人评模板作为历史材料保留，不再作为当前采用AI评价方案的必填步骤。", ""]
    (OUT / "report.md").write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
    print(f"Adopted AI evaluation: {OUT}; 160 quality scores, 80 evidence scores, 40 questions; independent human raters=0")
    print("lora_rag vs base_rag", json.dumps(difference, ensure_ascii=False))


if __name__ == "__main__":
    main()

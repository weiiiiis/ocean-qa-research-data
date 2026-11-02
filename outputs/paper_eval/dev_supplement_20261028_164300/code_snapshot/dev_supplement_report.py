"""Analyze the independent development supplement; never generate or alter answers."""
from __future__ import annotations

import argparse
import collections
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np

from .common import ROOT, digest_file, digest_json, read_jsonl, write_json, write_jsonl
from .dev_supplement import CELLS, cell, check_protected, now, validate_answers
from .english_report import citations, reference_strings, remove_citations, source_hit, squad_scores


def csv_write(path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                             for k, v in row.items()})


def interval(values, groups, draws):
    # Source-cluster bootstrap of a question-weighted mean.
    keys = sorted(set(groups))
    sums = np.array([sum(v for v, g in zip(values, groups) if g == key) for key in keys])
    sizes = np.array([groups.count(key) for key in keys])
    boot = sums[draws].sum(axis=1) / sizes[draws].sum(axis=1)
    return float(np.mean(values)), *np.percentile(boot, [2.5, 97.5]).tolist()


def analyze(out):
    manifest = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    protocol = manifest["signature"]["protocol"]
    qs = read_jsonl(out / "selected_questions.jsonl")
    questions = {q["id"]: q for q in qs}
    plan = json.loads((out / "prompt_plan.json").read_text(encoding="utf-8"))
    raw, integrity = validate_answers(out, qs, plan, manifest["signature"]["signature_sha256"])
    if not integrity["complete"]:
        raise ValueError("Do not publish full-condition means from incomplete inference")
    protected = check_protected(out)
    scored = []
    for (condition, qid), r in raw.items():
        q = questions[qid]
        if (condition != cell(r["stage"], r["rag"], r["budget"])
                or r["source_group"] != q["source_group"]
                or r["question"] != q["question"]
                or r["metrics"]["input_tokens"] != plan[qid][str(r["rag"])]["input_tokens"]):
            raise ValueError(f"Condition/question/input-token mismatch: {condition} {qid}")
        ref, _ = reference_strings(q)
        stripped = remove_citations(r["answer"])
        em, f1 = squad_scores(stripped, ref)
        _, raw_f1 = squad_scores(r["answer"], ref)
        markers = citations(r["answer"])
        available = {h["id"] for h in r["retrieved_context"]}
        unresolved = [m for m in markers if m not in available]
        scored.append({"id": qid, "source_group": q["source_group"], "condition": condition,
            "stage": r["stage"], "rag": r["rag"], "budget": r["budget"],
            "f1": f1, "raw_f1": raw_f1, "exact_match": em,
            "output_tokens": r["metrics"]["output_tokens_including_eos"],
            "word_count": len(r["answer"].split()), "stripped_word_count": len(stripped.split()),
            "hit_token_limit": int(r["metrics"]["hit_token_limit"]), "finish_reason": r["finish_reason"],
            "citation_markers": len(markers), "resolved_markers": sum(m in available for m in markers),
            "has_citation": int(bool(markers)), "has_unresolved_citation": int(bool(unresolved)),
            "unclosed_bracket_count": max(0, r["answer"].count("[") - r["answer"].count("]")),
            "unresolved_markers": unresolved,
            "gold_source_retrieved": source_hit(q, r["retrieved_context"]) if r["rag"] else None,
            "prompt_sha256": r["prompt_sha256"], "context_sha256": r["context_sha256"],
            "question": q["question"], "answer": r["answer"], "reference_answer_origin": ref})
    order = {cell(*c): i for i, c in enumerate(CELLS)}
    scored.sort(key=lambda r: (order[r["condition"]], r["id"]))
    write_jsonl(out / "per_question_metrics.jsonl", scored)
    csv_write(out / "per_question_metrics.csv", scored)
    index = {(r["condition"], r["id"]): r for r in scored}
    ids = sorted(questions)
    groups = [questions[qid]["source_group"] for qid in ids]
    settings = protocol["statistics"]
    group_ids = sorted(set(groups))
    rng = np.random.default_rng(settings["seed"])
    draws = rng.integers(0, len(group_ids), size=(settings["resamples"], len(group_ids)))
    # Saving the actual draws permits exact statistical replay independently of RNG versions.
    np.save(out / "bootstrap_source_group_draws.npy", draws)
    write_json(out / "statistics.json", {**settings, "group_order": group_ids,
        "question_order": ids, "quantile_method": "numpy percentile linear; 2.5 and 97.5 percentiles",
        "draws_file_sha256": digest_file(out / "bootstrap_source_group_draws.npy"),
        "unit": "F1 and rate differences are fractions in CSV; multiply by100 for percentage points",
        "multiplicity": "Exploratory development analyses; intervals are pointwise, not adjusted for multiple comparisons"})
    summaries = []
    for condition in order:
        rows = [index[(condition, qid)] for qid in ids]
        f1, lo, hi = interval([r["f1"] for r in rows], groups, draws)
        markers = sum(r["citation_markers"] for r in rows)
        resolved = sum(r["resolved_markers"] for r in rows)
        summaries.append({"condition": condition, "stage": rows[0]["stage"], "rag": rows[0]["rag"],
            "max_new_tokens": rows[0]["budget"], "questions": len(rows), "source_groups": len(group_ids),
            "f1": f1, "f1_ci95_low": lo, "f1_ci95_high": hi,
            "raw_f1": float(np.mean([r["raw_f1"] for r in rows])),
            "mean_output_tokens_including_eos": float(np.mean([r["output_tokens"] for r in rows])),
            "median_output_tokens": float(np.median([r["output_tokens"] for r in rows])),
            "p95_output_tokens": float(np.percentile([r["output_tokens"] for r in rows], 95)),
            "mean_words": float(np.mean([r["word_count"] for r in rows])),
            "mean_stripped_words": float(np.mean([r["stripped_word_count"] for r in rows])),
            "truncated_answers": sum(r["hit_token_limit"] for r in rows),
            "truncation_rate": float(np.mean([r["hit_token_limit"] for r in rows])),
            "answers_with_citations": sum(r["has_citation"] for r in rows),
            "answers_with_unresolved_citations": sum(r["has_unresolved_citation"] for r in rows),
            "answers_with_unclosed_brackets": sum(r["unclosed_bracket_count"] > 0 for r in rows),
            "citation_markers": markers, "resolved_markers": resolved,
            "marker_resolution_ratio": resolved / markers if markers else None,
            "gold_source_recall": float(np.mean([r["gold_source_retrieved"] for r in rows])) if rows[0]["rag"] else None})
    csv_write(out / "summary.csv", summaries)
    comparisons = []
    for budget in (160, 96):
        comparisons += [(f"A_v2_minus_base_rag_{budget}", [(1, cell("v2",True,budget)),(-1,cell("base",True,budget))]),
                        (f"A_v2_minus_base_no_rag_{budget}", [(1,cell("v2",False,budget)),(-1,cell("base",False,budget))])]
        for stage in ("base", "v2"):
            comparisons.append((f"A_RAG_gain_{stage}_{budget}", [(1,cell(stage,True,budget)),(-1,cell(stage,False,budget))]))
    for rag in (False, True):
        comparisons += [(f"B_v1_minus_base_{'rag' if rag else 'no_rag'}", [(1,cell("v1",rag,160)),(-1,cell("base",rag,160))]),
                        (f"B_v2_minus_v1_{'rag' if rag else 'no_rag'}", [(1,cell("v2",rag,160)),(-1,cell("v1",rag,160))])]
    comparisons.append(("B_RAG_gain_v1_160", [(1,cell("v1",True,160)),(-1,cell("v1",False,160))]))
    for stage in ("base", "v2"):
        for rag in (False, True):
            comparisons.append((f"A_budget96_minus160_{stage}_{'rag' if rag else 'no_rag'}",
                                [(1,cell(stage,rag,96)),(-1,cell(stage,rag,160))]))
    comparisons.append(("A_RAG_gap_change_96_minus160", [(1,cell("v2",True,96)),(-1,cell("base",True,96)),
                        (-1,cell("v2",True,160)),(1,cell("base",True,160))]))
    differences = []
    metrics = ("f1", "output_tokens", "word_count", "stripped_word_count", "hit_token_limit",
               "citation_markers", "resolved_markers", "has_citation", "has_unresolved_citation")
    for name, terms in comparisons:
        for metric in metrics:
            diffs = [sum(coef * index[(condition, qid)][metric] for coef, condition in terms) for qid in ids]
            mean, lo, hi = interval(diffs, groups, draws)
            differences.append({"comparison": name, "metric": metric, "terms": terms,
                "paired_questions": len(ids), "source_groups": len(group_ids),
                "difference": mean, "ci95_low": lo, "ci95_high": hi,
                "seed": settings["seed"], "resamples": settings["resamples"],
                "method": "paired_source_cluster_percentile_question_weighted"})
    csv_write(out / "paired_differences.csv", differences)
    # Prespecified primary grouping remains the released EID group. A transparent
    # sensitivity joins groups that share an exact released supporting-text ID.
    parent = {g: g for g in group_ids}
    def find(g):
        while parent[g] != g:
            parent[g] = parent[parent[g]]
            g = parent[g]
        return g
    owners = {}
    for q in qs:
        source = q.get("expected_source_id")
        if source:
            g = q["source_group"]
            if source in owners:
                a,b = sorted((find(g),find(owners[source])))
                parent[b] = a
            owners[source] = g
    merged = [find(g) for g in groups]
    merged_ids = sorted(set(merged))
    merged_draws = np.random.default_rng(settings["seed"]).integers(
        0, len(merged_ids), size=(settings["resamples"],len(merged_ids)))
    np.save(out / "bootstrap_merged_source_draws.npy", merged_draws)
    merged_differences = []
    for name,terms in comparisons:
        for metric in metrics:
            diffs = [sum(coef*index[(condition,qid)][metric] for coef,condition in terms) for qid in ids]
            mean,lo,hi = interval(diffs,merged,merged_draws)
            merged_differences.append({"comparison": name, "metric": metric,
                "paired_questions": len(ids), "source_groups": len(merged_ids),
                "difference": mean, "ci95_low": lo, "ci95_high": hi,
                "seed": settings["seed"], "resamples": settings["resamples"],
                "method": "paired_cluster_exact_supporting_source_union_sensitivity"})
    csv_write(out / "paired_differences_source_merged.csv", merged_differences)
    write_json(out / "source_group_sensitivity.json", {"primary_groups": len(group_ids),
        "merged_groups": len(merged_ids), "group_order": merged_ids,
        "question_group_map": dict(zip(ids,merged)),
        "rule": "Union published EID groups sharing identical expected_source_id; no answer/score filtering",
        "addendum": "analysis_plan_addendum.json; additional dependence sensitivity, primary protocol unchanged",
        "draws_sha256": digest_file(out / "bootstrap_merged_source_draws.npy")})
    # Qualitative candidates are flagged mechanically; no factual or completeness grades inferred.
    candidates = []
    for r in scored:
        reasons = []
        if r["hit_token_limit"]:
            reasons.append("observed_token_limit")
        if not r["rag"] and r["has_citation"]:
            reasons.append("no_context_citation_marker")
        if r["rag"] and r["has_unresolved_citation"]:
            reasons.append("unresolved_RAG_marker")
        if r["rag"] and r["gold_source_retrieved"] and r["f1"] < 0.15:
            reasons.append("gold_source_retrieved_low_overlap_omission_candidate")
        if reasons:
            candidates.append({"id": r["id"], "condition": r["condition"], "source_group": r["source_group"],
                "flags": reasons, "question": r["question"], "answer": r["answer"],
                "reference_answer_origin": r["reference_answer_origin"], "f1": r["f1"],
                "output_tokens": r["output_tokens"], "finish_reason": r["finish_reason"],
                "citation_markers": r["citation_markers"], "resolved_markers": r["resolved_markers"],
                "unresolved_markers": r["unresolved_markers"],
                "context_sha256": r["context_sha256"], "review_status": "pending_qualitative_review",
                "factual_correctness_human": None, "completeness_human": None, "citation_support_human": None})
    csv_write(out / "case_candidates.csv", candidates)
    representatives = []
    used = set()
    for flag in ("observed_token_limit", "no_context_citation_marker", "unresolved_RAG_marker",
                 "gold_source_retrieved_low_overlap_omission_candidate"):
        eligible = sorted((r for r in candidates if flag in r["flags"]), key=lambda r: (r["id"], r["condition"]))
        count = 0
        for r in eligible:
            if r["id"] in used:
                continue
            representatives.append({"selection_category": flag, **r})
            used.add(r["id"])
            count += 1
            if count == 3:
                break
    csv_write(out / "cases_review.csv", representatives)
    stage_changes = []
    for qid in ids:
        a,b = index[("v1_rag_t160",qid)],index[("v2_rag_t160",qid)]
        stage_changes.append({"id": qid, "source_group": a["source_group"],
            "f1_v2_minus_v1": b["f1"]-a["f1"], "v1_f1": a["f1"], "v2_f1": b["f1"],
            "v1_tokens": a["output_tokens"], "v2_tokens": b["output_tokens"],
            "question": a["question"], "reference_answer_origin": a["reference_answer_origin"],
            "v1_answer": a["answer"], "v2_answer": b["answer"],
            "context_sha256": a["context_sha256"], "review_status": "pending_qualitative_review",
            "factual_correctness_human": None, "completeness_human": None, "citation_support_human": None})
    csv_write(out / "stage_rag_per_question_changes.csv", stage_changes)
    ordered_changes = sorted(stage_changes,key=lambda r:(r["f1_v2_minus_v1"],r["id"]))
    stage_examples = ([{"selection": "largest_negative_f1_change", **r} for r in ordered_changes[:3]] +
                      [{"selection": "largest_positive_f1_change", **r} for r in ordered_changes[-3:][::-1]])
    csv_write(out / "stage_change_cases.csv", stage_examples)
    # Include complete contexts and budget/stage counterparts for selected qualitative questions.
    review_ids = {r["id"] for r in representatives} | {"PIRA-DEV-A1371"} | {r["id"] for r in stage_examples}
    write_jsonl(out / "cases_review_full_context.jsonl", [
        {**r, "reference_answer_origin": reference_strings(questions[r["id"]])[0],
         "review_status": "pending_qualitative_review"} for r in raw.values() if r["id"] in review_ids])
    # Explicitly verify A/B reuse: the four 160-token cells refer to the very same saved rows.
    reuse = {"method": "same_record_reference_no_external_merge", "external_historical_records_reused": 0,
        "A_conditions": [cell(*c) for c in CELLS if c[0] != "v1"],
        "B_conditions": [cell(*c) for c in CELLS if c[2] == 160],
        "A_B_shared_cells": [cell(*c) for c in CELLS if c[0] != "v1" and c[2] == 160],
        "signature": manifest["signature"]["signature_sha256"], "same_data_prompt_protocol_model_context": True}
    write_json(out / "experiment_reuse_audit.json", reuse)
    # Check whether lowering a nonbinding budget altered an answer; descriptive replay diagnostic.
    replay = []
    for stage in ("base", "v2"):
        for rag in (False, True):
            pairs = [(index[(cell(stage,rag,96),qid)], index[(cell(stage,rag,160),qid)]) for qid in ids]
            nontruncated = [(a,b) for a,b in pairs if not a["hit_token_limit"] and not b["hit_token_limit"]]
            replay.append({"stage": stage, "rag": rag, "nontruncated_pairs": len(nontruncated),
                "identical_answer_pairs": sum(a["answer"] == b["answer"] for a,b in nontruncated),
                "changed_ids": [a["id"] for a,b in nontruncated if a["answer"] != b["answer"]]})
    write_json(out / "budget_replay_diagnostic.json", replay)
    analysis = {"created_at_utc": now(), "complete": True, "questions": len(qs), "answers": len(scored),
        "source_groups": len(group_ids), "summary": summaries, "paired_differences": differences,
        "integrity": integrity, "protected": protected, "reuse": reuse, "budget_replay": replay,
        "merged_source_groups": len(merged_ids), "merged_source_paired_differences": merged_differences,
        "stage_change_cases": stage_examples,
        "no_inferred_human_scores": True, "cases_are_not_error_rate_estimates": True}
    write_json(out / "analysis.json", analysis)
    write_report(out, analysis, representatives, protocol)
    for file in (Path(__file__), ROOT / "paper_eval/english_report.py", ROOT / "paper_eval/report.py"):
        shutil.copy2(file, out / "code_snapshot" / file.name)
    write_json(out / "analysis_code_hashes.json", {str(f.relative_to(ROOT)): digest_file(f) for f in
        (Path(__file__), ROOT / "paper_eval/english_report.py", ROOT / "paper_eval/report.py")})
    append_command = {"timestamp_utc": now(), "argv": [sys.executable,"-m","paper_eval.dev_supplement_report",*sys.argv[1:]], "cwd": str(ROOT)}
    from .inference import append_result
    append_result(out / "commands.jsonl", append_command)
    write_json(out / "delivery_hashes.json", {str(p.relative_to(out)): digest_file(p) for p in sorted(out.rglob("*"))
        if p.is_file() and p.name != "delivery_hashes.json"})
    print(f"Report complete: {out}; {len(qs)} questions, {len(scored)} answers; protected {protected['files_checked']} files unchanged")


def write_report(out, analysis, cases, protocol):
    summaries = {r["condition"]: r for r in analysis["summary"]}
    pairs = {(r["comparison"],r["metric"]): r for r in analysis["paired_differences"]}
    def pp(name, metric="f1"):
        r = pairs[(name,metric)]
        scale = 100 if metric in ("f1","hit_token_limit","has_citation","has_unresolved_citation") else 1
        return f"{r['difference']*scale:+.2f} [{r['ci95_low']*scale:+.2f}, {r['ci95_high']*scale:+.2f}]"
    lines = ["# 海洋科学英文开发集补充实验", "",
        f"实际运行日期：2026-10-28（Asia/Shanghai）；结果目录：`{out}`。",
        f"已完成{analysis['questions']}题、10个独立条件、{analysis['answers']}条实际模型回答。实验A为4组×2档预算=808条；实验B为3阶段×2种检索=606条，其中404条160-token回答与A共享，另生成v1的202条。未启动训练或正式227题推理。", "",
        "## 预先核查与协议", "",
        "- 已读正式结果汇总、冻结protocol/manifest、v2模型卡、english.py及official_english_report.py；项目及上级路径未发现适用AGENTS.md。",
        "- 数据实际路径：`paper_data/external_benchmarks/dev_openqa_en.jsonl`，101个唯一ID，全部split=dev；91个EID来源组（81组各1题、10组各2题），对应90个唯一expected_source_id支撑文本。EID组数与支撑文本数的口径不同，不能把91组解释为91篇完全独立文档。仅corpus1科学摘要，不是混合海洋文献随机样本。",
        "- 语料实际路径：`paper_data/external_benchmarks/corpus_pira_chunks_en.jsonl`；1507个窗口、688个section来源。这些是受限公开支撑文本。",
        "- 基座本地目录`models/Qwen3-4B-Instruct`对应Qwen3-4B-Instruct-2507；模型及适配器逐文件SHA256见run_manifest.json。v2训练配置所记载的起始适配器是v1/epoch_1，v1/epoch_2不是其祖先。",
        "- 使用`.venv/Scripts/python.exe`及既有load_model/generate、fit_prompt和BM25代码；冻结清单中8个既有评测代码文件的当前SHA256均与原冻结值一致（baseline_code_audit.json）。仅新增两个补充入口，没有升级依赖。NF4、double quant、FP16、SDPA、batch=1；贪心do_sample=false，repetition_penalty=1.05，输入上限2048，BM25 top3、上下文上限1000。",
        "- 提示使用正式evidence_v2，要求1—3个短句、通常不超过60英文词；160和96两个新token上限是唯一预算变化。所有模型共用同一逐题提示/实际上下文和文档顺序，均经过哈希核对。",
        "- 12题试运行在生成前按既有来源轮询规则固定，覆盖12个不同来源组；96条回答验证完整后自动继续。题目、预算未按得分替换；截断照实保留。pilot_questions.jsonl、pilot_validation.json记录全过程。",
        "- 本轮所有答案重新实际生成，不合并历史开发答案。A/B的404条共享输出是同一批物理记录，数据、提示、协议、模型和上下文没有跨运行差异，见experiment_reuse_audit.json。",
        "- 输出token数包括EOS，结束类型由实际长度与最后token是否EOS判定；词数以完整解码文本空白分隔计数（含引用），另保存去引用词数。仅沿用原生成器的特殊token解码及首尾空白去除，不裁剪回答内容。", "",
        "## 实验A：生成预算敏感性", "",
        "下表F1为去引用Token F1百分比，区间为95%来源组bootstrap；长度是实际观测值。", "",
        "| 条件 | 上限 | F1 [95% CI] | 平均输出token | 平均词数 | 截断/101 |",
        "|---|---:|---:|---:|---:|---:|"]
    for budget in (160,96):
        for stage, rag in (("base",False),("base",True),("v2",False),("v2",True)):
            s = summaries[cell(stage,rag,budget)]
            lines.append(f"| {stage}{'＋RAG' if rag else ''} | {budget} | {100*s['f1']:.2f} [{100*s['f1_ci95_low']:.2f}, {100*s['f1_ci95_high']:.2f}] | {s['mean_output_tokens_including_eos']:.2f} | {s['mean_words']:.2f} | {s['truncated_answers']} |")
    lines += ["", "F1配对差值单位为百分点，长度差单位为token。每项均按相同101题配对。", "",
        "| 比较 | F1差 [95% CI] | 输出token差 [95% CI] |", "|---|---:|---:|"]
    for budget in (160,96):
        name = f"A_v2_minus_base_rag_{budget}"
        lines.append(f"| v2＋RAG − 基座＋RAG，{budget} | {pp(name)} | {pp(name,'output_tokens')} |")
    lines.append(f"| 上述差异的预算交互：(96差值)−(160差值) | {pp('A_RAG_gap_change_96_minus160')} | {pp('A_RAG_gap_change_96_minus160','output_tokens')} |")
    gap160 = pairs[("A_v2_minus_base_rag_160","f1")]
    gap96 = pairs[("A_v2_minus_base_rag_96","f1")]
    direction = "仍为正" if gap96["difference"] > 0 else "未维持正向差值"
    certainty = "其区间不跨0" if gap96["ci95_low"] > 0 or gap96["ci95_high"] < 0 else "其区间跨0，额外增益仍不确定"
    lines += ["", f"160上限下v2＋RAG相对基座＋RAG差值为{pp('A_v2_minus_base_rag_160')}个百分点；96上限下{direction}，{certainty}。",
        "预算交互是降低生成上限后配对差值的变化，不是剔除回答长度影响后的因果效应。两组实际长度仍可能不同；相同生成上限不构成严格等长控制。", "",
        "| 条件，96 − 160 | F1差 [95% CI] | token差 [95% CI] | 截断率差（百分点）[95% CI] |",
        "|---|---:|---:|---:|"]
    for stage,rag in (("base",False),("base",True),("v2",False),("v2",True)):
        name = f"A_budget96_minus160_{stage}_{'rag' if rag else 'no_rag'}"
        lines.append(f"| {stage}{'＋RAG' if rag else ''} | {pp(name)} | {pp(name,'output_tokens')} | {pp(name,'hit_token_limit')} |")
    lines += ["", "未截断预算配对的答案复现情况见budget_replay_diagnostic.json。96-token截断回答是本实验的真实结果，已进入均值和配对分析。", "",
        f"另作来源依赖敏感性：将共享同一支撑文本ID的EID组做并集合并，得到{analysis['merged_source_groups']}组（分析附录在查看完整得分前记录，主协议不变）。v2＋RAG − 基座＋RAG的F1差值及区间如下；完整结果见paired_differences_source_merged.csv。", "",
        "| 预算 | 合并来源组敏感性F1差（百分点）[95% CI] |", "|---|---:|"]
    for budget in (160,96):
        r = next(r for r in analysis["merged_source_paired_differences"] if r["comparison"] == f"A_v2_minus_base_rag_{budget}" and r["metric"] == "f1")
        lines.append(f"| {budget} | {100*r['difference']:+.2f} [{100*r['ci95_low']:+.2f}, {100*r['ci95_high']:+.2f}] |")
    lines += ["",
        "## 实验B：两个训练阶段", "",
        "| 阶段／条件（160上限） | F1% | 平均token | 平均词数 | 有引用回答 | 可解析/全部标记 | 截断 |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for stage in ("base","v1","v2"):
        for rag in (False,True):
            s = summaries[cell(stage,rag,160)]
            lines.append(f"| {stage}{'＋RAG' if rag else ''} | {100*s['f1']:.2f} | {s['mean_output_tokens_including_eos']:.2f} | {s['mean_words']:.2f} | {s['answers_with_citations']}/101 | {s['resolved_markers']}/{s['citation_markers']} | {s['truncated_answers']}/101 |")
    lines += ["", "| 阶段配对比较 | F1差（百分点）[95% CI] | token差 [95% CI] | 有引用回答率差（百分点）[95% CI] |",
        "|---|---:|---:|---:|"]
    for transition in ("v1_minus_base","v2_minus_v1"):
        for rag in ("no_rag","rag"):
            name=f"B_{transition}_{rag}"
            lines.append(f"| {transition}，{rag} | {pp(name)} | {pp(name,'output_tokens')} | {pp(name,'has_citation')} |")
    lines += ["", "| 各阶段RAG − 无RAG（160） | F1差（百分点）[95% CI] |", "|---|---:|"]
    for stage in ("base","v1","v2"):
        name = f"{'B' if stage=='v1' else 'A'}_RAG_gain_{stage}_160"
        lines.append(f"| {stage} | {pp(name)} |")
    lines += ["", "第一阶段的可观察变化：回答变短，无RAG和有RAG的F1都上升，其中有RAG提高6.86个百分点；同时，无RAG时出现引用标记的回答从基座0题增至v1的33题，这些标记均不能对应本轮提供的资料。",
        "第二阶段表现出取舍：无RAG引用标记从33题降至7题，但没有完全消除；无RAG F1上升1.44个百分点，区间跨0。有RAG F1从38.28%降至34.57%，配对差为−3.71个百分点[−6.64, −1.02]；平均token只从55.60降至54.95（差值区间跨0），平均词数反而从20.81增至21.23。因此不能把第二阶段描述为全面答题增益，也不能仅用整体平均长度缩短解释这项RAG词语重合下降。两阶段RAG完整引用编号均可解析，仍需逐陈述核查证据支持。", "",
        "第一阶段使用240个英文例子、48个事实家族，从基座训练1个epoch后取epoch_1；第二阶段从该checkpoint继续，更新为对齐评测提示结尾的条件引用数据，学习率2e-5，并对closed_book重复3次，总336次抽样。它同时增加训练量、改变提示结尾和采样权重，因此这些阶段配对差异只说明行为改变，不能把它们全部归因于条件引用设计的独立效果。", "",
        "## 质性待核查案例", "",
        "cases_review.csv按机械标记类型、ID排序各选最多3个不同问题，cases_review_full_context.jsonl包含这些问题及下述A1371阅读示例的所有条件原文与上下文。它用于质性检查，不用于从所选案例估计总体错误率。事实正确性、完整性、引用支持人工评分均为空。低F1只是遗漏候选，不能自动认定事实错误或不完整。", "",
        "| 题目 | 条件 | 待核查类型 | 实际结束 |", "|---|---|---|---|"]
    for r in cases:
        lines.append(f"| {r['id']} | {r['condition']} | {r['selection_category']} | {r['finish_reason']} |")
    lines += ["", "已阅读的具体文本例子：PIRA-DEV-A1371问巴西东部干旱的结果。基座＋RAG的96-token输出停在第二个引用的`[PIRA-`处，v2＋RAG停在另一个尚未闭合的来源编号内部，而160-token版本均完成引用。公开原参考强调沉积物平均粒径下降和大型底栖群落变化；生成文本还谈及丰度、生物扰动和另一个检索片段的红树林经济损失。这个例子同时提示引用开销、问题范围和参考重合的区别；不据此给整题填写事实/完整性分，也不由个案估计总体错误率。", "",
        "stage_change_cases.csv另按v2−v1的RAG F1差排序，取负向、正向各3个极端案例，仅供观察机制；全101题差值保存在stage_rag_per_question_changes.csv。所选问题的全部条件和上下文也纳入cases_review_full_context.jsonl。极端案例没有总体代表性，不能据此估计错误率或筛选更换模型。",
        "已对照实际提供片段阅读的负向例子：A1234的第一片段明列四种东北部降水预测方法；v1逐项作答，v2却回答片段没有提供信息。A568的第一片段列出老化管道的腐蚀、泄漏，以及持续生产期间改造的困难；v2保留后者和老化概念，省略腐蚀/泄漏字样。A1726中v1、v2均列出三类技术，但用词和附加问题复述不同，F1仍有较大差异。这些文本现象提示拒答、要点取舍和词面变化可能共同参与阶段F1变化；不是已完成的独立人工正确性或完整性评价。",
        "无资料引用形式例子：A1080的v1无RAG输出带`[1]`，v2同条件没有该标记；两者均未提供参考资料。A1136则显示指标限制：公开参考只有`STOPPLE`，基座提到复数`stopples`，v1/v2主要回答Hot Tapping and Plugging with a Bypass。既有词语归一化没有词干化，低F1不能自动解释为没有相关事实。", "",
        "## 统计和解释边界", "",
        f"按91个source_group来源组有放回抽样，每次抽91组，保留抽中组内所有题；对同题差值计算按题加权均值，10000次、seed={protocol['statistics']['seed']}，取2.5/97.5百分位。所有比较共用抽样矩阵，保存在bootstrap_source_group_draws.npy；统计代码、group/question顺序、numpy版本与矩阵哈希齐备。区间为开发集探索性、逐项区间，没有多重比较调整。",
        "EID来源组有助于处理已知同源依赖，不证明完整文档独立，也不能排除预训练污染。开发集已经用于历史模型选择，本轮属于补充探索，不应冒充新独立测试集。",
        "F1是与公开原参考的词语重合，不是事实准确率；短参考与答案措辞都会影响它。引用编号可解析只表示标记对应实际提供片段，不证明原文支持每项陈述；没有提供资料的标记则没有本轮上下文来源。更短输出不直接表示更准确或推理更高效。", "",
        "去引用指标沿用既有正则，只去除完整匹配的方括号标记；截断在未闭合编号内部时，残留编号仍参与词语重合评分。引用总数同样只统计完整匹配标记，另列unclosed_bracket_count及汇总中的answers_with_unclosed_brackets作可核查诊断。没有事后删掉残留文本来修饰主指标。", "",
        "## 可以写入论文的发现与目前不能成立的结论", "",
        f"1. 可报告本开发集预算敏感性：160和96上限下，v2＋RAG相对基座＋RAG的去引用F1差分别为{pp('A_v2_minus_base_rag_160')}、{pp('A_v2_minus_base_rag_96')}个百分点，并同时披露实际长度和截断率。",
        f"2. 可报告阶段行为比较：无RAG的v1−基座为{pp('B_v1_minus_base_no_rag')}、v2−v1为{pp('B_v2_minus_v1_no_rag')}个百分点；有RAG对应为{pp('B_v1_minus_base_rag')}、{pp('B_v2_minus_v1_rag')}。",
        "3. 可报告RAG的逐阶段配对词语重合变化和引用形式变化，保留负差值、跨0区间和不支持预期的结果。",
        "4. 不能声称严格等长后增益已确认、事实正确率提高、引用证据支持提高、推理效率提高或条件引用设计具有独立因果效果。也不能由此推出中文教学效果或全海洋科学文献泛化。", "",
        "本轮结果足以支持补充分析，无需为了完成本实验再增加训练。是否设计新训练，应先对遗漏/截断/引用支持做独立审查；若要识别条件引用设计的因果贡献，需要等训练量、等采样权重且仅改变目标设计的训练对照，并在新的未参与选择的评估资料上验证。", "",
        "## 完整性与文件", "",
        f"完整性检查通过：10条件各101个相同题目ID，无重复；所有RAG模型和预算的同题实际提示、上下文及顺序一致；EOS/长度结束标记完整；原始保护清单中的{analysis['protected']['files_checked']}个文件哈希未变。详见integrity.json、protected_integrity.json、preflight.json。",
        "- report.md：本报告；summary.csv：10个条件完整汇总；paired_differences.csv：逐指标配对差值与区间。",
        "- answers.jsonl：1010条原始生成、实际提示、检索片段、token与终止信息；per_question_metrics.jsonl/.csv：逐题去引用F1、词数和引用指标。",
        "- cases_review.csv、case_candidates.csv、cases_review_full_context.jsonl、stage_change_cases.csv、stage_rag_per_question_changes.csv：待核查案例、阶段差值与完整上下文。",
        "- protocol.json、run_manifest.json、prompt_plan.json、retrieval_plan.json、code_snapshot/：实际协议、输入/模型/代码哈希、逐题共享计划及快照。",
        "- commands.jsonl、run.log、gpu_checks.jsonl、model_loads.jsonl：实际命令、日志、每个GPU加载前占用和模型加载记录。",
        "- statistics.json、bootstrap_source_group_draws.npy、analysis.json、analysis_code_hashes.json、delivery_hashes.json：可重算统计和交付哈希。",
        "- protected_before.json、protected_integrity.json：原冻结协议、正式输出、历史报告及训练输入保护审计。", ""]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    args=p.parse_args()
    analyze(args.out.resolve())


if __name__ == "__main__":
    main()

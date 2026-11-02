from __future__ import annotations

import collections
import csv
import json
import re
from pathlib import Path

import numpy as np

from .common import CONDITIONS, CONDITION_NAMES, DATA, digest_file, normalize, read_jsonl, write_json


def clustered_interval(values: list[tuple[str, float]], resamples: int = 2000) -> dict:
    groups = collections.defaultdict(list)
    for group, value in values:
        groups[group].append(value)
    if not groups:
        return {"mean": None, "ci95": None, "questions": 0, "source_groups": 0}
    arrays = list(groups.values())
    rng = np.random.default_rng(20261016)
    means = []
    for _ in range(resamples):
        draws = rng.integers(0, len(arrays), len(arrays))
        means.append(float(np.mean([v for i in draws for v in arrays[int(i)]])))
    return {"mean": float(np.mean([v for _, v in values])), "ci95": np.percentile(means, [2.5, 97.5]).tolist() if len(groups) > 1 else None,
            "questions": len(values), "source_groups": len(groups)}


def read_ratings(run: Path, paths: list[Path]) -> tuple[list[dict], dict]:
    if not paths:
        return [], {"status": "pending", "complete_quality_ratings": 0, "complete_evidence_ratings": 0}
    mapping = json.loads((run / "private/blind_mapping.json").read_text(encoding="utf-8"))
    if mapping["answers_sha256"] != digest_file(run / "answers.jsonl"):
        raise ValueError("Answers changed after packet creation; ratings cannot be joined")
    questions = {q["id"]: q for q in read_jsonl(run / "selected_questions.jsonl")}
    joined, seen_packets = [], set()
    for path in paths:
        export = json.loads(path.read_text(encoding="utf-8-sig"))
        packet_id = export["packet_id"]
        if packet_id in seen_packets:
            raise ValueError("Supply only the latest export for each packet, not multiple snapshots")
        seen_packets.add(packet_id)
        packet = mapping["packets"][packet_id]
        if export["kind"] != packet["kind"]:
            raise ValueError("Packet kind mismatch")
        seen = set()
        for rating in export["ratings"]:
            if rating["token"] in seen:
                raise ValueError("Duplicate rating token")
            seen.add(rating["token"])
            link = packet["items"][rating["token"]]
            # An incomplete export remains missing, never zero.
            if not rating.get("complete"):
                continue
            item = {**link, "reviewer": packet["reviewer"], "kind": packet["kind"], "notes": rating.get("notes", "")}
            if packet["kind"] == "quality":
                factual = rating.get("factual")
                points = rating.get("points", [])
                if factual not in [0, 1, 2, 3, 4] or len(points) != len(questions[link["id"]]["key_points"]) or any(p not in [0, 0.5, 1] for p in points):
                    raise ValueError("Invalid or missing quality rating")
                if questions[link["id"]]["mode"] == "popular" and rating.get("age") not in [0, 1, 2, 3, 4]:
                    raise ValueError("Popular answers need age-appropriateness rating")
                item.update(factual=factual, completeness=float(np.mean(points)), age=rating.get("age"))
            else:
                faith = rating.get("faithfulness")
                if faith not in [None, 0, 1, 2, 3, 4] or (faith is None and not item["notes"].strip()):
                    raise ValueError("Invalid evidence rating")
                fields = ["citation_links", "supported_links", "claims_requiring_citation", "claims_with_supported_citation"]
                counts = [rating.get(k) for k in fields]
                if any(type(v) is not int or v < 0 for v in counts) or counts[1] > counts[0] or counts[3] > counts[2]:
                    raise ValueError("Evidence counts are missing or inconsistent")
                item.update(faithfulness=faith, **dict(zip(fields, counts)))
                item["citation_precision"] = counts[1] / counts[0] if counts[0] else None
                item["citation_completeness"] = counts[3] / counts[2] if counts[2] else None
            joined.append(item)
    return joined, {"status": "partial_until_all_answers_have_two_independent_ratings",
        "complete_quality_ratings": sum(r["kind"] == "quality" for r in joined), "complete_evidence_ratings": sum(r["kind"] == "evidence" for r in joined)}


def report(run: Path, ratings: list[Path]) -> None:
    answers = read_jsonl(run / "answers.jsonl")
    probes = read_jsonl(run / "cost_probes.jsonl") if (run / "cost_probes.jsonl").exists() else []
    questions = {q["id"]: q for q in read_jsonl(run / "selected_questions.jsonl")}
    manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    if len({(r["condition"], r["id"]) for r in answers}) != len(answers):
        raise ValueError("Duplicate condition/question results")
    grouped = {(r["condition"], r["id"]): r for r in answers}
    for qid in questions:
        a, b = grouped.get(("base_rag", qid)), grouped.get(("lora_rag", qid))
        if a and b and (a["context_sha256"] != b["context_sha256"] or a["prompt_sha256"] != b["prompt_sha256"]):
            raise ValueError("Unfair RAG comparison: context or prompt differs")
    summaries = []
    for phase, rows in [("natural_answers", answers), ("fixed_128_tokens", probes)]:
        for condition in CONDITIONS:
            records = [r for r in rows if r["condition"] == condition]
            if not records:
                continue
            item = {"phase": phase, "condition": condition, "records": len(records), "questions": len({r['id'] for r in records})}
            for key in ["pipeline_ms", "pipeline_ttft_ms", "retrieval_ms"]:
                vals = [r[key] for r in records]
                item[key + "_mean"] = float(np.mean(vals))
                item[key + "_p50"] = float(np.median(vals))
                item[key + "_p95"] = float(np.percentile(vals, 95))
            for key in ["model_ttft_ms", "decode_tokens_per_second", "input_tokens", "output_tokens_including_eos", "output_characters", "peak_allocated_mib", "peak_reserved_mib", "resident_allocated_mib", "incremental_peak_allocated_mib"]:
                vals = [r["metrics"][key] for r in records if r["metrics"][key] is not None]
                item[key + "_mean"] = float(np.mean(vals)) if vals else None
                if key.startswith("peak_"):
                    item[key + "_max"] = max(vals)
            item["hit_token_limit_fraction"] = float(np.mean([r["metrics"]["hit_token_limit"] for r in records]))
            summaries.append(item)
    with (run / "performance_summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        fields = sorted({k for row in summaries for k in row})
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)
    retrieval_proxies = {}
    for condition in ["base_rag", "lora_rag"]:
        recalls, invalid, citations = [], 0, 0
        for row in answers:
            if row["condition"] != condition:
                continue
            shown = normalize(" ".join(h["text"] for h in row["retrieved_context"]))
            anchors = [e["anchor"] for e in questions[row["id"]]["evidence"]]
            recalls.append(float(np.mean([normalize(a) in shown for a in anchors])))
            ids = re.findall(r"\[(WB_p\d+_c\d+)\]", row["answer"])
            available = {c["id"] for c in row["retrieved_context"]}
            invalid += sum(i not in available for i in ids)
            citations += len(ids)
        retrieval_proxies[condition] = {"gold_anchor_recall_mean": float(np.mean(recalls)) if recalls else None,
            "formatted_citation_markers": citations, "markers_not_in_provided_context": invalid,
            "warning": "Anchor overlap and valid IDs do not establish factual accuracy, entailment, or citation correctness."}
    joined, review_status = read_ratings(run, ratings)
    quality = {}
    for condition in CONDITIONS:
        by_q = collections.defaultdict(list)
        for row in joined:
            if row["condition"] == condition and row["kind"] == "quality":
                by_q[row["id"]].append(row)
        # Primary analysis uses only answers rated by both independent reviewers.
        paired = {qid: vals for qid, vals in by_q.items() if len({r["reviewer"] for r in vals}) == 2}
        quality[condition] = {metric: clustered_interval([(questions[qid]["source_group"], float(np.mean([r[metric] for r in vals]))) for qid, vals in paired.items()]) for metric in ["factual", "completeness"]}
    averages = {}
    for condition in CONDITIONS:
        for qid in questions:
            vals = [r for r in joined if r["kind"] == "quality" and r["condition"] == condition and r["id"] == qid]
            if len({r["reviewer"] for r in vals}) == 2:
                averages[(condition, qid)] = float(np.mean([v["factual"] for v in vals]))
    comparisons = {}
    for a, b in [("lora", "base"), ("base_rag", "base"), ("lora_rag", "base_rag"), ("lora_rag", "lora")]:
        vals = [(q["source_group"], averages[(a, qid)] - averages[(b, qid)]) for qid, q in questions.items() if (a, qid) in averages and (b, qid) in averages]
        comparisons[a + "_minus_" + b] = clustered_interval(vals)
    interaction = [(q["source_group"], averages[("lora_rag", qid)] - averages[("lora", qid)] - averages[("base_rag", qid)] + averages[("base", qid)]) for qid, q in questions.items() if all((c, qid) in averages for c in CONDITIONS)]
    comparisons["lora_by_rag_interaction"] = clustered_interval(interaction)
    reliability = {}
    both = []
    for condition in CONDITIONS:
        for qid in questions:
            vals = {r["reviewer"]: r["factual"] for r in joined if r["kind"] == "quality" and r["condition"] == condition and r["id"] == qid}
            if len(vals) == 2:
                both.append((vals["reviewer_1"], vals["reviewer_2"]))
    if both:
        from sklearn.metrics import cohen_kappa_score
        a, b = zip(*both)
        # Kappa is undefined when all ratings fall in a single category.
        kappa = float(cohen_kappa_score(a, b, labels=[0, 1, 2, 3, 4], weights="quadratic")) if len(set(a) | set(b)) > 1 else None
        reliability = {"quadratic_weighted_kappa": kappa, "paired_answers": len(both), "exact_agreement": float(np.mean(np.array(a) == np.array(b)))}
    evidence = {}
    for condition in ["base_rag", "lora_rag"]:
        rows = [r for r in joined if r["kind"] == "evidence" and r["condition"] == condition]
        evidence[condition] = {"ratings": len(rows)}
        for metric in ["faithfulness", "citation_precision", "citation_completeness"]:
            vals = [r[metric] for r in rows if r[metric] is not None]
            evidence[condition][metric + "_mean"] = float(np.mean(vals)) if vals else None
    audit = json.loads((DATA / "overlap_audit.json").read_text(encoding="utf-8")) if (DATA / "overlap_audit.json").exists() else {}
    write_json(run / "analysis.json", {"performance": summaries, "retrieval_proxies": retrieval_proxies, "expert_review": review_status,
        "quality_two_reviewers_only": quality, "paired_factual_differences": comparisons, "inter_rater_reliability": reliability, "expert_evidence": evidence,
        "audit_status_counts_full_candidate_test": audit.get("status_counts"), "quality_ratings": joined,
        "primary_comparison": "lora_rag minus base_rag; all others secondary/exploratory",
        "analysis_boundary": "Confidence intervals resample source sections, not individual sibling questions. No factual score is inferred from keyword hits. Missing scores remain null. All current references are drafts pending expert approval."})
    lines = ["# 海洋问答评测结果", "", f"数据划分：{manifest['signature']['selection']['split']}；选择题数：{len(questions)}。", "",
        "## 自然回答的实测成本", "", "| 条件 | 完成数 | 平均输出token | 端到端均值 ms | 首token均值 ms | 解码 token/s | 显存峰值 MiB | 截断率 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summaries:
        if row["phase"] == "natural_answers":
            lines.append(f"| {CONDITION_NAMES[row['condition']]} | {row['records']} | {row['output_tokens_including_eos_mean']:.1f} | {row['pipeline_ms_mean']:.1f} | {row['pipeline_ttft_ms_mean']:.1f} | {row['decode_tokens_per_second_mean']:.1f} | {row['peak_allocated_mib_max']:.1f} | {row['hit_token_limit_fraction']:.1%} |")
    lines += ["", "## 固定生成工作量", "", "| 条件 | 探针次数 | token/次 | 端到端均值 ms | 首token均值 ms | 解码 token/s |", "|---|---:|---:|---:|---:|---:|"]
    for row in summaries:
        if row["phase"] == "fixed_128_tokens":
            lines.append(f"| {CONDITION_NAMES[row['condition']]} | {row['records']} | {row['output_tokens_including_eos_mean']:.0f} | {row['pipeline_ms_mean']:.1f} | {row['pipeline_ttft_ms_mean']:.1f} | {row['decode_tokens_per_second_mean']:.1f} |")
    lines += ["", "固定长度探针强制继续生成，只用于成本比较，不能用于答案质量评分。自然回答延迟同时受输入长度、输出长度、首token计算和解码速度影响。显存为PyTorch分配器峰值，不是整卡占用；端到端不含网络、队列和UI。", "",
        "## 专家评价", "", f"已导入完整质量评分 {review_status['complete_quality_ratings']} 份，完整证据评分 {review_status['complete_evidence_ratings']} 份。每个答案应由两人独立评价。", "",
        "事实正确性、完整性、引用支持率及置信区间见 analysis.json；空值表示尚无相应人工证据。自动锚点重合率只诊断检索，不等于正确率。", "",
        "## 解释边界", "", "300道测试题是待专家审核的候选集，30道开发题按来源小节分开。已知训练问答审计不等于重建了全部训练来源，更不证明基座预训练未见教材。本次设备单次模型分块运行只能提供初步成本记录；正式效率结论应在安静设备重复并反转模型先后顺序。开发集结果不得写成独立测试集结果。"]
    (run / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Report saved: {run / 'report.md'}; human ratings = {len(joined)}")

"""Assemble transparent AI pre-scores without populating human review records."""
from __future__ import annotations
import csv
import hashlib
import html
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JOB = ROOT / "outputs/paper_eval/ai_prescore_teachercheck_v1_20261018"
REF = ROOT / "outputs/paper_eval/reference_suitability21_frozen_v1_20261017"
OUT = JOB / "delivery"


def jsonl(path):
    return [json.loads(s) for s in path.read_text(encoding="utf-8-sig").splitlines() if s.strip()]


def dump(path, value):
    path.write_bytes((json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))


def csv_out(name, records):
    with (OUT / name).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main():
    bundles = []
    grades = []
    for shard in (1, 2, 3):
        bundles.extend(json.loads((JOB / "inputs" / f"shard_{shard}.json").read_text(encoding="utf-8")))
        grades.extend(jsonl(JOB / "graded" / f"shard_{shard}.jsonl"))
    lookup = {g["item_id"]: g for g in grades}
    candidates = {c["item_id"]: (bundle, c) for bundle in bundles for c in bundle["candidates"]}
    if len(grades) != 160 or len(lookup) != 160 or set(lookup) != set(candidates):
        raise ValueError("AI quality scores must cover all 160 authentic candidate IDs exactly once")
    evidence_count = 0
    for item_id, grade in lookup.items():
        for field in ("ai_factual_correctness", "ai_essential_completeness"):
            if type(grade[field]) is not int or not 0 <= grade[field] <= 4:
                raise ValueError(f"Invalid AI score {item_id}/{field}")
        if not grade["ai_quality_reason_zh"].strip():
            raise ValueError(f"Missing content-based AI rationale: {item_id}")
        _, candidate = candidates[item_id]
        evidence = grade.get("ai_evidence")
        if bool(candidate.get("evidence_review")) != (evidence is not None):
            raise ValueError(f"Evidence coverage differs: {item_id}")
        if evidence is not None:
            evidence_count += 1
            if type(evidence["faithfulness"]) is not int or not 0 <= evidence["faithfulness"] <= 4:
                raise ValueError(f"Invalid evidence score: {item_id}")
            for field in ("citation_links_count", "supported_citation_links_count", "claims_requiring_citation", "claims_with_supported_citation"):
                if type(evidence[field]) is not int or evidence[field] < 0:
                    raise ValueError(f"Invalid AI citation count: {item_id}/{field}")
            if evidence["supported_citation_links_count"] > evidence["citation_links_count"] or evidence["claims_with_supported_citation"] > evidence["claims_requiring_citation"]:
                raise ValueError(f"Invalid AI count bounds: {item_id}")
            checks = evidence["claim_checks"]
            if (sum(len(set(c["cited_ids"])) for c in checks) != evidence["citation_links_count"] or
                    sum(len(set(c["supported_ids"])) for c in checks) != evidence["supported_citation_links_count"]):
                raise ValueError(f"Claim trace and AI count disagree: {item_id}")
    if evidence_count != 80:
        raise ValueError("All 80 actual-evidence answers require AI evidence scores")
    ref_sources = jsonl(REF / "reference_sources_21.jsonl")
    ref_grades = jsonl(JOB / "graded/reference_21.jsonl")
    ref_lookup = {r["id"]: r for r in ref_grades}
    if len(ref_grades) != 21 or set(ref_lookup) != {r["id"] for r in ref_sources}:
        raise ValueError("AI reference review must cover exactly the planned 21 questions")
    allowed = {"yes", "partial", "no", "uncertain"}
    for row in ref_grades:
        if any(row[k] not in allowed for k in ("ai_reference_suitable", "ai_evidence_sufficient", "ai_question_meaningful")):
            raise ValueError(f"Invalid reference category: {row['id']}")

    OUT.mkdir(exist_ok=True)
    quality_rows, evidence_rows, ref_rows = [], [], []
    priorities = {"high": "高", "medium": "中", "low": "低"}
    cards = []
    esc = lambda value: html.escape(str(value))
    for index, (item_id, (bundle, candidate)) in enumerate(candidates.items(), 1):
        g = lookup[item_id]
        evidence = g.get("ai_evidence")
        priority = g["ai_review_priority"]
        if (min(g["ai_factual_correctness"], g["ai_essential_completeness"]) <= 1 or
                g["ai_reference_problem_zh"] or g["ai_uncertainty_reason_zh"] or
                (evidence and evidence["faithfulness"] <= 1)):
            priority = "high"
        qrow = {"序号": index, "匿名ID": item_id, "复核优先级": priorities[priority],
            "AI事实正确性_0至4": g["ai_factual_correctness"], "AI要点完整性_0至4": g["ai_essential_completeness"],
            "教师事实正确性_0至4": "", "教师要点完整性_0至4": "", "教师姓名": "", "教师意见": "",
            "教师核验状态": "待核验", "题目": bundle["question"], "候选回答_已去引用标记": candidate["answer_for_quality"],
            "AI评分理由": g["ai_quality_reason_zh"], "公开参考疑问": g["ai_reference_problem_zh"], "AI不确定性说明": g["ai_uncertainty_reason_zh"]}
        quality_rows.append(qrow)
        paragraph = [f'<article id="item-{item_id}" data-priority="{priority}"><h2>{index}. {esc(bundle["question"])}</h2>',
            f'<p class="meta">匿名ID {item_id}　复核优先级：{priorities[priority]}　教师核验：待完成</p>',
            f'<div class="score">AI事实正确性 {g["ai_factual_correctness"]}/4　AI要点完整性 {g["ai_essential_completeness"]}/4</div>',
            f'<h3>候选回答（质量评分用，已去引用标记）</h3><pre>{esc(candidate["answer_for_quality"])}</pre>',
            f'<p><b>AI理由：</b>{esc(g["ai_quality_reason_zh"])}</p>']
        for label, key in (("公开参考疑问", "ai_reference_problem_zh"), ("不确定性", "ai_uncertainty_reason_zh")):
            if g[key]: paragraph.append(f'<p class="issue"><b>{label}：</b>{esc(g[key])}</p>')
        paragraph += [f'<details><summary>公开参考及质量审查支撑原文</summary><p><b>公开参考（未经独立验证）：</b>{esc(bundle["published_reference_unverified"])}</p><pre>{esc(bundle["supporting_passage_for_quality_review"])}</pre></details>']
        if evidence:
            e = candidate["evidence_review"]
            traces = "\n".join(f"{i}. {c['claim']}\n引用ID：{', '.join(c['cited_ids']) or '无'}\n完全支持ID：{', '.join(c['supported_ids']) or '无'}\n{c['support']}" for i, c in enumerate(evidence["claim_checks"], 1))
            evidence_rows.append({"序号": len(evidence_rows)+1, "匿名ID": item_id, "复核优先级": priorities[priority],
                "AI证据忠实性_0至4": evidence["faithfulness"], "教师证据忠实性_0至4": "",
                "AI引用关联总数": evidence["citation_links_count"], "AI受支持引用关联数": evidence["supported_citation_links_count"],
                "AI需引用陈述数": evidence["claims_requiring_citation"], "AI有支持引用陈述数": evidence["claims_with_supported_citation"],
                "教师引用关联总数": "", "教师受支持引用关联数": "", "教师需引用陈述数": "", "教师有支持引用陈述数": "",
                "教师姓名": "", "教师意见": "", "教师核验状态": "待核验", "题目": bundle["question"],
                "原始回答_保留引用": e["answer_with_original_citations"], "AI证据评分理由": evidence["reason_zh"], "AI逐陈述核查": traces})
            paragraph += [f'<h3>AI证据忠实性 {evidence["faithfulness"]}/4</h3><p>{esc(evidence["reason_zh"])}</p>',
                f'<p>引用关联支持：{evidence["supported_citation_links_count"]}/{evidence["citation_links_count"]}；有支持引用陈述：{evidence["claims_with_supported_citation"]}/{evidence["claims_requiring_citation"]}。分母0时比率不适用。</p>',
                f'<details><summary>原始引用与逐陈述核查</summary><pre>{esc(e["answer_with_original_citations"])}</pre><pre>{esc(traces)}</pre></details>',
                '<details><summary>模型实际收到的资料（证据评分唯一来源）</summary>']
            for source in e["actual_evidence_provided"]:
                paragraph += [f'<h4>{esc(source["id"])}</h4><pre>{esc(source["text"])}</pre>']
            paragraph += ['</details>']
        paragraph += ['</article>']
        cards.append("\n".join(paragraph))
    cat = {"yes": "是", "partial": "部分", "no": "否", "uncertain": "待定"}
    ref_cards = []
    for index, source in enumerate(ref_sources, 1):
        g = ref_lookup[source["id"]]
        ref_rows.append({"序号": index, "题目ID": source["id"], "复核优先级": priorities[g["ai_review_priority"]],
            "AI参考适用": cat[g["ai_reference_suitable"]], "AI证据充分": cat[g["ai_evidence_sufficient"]],
            "AI问题有意义": cat[g["ai_question_meaningful"]], "教师参考适用": "", "教师证据充分": "", "教师问题有意义": "",
            "教师姓名": "", "教师意见": "", "教师核验状态": "待核验", "题目": source["question"],
            "公开原始参考": source["reference_answer_origin"], "原始answer_in_text": "缺失" if source["published_answer_in_text"] is None else source["published_answer_in_text"],
            "AI参考复核理由": g["ai_reason_zh"]})
        body = [f'<article id="ref-{source["id"]}" data-priority="{g["ai_review_priority"]}"><h2>参考复核 {index}. {esc(source["question"])}</h2>',
            f'<p>{esc(source["id"])}　AI参考适用：{cat[g["ai_reference_suitable"]]}　证据充分：{cat[g["ai_evidence_sufficient"]]}　问题有意义：{cat[g["ai_question_meaningful"]]}</p>',
            f'<p><b>公开原始参考：</b>{esc(source["reference_answer_origin"])}</p><p><b>AI理由：</b>{esc(g["ai_reason_zh"])}</p><details><summary>已发布支撑原文</summary>']
        for window in source["released_source_windows"]:
            body += [f'<h4>{esc(window["id"])}</h4><pre>{esc(window["text"])}</pre>']
        ref_cards.append("\n".join(body + ['</details></article>']))
    csv_out("AI质量预评分160条.csv", quality_rows)
    csv_out("AI证据预评分80条.csv", evidence_rows)
    csv_out("AI参考适用性复核21题.csv", ref_rows)
    dump(OUT / "workbook_data.json", {"quality": quality_rows, "evidence": evidence_rows, "reference": ref_rows})
    notes = """# AI预评分与教师核验

已填写160条质量、80条证据和21题参考适用性建议。这是一套AI预评分，采用同一流程分片阅读、内部校对。它不是两位独立人类专家评分，当前教师核验仍为零。请在黄色教师列确认或修改，填写姓名与意见。

质量：分别评价题目所求事实和关键要点。0为错误或无实质回答，1主要错误／严重缺失，2部分，3大体正确，4该维度充分正确。纯拒答未给出所求事实时，任务质量两维为0；若实际资料确实不足，证据忠实性仍可高分。公开参考不是绝对真值，疑问已单列。

证据：只按模型实际收到的资料评价，不用质量审查的Gold原文补证据。忠实性0无支持／冲突、1多数不支持、2混合、3多数支持、4充分支持。编号存在不代表陈述受支持。

计数：按最小独立可核实陈述与所附来源ID形成“陈述—引用”关联，同一陈述同ID去重。段末唯一引文覆盖同段此前未单独引用陈述；已有明确引用的句子按各自来源。只有完全受该源支持才计入受支持数，部分支持留给老师核验。无分母不计算比率。关于“所供资料没有X”的可核实说明也是资料陈述；纯“I cannot answer”没有事实陈述。

复核优先级“高”包括任一质量分≤1、参考或事实不确定性、证据忠实性≤1；其他优先级沿用AI逐项判断。优先级用于安排检查，不等于评分已可靠。所有条目仍待教师核验。

Excel用于填写分数与意见；核查材料.html可搜索匿名ID并展开完整参考、Gold支撑、模型实际资料和逐陈述核查。原始空白双人盲评包、原始公开标签和正式自动指标均保留。本版没有把AI分导入正式人工成绩，也没有计算或声称双人一致性。若用于论文，应如实说明AI辅助预评分及实际教师核验方式；只有独立完成的真实人类评分才能作为相应人评结果。
"""
    (OUT / "填写说明.md").write_bytes(notes.encode("utf-8"))
    document = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>AI预评分待教师核验</title>
<style>body{font:16px/1.7 Arial,"Microsoft YaHei",sans-serif;max-width:1080px;margin:28px auto;padding:0 20px;color:#243447;background:#f5f7fb}header,article{background:white;border:1px solid #dce2eb;border-radius:8px;padding:24px;margin:20px 0}h1{font-size:26px}h2{font-size:20px}.meta{font-size:13px;color:#596676}.score{padding:12px;background:#edf1f8;font-weight:bold}.issue{background:#fff2d5;padding:10px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:inherit;background:#f6f8fa;padding:14px}details{margin:14px 0}summary{cursor:pointer;font-weight:bold}nav{position:sticky;top:0;background:#f5f7fb;padding:12px 0;display:flex;gap:12px}input,select{font:inherit;padding:6px}input{flex:1}</style>
<header><h1>AI预评分，待教师核验</h1><p>160条回答质量、80条证据、21题参考适用性。AI建议已填，真实教师核验尚未完成。</p><p>Excel黄色列供教师确认或修改。此页可搜索匿名ID、题干与理由，展开原文检查。质量评分与证据评分分别判断；资料不足的拒答可能忠实，但不等于完成了题目。</p><p>来源：Pirá 2.0，CC BY 4.0。仅使用已发布支撑文本，保留底层原文适用权利。<a href="https://huggingface.co/datasets/paulopirozelli/pira/tree/756f73c8a52e2b0fbdc8f42569e1e8c6f3865a49">固定数据版本</a>。完整计数与评分规则见填写说明。</p></header>
<nav><input id="search" placeholder="搜索匿名ID、题干或理由"><select id="priority"><option value="all">全部优先级</option><option value="high">高优先级</option><option value="medium">中优先级</option><option value="low">低优先级</option></select></nav>
''' + "\n".join(cards) + '<h1>另外21题：参考适用性</h1>' + "\n".join(ref_cards) + '''<script>
const items=[...document.querySelectorAll('article')];function filter(){let q=document.querySelector('#search').value.toLowerCase(),p=document.querySelector('#priority').value;items.forEach(x=>x.hidden=!(x.textContent.toLowerCase().includes(q)&&(p==='all'||x.dataset.priority===p)))}document.querySelector('#search').addEventListener('input',filter);document.querySelector('#priority').addEventListener('change',filter);
</script></html>'''
    (OUT / "核查材料.html").write_bytes(document.encode("utf-8"))
    manifest = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "scoring_origin": "AI pre-score, teacher verification pending",
        "ai_grading_workflows": 1, "parallel_chunks": 3, "independent_human_raters": 0,
        "quality_scores": 160, "evidence_scores": 80, "reference_reviews": 21, "teacher_fields": "blank",
        "formal_human_scores_modified": False, "frozen_evaluation_modified": False,
        "source_hashes": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted((JOB / "graded").glob("*.jsonl"))},
        "priority_rule": "High for quality<=1, flagged reference/uncertainty, or faithfulness<=1; otherwise AI's item priority",
        "ai_internal_consistency_check": "AI check only; not an independent human rating"}
    dump(JOB / "ai_prescore_manifest.json", manifest)
    print(f"Assembled 160 quality + 80 evidence + 21 reference pre-scores: {OUT}")


if __name__ == "__main__":
    main()

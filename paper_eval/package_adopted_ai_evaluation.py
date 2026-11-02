"""Deliver the adopted AI ratings without creating human rating records."""
from __future__ import annotations
import csv
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/paper_eval/ai_evaluation_adopted_20261018"
OLD = ROOT / "outputs/paper_eval/ai_prescore_teachercheck_v1_20261018"


def save_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    analysis = json.loads((OUT / "analysis.json").read_text(encoding="utf-8"))
    scores = json.loads((OUT / "per_answer_ai_scores.json").read_text(encoding="utf-8"))
    lookup = {r["item_id"]: r for r in scores}
    names = {"base": "基座", "base_rag": "基座＋RAG", "lora": "微调", "lora_rag": "微调＋RAG"}
    summary = []
    for condition, values in analysis["groupings"]["source_group"]["by_condition"].items():
        row = {"条件": condition, "名称": names[condition], "题数": 40, "评价来源": "AI"}
        for metric in ("factual_quality", "completeness", "faithfulness", "citation_support", "citation_coverage"):
            m = values[metric]
            row[f"{metric}_mean"] = m["mean"]
            row[f"{metric}_ci_low"] = m["ci95"][0] if m["ci95"] else None
            row[f"{metric}_ci_high"] = m["ci95"][1] if m["ci95"] else None
            row[f"{metric}_numerator"] = m["numerator"] if m["denominator"] else None
            row[f"{metric}_denominator"] = m["denominator"] if m["denominator"] else None
        summary.append(row)
    save_csv(OUT / "AI评分分组汇总.csv", summary)
    paired = []
    for comparison, metrics in analysis["groupings"]["source_group"]["paired_differences"].items():
        for metric, values in metrics.items():
            paired.append({"比较": comparison, "AI指标": metric, "均值差": values["mean"],
                           "95%区间下限": values["ci95"][0], "95%区间上限": values["ci95"][1],
                           "配对题数": values["paired_questions"], "评价来源": "AI"})
    save_csv(OUT / "AI评分配对差异.csv", paired)
    table_files = (("AI质量预评分160条.csv", "AI质量评分160条.csv"),
                   ("AI证据预评分80条.csv", "AI证据评分80条.csv"),
                   ("AI参考适用性复核21题.csv", "AI参考适用性复核21题.csv"))
    for source_name, target_name in table_files:
        with (OLD / "delivery" / source_name).open(encoding="utf-8-sig", newline="") as stream:
            original = list(csv.DictReader(stream))
        records = []
        for row in original:
            prepared = {key: value for key, value in row.items() if not key.startswith("教师")}
            prepared["评价来源"] = "AI"
            prepared["采用状态"] = "已采用AI评分"
            if "匿名ID" in row:
                link = lookup[row["匿名ID"]]
                prepared["条件"] = link["condition"]
                prepared["原始问题ID"] = link["id"]
            records.append(prepared)
        save_csv(OUT / target_name, records)
    html = (OLD / "delivery/核查材料.html").read_text(encoding="utf-8")
    replacements = {
        "AI预评分待教师核验": "已采用AI评价",
        "AI预评分，待教师核验": "AI评价记录",
        "AI建议已填，真实教师核验尚未完成。": "已采用AI建议分；未开展独立人类逐项评分。",
        "Excel黄色列供教师确认或修改。此页可搜索匿名ID、题干与理由，展开原文检查。": "可搜索匿名ID、题干与理由，展开原文检查。评分来源为AI。",
        "教师核验：待完成": "评价来源：AI；独立人评未开展",
        "完整计数与评分规则见填写说明。": "完整计数与评分规则见使用说明。",
    }
    for before, after in replacements.items():
        html = html.replace(before, after)
    (OUT / "原文与逐项依据.html").write_bytes(html.encode("utf-8"))
    shutil.copyfile(OLD / "AI内部校对记录.md", OUT / "AI内部校对记录.md")
    text = """# AI评价采用版

按用户转述的老师意见，直接采用已归档的AI建议分。本包无需继续填写旧版空白人评表。

本包包含40题四条件共160条质量AI评分、80条证据AI评分，以及另外21题的参考适用性AI复核。论文称AI-based evaluation或AI-judged answer quality；本包没有独立人类逐题评分，没有双人一致性或专家盲评结果。

先看report.md的分组结果，再看AI_evaluation_methods_and_results.md的英文写法。CSV保存精确均值、区间、计数和逐条理由；比例为0到1，非0到100。非RAG证据指标留空表示不适用，不是0分。原文与逐项依据.html支持搜索ID和展开资料。

当前AI样本支持RAG提高答题质量；微调在已有RAG之上的额外事实、完整性、忠实性和引用支持差值区间均跨0。不能写成微调在所有质量维度显著提高。

所有AI原始建议分保持不变，正式227题自动结果和公开参考保持不变。AI对21题参考的判断不是对官方标签的替换。采用AI评分与老师已逐题核验是不同事件；当前仅有用户转述的采用授权。精确AI评审模型版本和解码参数未独立记录，该限制已写入英文方法。

引用计数使用陈述—引用关联，段末唯一引用覆盖此前同段未另引证的陈述；仅完全受所引资料支持才计支持。AI可识别[Citation: ID]等语义可读样式，与正式自动编号解析器的格式指标分开。完整规则和细节见逐项依据与内部校对记录。
"""
    (OUT / "使用说明.md").write_bytes(text.encode("utf-8"))
    deliverables = ["report.md", "AI_evaluation_methods_and_results.md", "AI评分分组汇总.csv", "AI评分配对差异.csv",
        "AI质量评分160条.csv", "AI证据评分80条.csv", "AI参考适用性复核21题.csv", "原文与逐项依据.html",
        "AI内部校对记录.md", "使用说明.md", "analysis.json", "per_answer_ai_scores.json", "per_answer_ai_scores.csv"]
    archive = OUT / "AI评价采用版_20261018.zip"
    if archive.exists():
        raise FileExistsError(archive)
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as stream:
        for name in deliverables:
            stream.write(OUT / name, name)
    manifest = {"status": "adopted_as_AI_evaluation", "adoption_basis": analysis["adoption_basis"],
        "independent_human_raters": 0, "AI_scores_changed": False,
        "files_sha256": {name: hashlib.sha256((OUT / name).read_bytes()).hexdigest() for name in deliverables},
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "analysis_script_sha256": hashlib.sha256((ROOT / "paper_eval/adopt_ai_evaluation.py").read_bytes()).hexdigest(),
        "packaging_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT / "delivery_manifest.json").write_bytes(json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"))
    print(f"AI evaluation adopted: {archive} ({archive.stat().st_size} bytes)")


if __name__ == "__main__":
    main()

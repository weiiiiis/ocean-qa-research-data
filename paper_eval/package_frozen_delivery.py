"""Package completed frozen reports and blank, separately blinded review files.

This is a delivery helper only. It does not generate answers, change scores,
modify frozen inputs, or assign human ratings.
"""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "outputs/paper_eval"
MC = EVAL / "en_test_mcqa_frozen_v1_report_20261017"
OPEN = EVAL / "en_test_openqa_frozen_v1_report_20261017"
REVIEW = EVAL / "en_test_blind40_frozen_v1_20261017"
REF = EVAL / "reference_suitability21_frozen_v1_20261017"
OUT = EVAL / "teacher_review_packets_frozen_v1_20261017"
GUIDE = ROOT / "paper_data/教师盲评填写说明.md"
SUMMARY = ROOT / "paper_data/正式实验结果汇总.md"
NAMES = {"base": "基座", "base_rag": "基座＋RAG", "lora": "新微调", "lora_rag": "新微调＋RAG"}


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def number(value: float | None, percentage: bool = True) -> str:
    return "—" if value is None else (f"{100 * value:.1f}%" if percentage else f"{value:.1f}")


def interval(metric: dict) -> str:
    ci = metric.get("ci95")
    return number(metric.get("mean")) + (f" [{100 * ci[0]:.1f}, {100 * ci[1]:.1f}]" if ci else "")


def validate_blank_csv(path: Path, expected: int, fields: tuple[str, ...]) -> None:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        records = list(csv.DictReader(stream))
    if len(records) != expected or any(row[field].strip() for row in records for field in fields):
        raise ValueError(f"Expected {expected} authentic rows with blank expert scores: {path}")


def main() -> None:
    if OUT.exists() or SUMMARY.exists():
        raise FileExistsError("Delivery output already exists; preserve all human work and summaries")
    mc, op, review = load(MC / "analysis.json"), load(OPEN / "analysis.json"), load(REVIEW / "review_export_manifest.json")
    for analysis in (mc, op):
        if not analysis.get("complete") or analysis.get("completed_answers") != 908:
            raise ValueError("Both official tasks must be fully complete before delivery")
        if analysis.get("requested_conditions") != list(NAMES):
            raise ValueError("The delivery requires the fixed four-condition comparison")
    if not review.get("subset_complete") or review.get("raw_answers_exported") != 160:
        raise ValueError("The fixed 40-question review export is incomplete")
    quality_fields = ("factual_correctness_0_to_4", "essential_completeness_0_to_4")
    evidence_fields = ("evidence_faithfulness_0_to_4", "citation_links_count", "supported_citation_links_count",
                       "claims_requiring_citation", "claims_with_supported_citation")
    shared = REVIEW / "expert_review"
    for reviewer in (1, 2):
        validate_blank_csv(shared / f"reviewer_{reviewer}_quality.csv", 160, quality_fields)
        validate_blank_csv(shared / f"reviewer_{reviewer}_evidence.csv", 80, evidence_fields)
    validate_blank_csv(REF / "reference_suitability_review_21.csv", 21,
                       ("human_reference_suitable", "human_evidence_sufficient", "human_question_meaningful"))
    mg = mc["slices"]["all_227"]["groupings"]["source_group"]
    og = op["slices"]["all_227"]["groupings"]["source_group"]
    lines = ["# 正式英文海洋问答实验结果", "",
             "实验设置于2026-10-17在正式生成前冻结，模型仅按开发集选择。Pirá官方test共227个原始问题；选择题与开放问答每项任务均完成227题×4组=908条回答，两项任务共1816条输出，来自227个原始问题，不能写成454道独立测试题。", "",
             "## 已完成的修复", "",
             "1. 将正式检索设置固定为BM25，使用688段去重的已发布支撑文本所形成的1507个窗口，top-3、上下文1000 token。", 
             "2. 修订英文提示，限定证据范围和真实资料编号；从基座新训英文LoRA，并做一轮条件引用续训。训练为240个不同示例、48个事实家族，另60例／12家族验证；数据来自Webb教材，未用Pirá问答或支撑文本作SFT。", 
             "3. 保存完整开发筛选、数据与代码哈希，冻结后完成四组正式评测；原始公开标签、模型回答及失败样本均保留。", "",
             "最终适配器：`outputs/ocean_qwen3_english_grounded_v2/epoch_1`。它由英文v1的epoch_1继续训练；v1的epoch_2不是其祖先。原中文应用模型尚未切换。", "",
             "## 正式选择题：同227题", "",
             "选项解析规则在正式输出前固定，四组一致。格式合规率与答案正确率分开。", "",
             "| 条件 | 选项准确率 [95% CI] | 严格单字母准确率 | 格式合规率 |",
             "|---|---:|---:|---:|"]
    for condition, name in NAMES.items():
        metrics = mg["by_condition"][condition]["metrics"]
        lines.append(f"| {name} | {interval(metrics['mc_prefix_sensitivity_accuracy'])} | {number(metrics['accuracy']['mean'])} | {number(1-metrics['mc_format_invalid']['mean'])} |")
    lines += ["", "RAG相对无RAG提高选项正确率：基座+6.6个百分点[2.7,10.8]、微调+10.6个百分点[6.7,14.7]。微调＋RAG相对基座＋RAG为+1.8个百分点[-0.4,4.1]，尚不足以证明额外正确率增益。在选择题中，微调主要可确认的是输出格式改善。225题及206题敏感性没有改变此判断。", "",
              "## 正式开放问答：同227题", "",
              "F1和EM只度量与原始公开参考的文字重合，均不等于事实正确性。此处使用正式冻结的去引用指标，并保留原始F1。引用编号可解析也不等于原文支持陈述。", "",
              "| 条件 | 去引用F1 [95% CI] | 去引用EM | 原始F1 | 有引用标记回答 | 可解析标记／全部标记 | 截断 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for condition, name in NAMES.items():
        data = og["by_condition"][condition]
        metrics, counts = data["metrics"], data["counts"]
        cited = int(metrics["answer_has_citation"]["numerator_total"])
        truncated = int(metrics["hit_token_limit"]["numerator_total"])
        lines.append(f"| {name} | {interval(metrics['citation_stripped_token_f1'])} | {number(metrics['citation_stripped_exact_match']['mean'])} | {number(metrics['token_f1']['mean'])} | {cited}/227 | {counts['resolved_citation_markers']}/{counts['citation_markers']} | {truncated}/227 |")
    no_rag_cited = int(og["by_condition"]["lora"]["metrics"]["answer_has_citation"]["numerator_total"])
    rag_counts = og["by_condition"]["lora_rag"]["counts"]
    unresolved = rag_counts["citation_markers"] - rag_counts["resolved_citation_markers"]
    lines += ["", f"引用行为尚未完全修复：新微调无RAG时仍有{no_rag_cited}/227条回答带无法解析的引用标记；有RAG时仍有{unresolved}个标记无法解析。引用出现更多不能直接解释为引用支持更准确，真实证据支持仍须人工判断。", "",
              "### 去引用F1的配对差值", "", "| 比较 | 差值及95%区间（百分点） |", "|---|---:|"]
    for pair, comparison in og["paired_differences"].items():
        left, right = pair.split("_minus_")
        lines.append(f"| {NAMES[left]} − {NAMES[right]} | {interval(comparison['metrics']['citation_stripped_token_f1'])} |")
    lines += ["", "主分析按181个来源组配对bootstrap，2000次、seed=20261017。完整报告另列保守144组、排除两条历史复核标记后225题、answer_in_text=1的206题，以及其余21题诊断。差值表的%表示百分点；差值用未四舍五入均值计算，因此可能与表中已取一位小数的均值相减相差0.1。", "",
              "## 检索与实际回答成本", "",
              "正式BM25 Gold来源召回为202/227（89.0%），原dense补充对照为49.8%。这是受限支撑文本语料上的来源检索指标，不是回答准确率。两个RAG条件使用相同实际提示和上下文。", "",
              "下表仅为开放问答的自然回答实测。输出长度和输入长度影响耗时，没有固定工作量复测，因此不据此声称推理效率提高。显存指PyTorch分配器峰值，不是整卡占用。", "",
              "| 条件 | 输出token均值 | 总耗时p50／p95（ms） | 首token p50（ms） | 解码token/s均值 | 最大峰值MiB |",
              "|---|---:|---:|---:|---:|---:|"]
    for condition, name in NAMES.items():
        cost = op["natural_answer_cost_descriptive"][condition]
        lines.append(f"| {name} | {number(cost['output_tokens_including_eos']['mean'],False)} | {number(cost['pipeline_ms']['p50'],False)}／{number(cost['pipeline_ms']['p95'],False)} | {number(cost['pipeline_ttft_ms']['p50'],False)} | {number(cost['decode_tokens_per_second']['mean'],False)} | {number(cost['peak_allocated_mib']['max'],False)} |")
    lines += ["", "## 真实人工评价仍待完成", "",
              "已导出正式生成前固定抽取的40题×4组=160条真实回答，两位评审各有160行质量和80行证据材料。所有人工分保持空白；另21题附参考与支撑原文，单独复核参考适用性。", "",
              f"交给老师的材料目录：`{OUT.relative_to(ROOT).as_posix()}`，分别发放reviewer_1.zip、reviewer_2.zip；reference_review_21.zip用于参考复核。不含私有模型映射。请按《教师盲评填写说明》保留独立初评。", "",
              "## 写论文前的边界", "",
              "- 可以报告真实正式基准、来源组区间、格式和检索改善；不能把F1、引用编号合法率写成人工事实准确率。", 
              "- 新训练仅48个事实家族，验证与训练来自同一教材；未知预训练污染仍不能排除。公开来源/EID分组也不能证明整篇文档完全独立。", 
              "- 该实验未评价中文教学效果、学生学习提升或完整知识图谱模块。", 
              "- 投稿前还需真实评审打分、分析一致性，完成英文论文和导师审核；目前没有实际投稿或期刊录用。", "",
              "## 文件索引", "",
              "- 正式选择题：`outputs/paper_eval/en_test_mcqa_frozen_v1_report_20261017/report.md`。", 
              "- 正式开放问答：`outputs/paper_eval/en_test_openqa_frozen_v1_report_20261017/report.md`。", 
              "- 正式开放问答原始答案：`outputs/paper_eval/en_test_openqa_frozen_v1_20261017/answers.jsonl`。", 
              "- 模型与训练来源：`outputs/ocean_qwen3_english_grounded_v2/MODEL_CARD.md`。", 
              "- 冻结清单：`paper_data/frozen_english_v1_20261017/freeze_manifest.json`。", ""]
    summary_text = "\n".join(lines) + "\n"
    # Resolve all report fields before creating any delivery output.
    GUIDE.read_text(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=False)
    for reviewer in (1, 2):
        with zipfile.ZipFile(OUT / f"reviewer_{reviewer}.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
            for name in (f"reviewer_{reviewer}_quality.csv", f"reviewer_{reviewer}_evidence.csv", "README.md"):
                archive.write(shared / name, name)
            archive.write(GUIDE, GUIDE.name)
    with zipfile.ZipFile(OUT / "reference_review_21.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in ("reference_suitability_review_21.csv", "支撑原文与填写说明.md", "reference_sources_21.jsonl"):
            archive.write(REF / name, name)
    shutil.copyfile(GUIDE, OUT / GUIDE.name)
    manifest = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
                "expert_scores": "blank/pending", "questions_in_review_packets": 40,
                "quality_rows_per_reviewer": 160, "evidence_rows_per_reviewer": 80,
                "separate_reference_review_questions": 21,
                "private_mapping_included": False, "models_or_raw_run_manifests_included": False,
                "summary_sha256": hashlib.sha256(summary_text.encode("utf-8")).hexdigest(),
                "delivery_script_sha256": digest(Path(__file__)),
                "inputs_sha256": {str(p.relative_to(ROOT)): digest(p) for p in
                                  (MC / "analysis.json", OPEN / "analysis.json",
                                   REVIEW / "review_export_manifest.json", REF / "manifest.json", GUIDE)},
                "files_sha256": {p.name: digest(p) for p in OUT.iterdir() if p.is_file()}}
    (OUT / "package_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    SUMMARY.write_bytes(summary_text.encode("utf-8"))
    print(f"Completed reports summarized: {SUMMARY}")
    print(f"Separate blank review packets: {OUT}")


if __name__ == "__main__":
    main()

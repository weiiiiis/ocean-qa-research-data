"""Prepare source-separated English development artifacts without training.

Only official validation questions selected before looking at model outputs are
adapted here. The official test is read solely to assert separation; no test
answers are generated and no Pira QA is converted into an SFT record.
"""
from __future__ import annotations

import collections
import csv
import hashlib
import json
import math
import re
import urllib.request
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REVISION = "756f73c8a52e2b0fbdc8f42569e1e8c6f3865a49"
BASE_URL = f"https://huggingface.co/datasets/paulopirozelli/pira/resolve/{REVISION}/"
MCQA_PATH = "mcqa/validation-00000-of-00001-42647ea31fbd61e7.parquet"
MCQA_SHA256 = "eaf5dc197253785df1293fa8a6ca18c2243bf1ee9dd33f234873e68a28320244"
LICENSE = "CC BY 4.0 (Pira dataset); preserve original supporting-text source rights"
ASSESSMENT_FIELDS = ("answer_in_text", "at_labels", "question_meaningful", "answer_equivalent", "answer_difficulty", "question_generic")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows), encoding="utf-8")


def normalized(text: str) -> str:
    return re.sub(r"[^\w]", "", text.casefold())


def clean(value):
    if value is None or isinstance(value, float) and math.isnan(value):
        return None
    return value


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def file_record(path: Path) -> dict:
    return {"file": path.relative_to(HERE).as_posix(), "bytes": path.stat().st_size, "sha256": digest(path)}


def main() -> None:
    mcqa_file = HERE / "raw/pira" / MCQA_PATH
    if not mcqa_file.exists():
        mcqa_file.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(BASE_URL + MCQA_PATH, mcqa_file)
    if mcqa_file.stat().st_size != 268995 or digest(mcqa_file) != MCQA_SHA256:
        raise ValueError("Pinned MCQA validation file has unexpected byte length or SHA256")

    validation_file = next((HERE / "raw/pira/data").glob("validation-*.parquet"))
    test_file = next((HERE / "raw/pira/data").glob("test-*.parquet"))
    validation = pd.read_parquet(validation_file).to_dict("records")
    test = pd.read_parquet(test_file).to_dict("records")
    mcqa = pd.read_parquet(mcqa_file).to_dict("records")
    raw_by_id = {row["id_qa"]: row for row in validation}
    mcqa_by_id = {row["id"]: row for row in mcqa}
    if len(mcqa_by_id) != len(mcqa):
        raise ValueError("Duplicate official MCQA validation IDs")
    selected_file = HERE / "pira_dev_article_disjoint_en_101.jsonl"
    selected = read_jsonl(selected_file)
    passages_file = HERE / "pira_supporting_passages.jsonl"
    passages = read_jsonl(passages_file)
    if len(selected) != 101 or len(passages) != 688:
        raise ValueError("Unexpected development pool or supporting-passage size")
    passage_by_text = {normalized(row["text"]): row for row in passages}
    if len(passage_by_text) != len(passages):
        raise ValueError("Supporting passages are not normalized-text unique")

    test_ids = {row["id_qa"] for row in test}
    test_eids = {str(row["eid_article_scopus"]) for row in test if clean(row["eid_article_scopus"]) is not None}
    test_texts = {normalized(row["abstract"]) for row in test}
    dev_ids = {row["original_id"] for row in selected}
    dev_eids = {str(row["source_article_eid"]) for row in selected}
    dev_texts = {normalized(row["supporting_text"]) for row in selected}
    separation = {
        "question_id_overlap": sorted(dev_ids & test_ids),
        "known_article_eid_overlap": sorted(dev_eids & test_eids),
        "normalized_supporting_text_overlap_count": len(dev_texts & test_texts),
    }
    if any(separation.values()):
        raise ValueError(f"Development/test separation failed: {separation}")
    if len(dev_ids) != len(selected):
        raise ValueError("Duplicate selected development question ID")

    openqa, multiple_choice, review_rows = [], [], []
    question_differences = []
    for selected_row in selected:
        original_id = selected_row["original_id"]
        raw = raw_by_id[original_id]
        choice = mcqa_by_id[original_id]
        if raw["question_en_origin"] != selected_row["question"] or normalized(raw["abstract"]) != normalized(selected_row["supporting_text"]):
            raise ValueError(f"Selected record no longer matches official validation: {original_id}")
        if normalized(choice["text"]) != normalized(raw["abstract"]):
            raise ValueError(f"MCQA source differs from open-QA source: {original_id}")
        passage_id = passage_by_text[normalized(raw["abstract"])]["id"]
        origin = raw["answer_en_origin"]
        alternate = clean(raw["answer_en_validate"])
        references = list(dict.fromkeys(answer for answer in [origin, alternate] if isinstance(answer, str) and answer.strip()))
        if not references or references[0] != origin:
            raise ValueError(f"Missing primary original reference: {original_id}")
        assessments = {key: clean(raw[key]) for key in ASSESSMENT_FIELDS}
        common = {
            "original_id": original_id, "source_group": "PIRA-EID-" + str(raw["eid_article_scopus"]),
            "source_article_eid": str(raw["eid_article_scopus"]), "expected_source_id": passage_id,
            "mode": "professional", "language": "en", "split": "dev", "original_split": "validation",
            "original_assessments": assessments,
            "review_status": "Official labels retained; local expert factual, suitability and reference-equivalence review pending",
        }
        openqa.append({
            **common, "id": "PIRA-DEV-" + original_id, "task": "open_qa", "question": raw["question_en_origin"],
            "reference_answer_origin": origin, "reference_answer_validate": alternate,
            "reference_answers": references,
            "reference_policy": "Primary automatic scores use original answer only; alternate reference requires independent equivalence review before sensitivity analysis",
        })
        label = str(choice["alternative"]).strip().upper()
        choices = {letter: choice[letter] for letter in "ABCDE"}
        if label not in choices or choices[label].strip() != choice["correct"].strip():
            raise ValueError(f"MCQA official label/text mismatch: {original_id}")
        multiple_choice.append({
            **common, "id": "PIRA-DEV-MC-" + original_id, "task": "mcqa", "question": choice["question"],
            "choices": choices, "answer": label, "reference_answer_text": choice["correct"],
            "reference_answer_origin": origin, "reference_answers": references,
        })
        if choice["question"] != raw["question_en_origin"]:
            question_differences.append(original_id)
        text_label = assessments["answer_in_text"]
        label_note = "answer_in_text=0: support needs manual inspection" if text_label == 0 else "answer_in_text missing: support not certified" if text_label is None else "answer_in_text=1: published annotation, not local expert certification"
        contextual = bool(re.search(r"\b(this|these|their|the study|the paper|northeast|in the region)\b", raw["question_en_origin"], re.I))
        review_rows.append({
            "id": "PIRA-DEV-" + original_id, "original_id": original_id, "source_article_eid": raw["eid_article_scopus"],
            "expected_source_id": passage_id, "question_openqa": raw["question_en_origin"], "question_mcqa": choice["question"],
            "reference_answer_origin": origin, "reference_answer_validate": alternate or "", "supporting_text": raw["abstract"],
            "answer_in_text_original": "" if text_label is None else text_label,
            "at_labels_original": "" if assessments["at_labels"] is None else assessments["at_labels"],
            "answer_equivalent_original": "" if assessments["answer_equivalent"] is None else assessments["answer_equivalent"],
            "automatic_label_note": label_note,
            "automatic_context_word_flag": "context-dependent wording candidate; heuristic only" if contextual else "",
            "expert_supports_original_answer": "", "expert_alternate_equivalent": "", "expert_question_self_contained": "",
            "expert_keep_for_primary_openqa": "", "expert_corrected_reference_if_any": "", "reviewer": "", "review_notes": "",
        })

    # Corpus fields are deliberately restricted. Neither questions, answers,
    # alternatives nor human annotations may enter the retrieval index.
    chunks = []
    for passage in passages:
        text = passage["text"]
        for i, start in enumerate(range(0, len(text), 850), 1):
            window = text[start:start + 1100]
            if not window.strip():
                continue
            chunks.append({
                "id": passage["id"] + f"-C{i:03d}", "text": window,
                "source": "Pira 2.0 supplied supporting passage" + ("; Scopus EID " + str(passage["article_eid"]) if passage.get("article_eid") else ""),
                "pdf_page": None, "section": passage["id"], "license": LICENSE,
                "url": "https://huggingface.co/datasets/paulopirozelli/pira/tree/" + REVISION,
            })
            # Avoid adding a suffix window which repeats already-covered text.
            if start + 1100 >= len(text):
                break
    allowed = {"id", "text", "source", "pdf_page", "section", "license", "url"}
    if any(set(row) != allowed for row in chunks) or len({row["id"] for row in chunks}) != len(chunks):
        raise ValueError("Corpus schema or chunk ID uniqueness failed")
    if {row["section"] for row in chunks} != {row["id"] for row in passages}:
        raise ValueError("Not all 688 original passages are represented")
    if not {row["expected_source_id"] for row in openqa} <= {row["section"] for row in chunks}:
        raise ValueError("Missing expected source in corpus")

    output_files = []
    for filename, rows in [("dev_openqa_en.jsonl", openqa), ("dev_mcqa_en.jsonl", multiple_choice), ("corpus_pira_chunks_en.jsonl", chunks)]:
        path = HERE / filename
        write_jsonl(path, rows)
        output_files.append(path)
    review_file = HERE / "dev_reference_expert_review.csv"
    with review_file.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(review_rows[0]))
        writer.writeheader()
        writer.writerows(review_rows)
    output_files.append(review_file)
    uncertain = [row for row in review_rows if row["answer_in_text_original"] in (0, "")]
    focus_file = HERE / "dev_reference_support_uncertain.csv"
    with focus_file.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(review_rows[0]))
        writer.writeheader()
        writer.writerows(uncertain)
    output_files.append(focus_file)

    manifest = {
        "stage": "English development preparation only; no model score or SFT artifact",
        "dataset": "Pira 2.0", "revision": REVISION, "license": LICENSE,
        "download": {**file_record(mcqa_file), "url": BASE_URL + MCQA_PATH},
        "inputs": [file_record(path) for path in [selected_file, validation_file, test_file, passages_file]],
        "outputs": [file_record(path) for path in output_files],
        "counts": {"dev_openqa": len(openqa), "dev_mcqa": len(multiple_choice), "dev_unique_article_eids": len(dev_eids),
            "dev_unique_normalized_passages": len(dev_texts), "corpus_source_passages": len(passages), "corpus_chunks": len(chunks)},
        "test_separation_checks": separation,
        "answer_in_text": dict(collections.Counter("missing" if row["original_assessments"]["answer_in_text"] is None else str(int(row["original_assessments"]["answer_in_text"])) for row in openqa)),
        "answer_in_text_zero_ids": [row["original_id"] for row in openqa if row["original_assessments"]["answer_in_text"] == 0],
        "answer_in_text_missing_ids": [row["original_id"] for row in openqa if row["original_assessments"]["answer_in_text"] is None],
        "at_labels": dict(collections.Counter("missing" if row["original_assessments"]["at_labels"] is None else str(int(row["original_assessments"]["at_labels"])) for row in openqa)),
        "mcqa_question_differs_from_openqa_ids": question_differences,
        "retrieval_corpus_policy": "All 688 supplied support passages including test support are available as open-book source-only data; this is not QA or label access. Score source Recall@k by chunk.section against expected_source_id.",
        "reference_policy": "Original answer is primary. Nonempty validation answer is preserved second, not assumed equivalent. Original answer_in_text and at_labels are distinct labels; neither substitutes for local expert review.",
        "human_review": "All expert fields are blank; no fabricated expert rating. Support-uncertain rows are highlighted before viewing model outputs, not automatically excluded.",
        "boundaries": ["Development questions are excluded from test by known original ID, EID and normalized supporting text.",
            "Published EID/text separation cannot prove complete source-document or pretraining independence.",
            "No Pira question-answer records may be used for this run's SFT training.",
            "MCQA and open QA use the same 101 IDs, not 202 independent questions.",
            "Only development files are adapted; official test is neither run nor tuned against here."],
    }
    (HERE / "english_stage_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = [
        "# 英文开发集与检索库准备记录", "", "这是正式测试前的开发数据适配；本次没有训练模型，也没有生成模型成绩。", "",
        f"- 开放问答101题、同题五选一101题；已知来源91个EID、90段规范化支撑文本。两种题型不能相加为202道独立题。",
        f"- 检索库只含688段公开支撑原文，分为{len(chunks)}个窗口；窗口最大1100字符、步长850字符。每个窗口section保留原支撑文本ID。",
        "- 开发题与官方测试的题目ID、已知EID、规范化支撑文本重合均为0。这不能证明基座预训练或完整文档层面从未接触。",
        "- 检索库可含测试题支撑原文；这是预先固定的开放资料检索设置。库内没有题目、选项、答案或人工标签。",
        "- 主要EM/F1应只用reference_answer_origin。第二份人工回答保留在reference_answers后位，等价性尚需审核。", "",
        "## 原文含答案的原始标注", "",
        f"统计：{json.dumps(manifest['answer_in_text'], ensure_ascii=False)}。", "",
        "- answer_in_text=0：" + (", ".join(manifest["answer_in_text_zero_ids"]) or "无") + "。",
        "- answer_in_text缺失：" + (", ".join(manifest["answer_in_text_missing_ids"]) or "无") + "。",
        "- 这些题保留在开发诊断中，单列报告；不能把at_labels当作原文是否含答案的替代字段。", "",
        "## 人工核查", "",
        "dev_reference_expert_review.csv包含全部原题、两份参考答案、支撑原文和原始标签；dev_reference_support_uncertain.csv包含上述0／缺失条目。",
        "所有expert_*、reviewer、review_notes字段留空。自动标签提示和语境词启发式只帮助找问题，未冒充专家判断。",
        "应先核查原答案能否由原文支持、另一份答案是否等价、题目是否需要补足地域或论文语境。修改记录单独保留，不覆盖原参考答案。", "",
        "## 运行与复核", "", "```powershell", ".venv/Scripts/python.exe paper_data/external_benchmarks/prepare_english_stage.py", "```", "",
        "固定版本、原始下载URL、输入输出SHA256及检查结果在english_stage_manifest.json中。",
        "本轮禁止将Pirá开发／测试问答制作成SFT训练样本；需要补训时另找来源不重合的材料。", "",
        "数据归属：C4AI／Pirá作者，数据集CC BY 4.0；底层支撑文本继续保留原作者权利。",
        "来源：https://huggingface.co/datasets/paulopirozelli/pira/tree/" + REVISION,
    ]
    (HERE / "英文开发集准备说明.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"counts": manifest["counts"], "separation": separation, "answer_in_text": manifest["answer_in_text"], "answer_in_text_zero_ids": manifest["answer_in_text_zero_ids"], "answer_in_text_missing_ids": manifest["answer_in_text_missing_ids"], "mcqa_question_differences": len(question_differences)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

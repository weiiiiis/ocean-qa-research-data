"""Supply released source passages for the separately planned 21-item review.

This reads only fixed questions, corpus and blank review CSV. It never reads
generated answers, assigns human ratings or changes the frozen experiment.
"""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "paper_data/external_benchmarks"
OUT = ROOT / "outputs/paper_eval/reference_suitability21_frozen_v1_20261017"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Preserve any teacher work; output already exists: {OUT}")
    question_path = DATA / "official_openqa_en_227.jsonl"
    corpus_path = DATA / "corpus_pira_chunks_en.jsonl"
    plan_path = DATA / "pretest_review_plan_20261017/reference_suitability_review_21.csv"
    questions = {q["id"]: q for q in rows(question_path)}
    corpus = rows(corpus_path)
    with plan_path.open(encoding="utf-8-sig", newline="") as stream:
        planned = list(csv.DictReader(stream))
    expected = {qid for qid, q in questions.items()
                if q["original_assessments"].get("answer_in_text") != 1}
    if len(planned) != 21 or {q["id"] for q in planned} != expected:
        raise ValueError("The separate review must preserve exactly the fixed 21 items")
    lines = ["# 另外21题：参考答案与支撑原文复核", "",
             "这是预先单列的10题 answer_in_text=0 和11题标签缺失记录。原标签与参考答案原样保留；这些标签不能直接解释成应当拒答。", "",
             "请配合同目录的空白CSV填写参考答案是否适用、已发布支撑文本是否足够、问题是否有意义及理由。本材料没有模型答案或模型身份，也没有自动填写人工结论。", "",
             "下面展示Pirá发布的支撑文本窗口，可能存在相邻窗口重叠；它们不是完整论文。若现有原文不足以裁决，请保留待定并记录需要补查的来源。", ""]
    joined = []
    for index, plan in enumerate(planned, 1):
        q = questions[plan["id"]]
        if (plan["question"] != q["question"] or
                plan["published_original_reference"] != q["reference_answer_origin"] or
                plan["expected_source_id_for_review_only"] != q["expected_source_id"]):
            raise ValueError(f"Planned question/reference/source changed: {q['id']}")
        sources = [chunk for chunk in corpus if chunk["section"] == q["expected_source_id"]]
        if not sources:
            raise ValueError(f"Missing released support: {q['id']}")
        joined.append({"id": q["id"], "original_id": q["original_id"],
                       "question": q["question"], "reference_answer_origin": q["reference_answer_origin"],
                       "published_answer_in_text": q["original_assessments"].get("answer_in_text"),
                       "expected_source_id": q["expected_source_id"], "released_source_windows": sources,
                       "human_review_status": "pending"})
        label = q["original_assessments"].get("answer_in_text")
        lines += [f"## {index}. {q['id']}", "", f"**Question:** {q['question']}", "",
                  f"**Published original reference:** {q['reference_answer_origin']}", "",
                  f"**Published answer_in_text:** {'missing' if label is None else label}", "",
                  f"**Source ID:** `{q['expected_source_id']}`", ""]
        for chunk in sources:
            lines += [f"### Source window {chunk['id']}", "", chunk["text"], "",
                      f"Source: {chunk['source']}", "", f"URL: {chunk['url']}", "",
                      f"Rights: {chunk['license']}", ""]
    OUT.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(plan_path, OUT / plan_path.name)
    (OUT / "支撑原文与填写说明.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "reference_sources_21.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in joined), encoding="utf-8")
    manifest = {"purpose": "Separate reference/evidence suitability review; no generated answers",
                "questions": len(joined), "human_scores": "blank/pending", "model_answers_read": False,
                "inputs_sha256": {str(p.relative_to(ROOT)): digest(p)
                                  for p in (question_path, corpus_path, plan_path)},
                "script_sha256": digest(Path(__file__)),
                "outputs_sha256": {p.name: digest(p) for p in OUT.iterdir() if p.is_file()}}
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared {len(joined)} original questions with released source windows: {OUT}")


if __name__ == "__main__":
    main()

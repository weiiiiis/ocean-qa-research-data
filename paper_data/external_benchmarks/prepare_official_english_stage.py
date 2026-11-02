"""Adapt the pinned 227 official test records without running or tuning a model.

Original test files and the existing source-only retrieval corpus stay unchanged.
Question/choice/answer text is copied verbatim from each official task variant.
"""
from __future__ import annotations

import collections
import csv
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REVISION = "756f73c8a52e2b0fbdc8f42569e1e8c6f3865a49"
LICENSE = "CC BY 4.0 (Pira dataset); preserve original supporting-text source rights"
ASSESSMENTS = ("answer_in_text", "at_labels", "question_meaningful", "answer_equivalent",
               "answer_difficulty", "question_generic", "question_type")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def normalized(text: str) -> str:
    # Match the pre-existing development/corpus preparation normalization.
    return re.sub(r"[^\w]", "", text.casefold())


def clean(value):
    return None if value is None or isinstance(value, float) and math.isnan(value) else value


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def file_record(path: Path) -> dict:
    return {"file": path.relative_to(HERE).as_posix(), "bytes": path.stat().st_size, "sha256": digest(path)}


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows), encoding="utf-8")


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})


def main() -> None:
    raw_data = HERE / "raw/pira/data"
    inputs = {split: next(raw_data.glob(f"{split}-*.parquet")) for split in ("train", "validation", "test")}
    inputs.update(mcqa_test=next((HERE / "raw/pira/mcqa").glob("test-*.parquet")),
                  selected_dev=HERE / "pira_dev_article_disjoint_en_101.jsonl",
                  prepared_dev=HERE / "dev_openqa_en.jsonl", passages=HERE / "pira_supporting_passages.jsonl",
                  corpus=HERE / "corpus_pira_chunks_en.jsonl", previous_openqa=HERE / "pira_test_en_227.jsonl",
                  previous_mcqa=HERE / "pira_mcqa_test_en_227.jsonl",
                  historical_training_overlap_review=HERE / "需重点复核的2题.csv")
    hashes_before = {key: digest(path) for key, path in inputs.items()}
    pinned = json.loads((HERE / "download_manifest.json").read_text(encoding="utf-8"))
    pinned_files = {record["local_file"].replace("\\", "/"): record for record in pinned["files"]}
    for key in ("train", "validation", "test", "mcqa_test"):
        path = inputs[key]
        record = pinned_files[path.relative_to(HERE).as_posix()]
        if record["revision"] != REVISION or record["sha256"] != hashes_before[key]:
            raise ValueError(f"Pinned source file changed: {path}")
    raw_splits = {split: pd.read_parquet(inputs[split]).to_dict("records") for split in ("train", "validation", "test")}
    test = raw_splits["test"]
    choices_raw = pd.read_parquet(inputs["mcqa_test"]).to_dict("records")
    choices_by_id = {row["id"]: row for row in choices_raw}
    test_ids = {row["id_qa"] for row in test}
    if len(test) != 227 or len(test_ids) != 227 or len(choices_by_id) != 227 or test_ids != set(choices_by_id):
        raise ValueError("Official openQA and MCQA must contain the same 227 unique original IDs")
    previous_openqa = {row["original_id"]: row for row in read_jsonl(inputs["previous_openqa"])}
    previous_mcqa = {row["original_id"]: row for row in read_jsonl(inputs["previous_mcqa"])}
    if test_ids != set(previous_openqa) or test_ids != set(previous_mcqa):
        raise ValueError("Existing original exports do not match pinned official IDs")
    with inputs["historical_training_overlap_review"].open(encoding="utf-8-sig", newline="") as stream:
        historical_review = list(csv.DictReader(stream))
    historical_flag_original_ids = {row["id"].removeprefix("PIRA-") for row in historical_review}
    if historical_flag_original_ids != {"B2585", "A1329"} or not historical_flag_original_ids <= test_ids:
        raise ValueError("Historical two-question review identity changed; inspect its original-ID mapping")
    passages = read_jsonl(inputs["passages"])
    passage_by_text = {normalized(row["text"]): row for row in passages}
    chunks = read_jsonl(inputs["corpus"])
    if len(passages) != 688 or len(passage_by_text) != 688 or len(chunks) != 1507:
        raise ValueError("Expected existing 688 source passages / 1507 source-only chunks")
    allowed = {"id", "text", "source", "pdf_page", "section", "license", "url"}
    if any(set(chunk) != allowed for chunk in chunks):
        raise ValueError("Existing retrieval corpus includes fields outside the source-only schema")
    chunk_sections = {chunk["section"] for chunk in chunks}

    # Index only source metadata/text for the cross-split audit, never use these
    # matches to edit questions, answers, selection, retrieval or model settings.
    comparison_splits = {"official_train": raw_splits["train"], "official_validation": raw_splits["validation"]}
    selected_dev = read_jsonl(inputs["selected_dev"])
    if len(selected_dev) != 101:
        raise ValueError("Expected the selected 101-question development pool")
    comparison_splits["selected_dev101"] = [{"id_qa": row["original_id"], "abstract": row["supporting_text"],
                                             "eid_article_scopus": row["source_article_eid"]} for row in selected_dev]
    prepared_dev = read_jsonl(inputs["prepared_dev"])
    if {row["original_id"] for row in prepared_dev} != {row["original_id"] for row in selected_dev}:
        raise ValueError("Selected and prepared development pools differ")
    indexes = {}
    for name, rows in comparison_splits.items():
        eid_index, text_index = collections.defaultdict(list), collections.defaultdict(list)
        for row in rows:
            eid = clean(row["eid_article_scopus"])
            if eid is not None:
                eid_index[str(eid)].append(row["id_qa"])
            text_index[normalized(row["abstract"])].append(row["id_qa"])
        indexes[name] = {"eid": eid_index, "text": text_index, "ids": {row["id_qa"] for row in rows}}

    openqa, mcqa, overlap_rows, uncertain_rows = [], [], [], []
    question_differences, group_metadata = [], {}
    for raw in test:
        original_id = raw["id_qa"]
        choice = choices_by_id[original_id]
        text_key = normalized(raw["abstract"])
        passage = passage_by_text[text_key]
        passage_id = passage["id"]
        if passage_id not in chunk_sections or normalized(choice["text"]) != text_key:
            raise ValueError(f"Source mapping mismatch: {original_id}")
        eid = clean(raw["eid_article_scopus"])
        eid = str(eid) if eid is not None else None
        source_group = "PIRA-EID-" + eid if eid else passage_id
        group_unit = "known_article_eid" if eid else "supporting_passage_id"
        origin, alternate = raw["answer_en_origin"], clean(raw["answer_en_validate"])
        if not isinstance(origin, str) or not origin.strip():
            raise ValueError(f"Missing original primary answer: {original_id}")
        references = list(dict.fromkeys(value for value in (origin, alternate) if isinstance(value, str) and value.strip()))
        assessments = {field: clean(raw[field]) for field in ASSESSMENTS}
        common = {
            "id": "PIRA-TEST-" + original_id, "original_id": original_id, "dataset": "pira_2_official_test",
            "prior_normalized_id": previous_openqa[original_id]["id"],
            "prior_mcqa_normalized_id": previous_mcqa[original_id]["id"],
            "historical_training_overlap_review_flag": original_id in historical_flag_original_ids,
            "dataset_revision": REVISION, "corpus": int(raw["corpus"]), "source_group": source_group,
            "source_group_unit": group_unit, "source_article_eid": eid, "expected_source_id": passage_id,
            "mode": "professional", "language": "en", "split": "test", "original_split": "test",
            "original_assessments": assessments,
            "reference_answer_origin": origin, "reference_answer_validate": alternate, "reference_answers": references,
            "reference_answer_origin_source": "Pinned Pira data/test answer_en_origin, copied verbatim",
            "reference_policy": "Original answer is primary; alternate reference is retained without assuming equivalence",
            "review_status": "Official labels retained; independent local factual, suitability and equivalence review pending",
        }
        openqa.append({**common, "task": "open_qa", "question": raw["question_en_origin"],
                       "question_source_field": "data/test.question_en_origin"})
        options = {letter: choice[letter] for letter in "ABCDE"}
        label = choice["alternative"]
        if label not in options or options[label].strip() != choice["correct"].strip():
            raise ValueError(f"Official MC label/text mismatch: {original_id}")
        mcqa.append({**common, "task": "mcqa", "question": choice["question"], "choices": options,
                     "answer": label, "reference_answer_text": choice["correct"],
                     "question_source_field": "mcqa/test.question", "mc_answer_source": "mcqa/test.alternative, copied verbatim"})
        if openqa[-1]["question"] != previous_openqa[original_id]["question"]:
            raise ValueError(f"OpenQA stem changed: {original_id}")
        old_mc = previous_mcqa[original_id]
        if any(mcqa[-1][key] != old_mc[key] for key in ("question", "choices", "answer", "reference_answer_text")):
            raise ValueError(f"MC question, choices or labels changed: {original_id}")
        if choice["question"] != raw["question_en_origin"]:
            question_differences.append(original_id)
        group_metadata[source_group] = {"unit": group_unit, "corpus": int(raw["corpus"]), "article_eid": eid}
        overlap = {"id": common["id"], "original_id": original_id, "corpus": common["corpus"],
                   "source_group": source_group, "source_group_unit": group_unit, "source_article_eid": eid,
                   "expected_source_id": passage_id}
        for name, index in indexes.items():
            overlap[name + "_question_id_overlap"] = original_id in index["ids"]
            overlap[name + "_matching_known_eid_question_ids"] = index["eid"].get(eid, []) if eid else []
            overlap[name + "_matching_normalized_passage_question_ids"] = index["text"].get(text_key, [])
        overlap_rows.append(overlap)
        if assessments["answer_in_text"] in (None, 0):
            uncertain_rows.append({"id": common["id"], "original_id": original_id, "corpus": common["corpus"],
                "question": raw["question_en_origin"], "reference_answer_origin": origin,
                "answer_in_text_original": assessments["answer_in_text"], "at_labels_original": assessments["at_labels"],
                "expected_source_id": passage_id, "published_status": "answer_in_text missing" if assessments["answer_in_text"] is None else "answer_in_text=0",
                "expert_supports_original_answer": "", "expert_notes": "", "reviewer": ""})

    # Merge released EIDs that share the same normalized support text. Union
    # source groups, not QA records, so same EID OR same text defines a component.
    original_group_metadata = group_metadata
    parent = {group: group for group in original_group_metadata}
    def find(group):
        while parent[group] != group:
            parent[group] = parent[parent[group]]
            group = parent[group]
        return group
    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b)
    original_passage_groups = collections.defaultdict(set)
    for row in openqa:
        original_passage_groups[row["expected_source_id"]].add(row["source_group"])
    for groups in original_passage_groups.values():
        groups = sorted(groups)
        for group in groups[1:]:
            union(groups[0], group)
    components = collections.defaultdict(list)
    for group in sorted(parent):
        components[find(group)].append(group)
    group_map, group_metadata = {}, {}
    for members in components.values():
        label = (members[0] if len(members) == 1 else "PIRA-SOURCE-COMP-" +
                 hashlib.sha256("\n".join(members).encode("utf-8")).hexdigest()[:16])
        for member in members:
            group_map[member] = label
        original_metadata = [original_group_metadata[member] for member in members]
        group_metadata[label] = {
            "unit": original_metadata[0]["unit"] if len(members) == 1 else "known_eid_or_exact_passage_component",
            "corpora": sorted({item["corpus"] for item in original_metadata}),
            "article_eids": sorted({item["article_eid"] for item in original_metadata if item["article_eid"]}),
            "original_source_groups": members,
        }
    by_prepared_id = {}
    for rows in (openqa, mcqa):
        for row in rows:
            row["source_group_original"] = row["source_group"]
            row["source_group"] = group_map[row["source_group"]]
            row["source_group_unit"] = group_metadata[row["source_group"]]["unit"]
            row["source_group_conservative"] = "PIRA-UNRESOLVED-CORPUS2" if row["corpus"] == 2 else row["source_group"]
            by_prepared_id[row["id"]] = row
    for row in overlap_rows:
        prepared = by_prepared_id[row["id"]]
        row["source_group_original"] = row["source_group"]
        row["source_group"] = prepared["source_group"]
        row["source_group_unit"] = prepared["source_group_unit"]
        row["source_group_conservative"] = prepared["source_group_conservative"]

    if [row["id"] for row in openqa] != [row["id"] for row in mcqa]:
        raise ValueError("Prepared task IDs must be identical and identically ordered")
    answer_counts = dict(collections.Counter("missing" if row["original_assessments"]["answer_in_text"] is None
                         else str(int(row["original_assessments"]["answer_in_text"])) for row in openqa))
    if answer_counts != {"1": 206, "0": 10, "missing": 11}:
        raise ValueError(f"Original answer_in_text annotations changed: {answer_counts}")
    overlap_summary = {}
    for name in indexes:
        eid_rows = [row for row in overlap_rows if row[name + "_matching_known_eid_question_ids"]]
        text_rows = [row for row in overlap_rows if row[name + "_matching_normalized_passage_question_ids"]]
        overlap_summary[name] = {
            "question_id_overlap_count": sum(row[name + "_question_id_overlap"] for row in overlap_rows),
            "test_questions_with_known_eid_overlap": len(eid_rows),
            "unique_overlapping_known_eids": len({row["source_article_eid"] for row in eid_rows}),
            "test_questions_with_normalized_passage_overlap": len(text_rows),
            "unique_overlapping_supporting_passage_ids": len({row["expected_source_id"] for row in text_rows}),
        }
    if any(overlap_summary["selected_dev101"].values()):
        raise ValueError("Selected development101 separation failed within known ID/EID/text scope")
    if {row["source_group"] for row in openqa} & {row["source_group"] for row in prepared_dev}:
        raise ValueError("Prepared development and test source groups unexpectedly overlap")
    if any(digest(path) != hashes_before[key] for key, path in inputs.items()):
        raise ValueError("An input changed during preparation; no outputs will be published")
    uncertain_path = HERE / "official_test_support_uncertain.csv"
    # Protect any future human notes before reproducing metadata.
    if uncertain_path.exists():
        with uncertain_path.open(encoding="utf-8-sig", newline="") as stream:
            if any(any(row.get(key, "").strip() for key in ("expert_supports_original_answer", "expert_notes", "reviewer"))
                   for row in csv.DictReader(stream)):
                raise ValueError("Support-review file contains human work; preserve it before rerunning preparation")
    outputs = []
    for name, rows in (("official_openqa_en_227.jsonl", openqa), ("official_mcqa_en_227.jsonl", mcqa)):
        path = HERE / name
        write_jsonl(path, rows)
        outputs.append(path)
    overlap_path = HERE / "official_test_cross_split_overlap.csv"
    write_csv(overlap_path, overlap_rows, list(overlap_rows[0])); outputs.append(overlap_path)
    write_csv(uncertain_path, uncertain_rows, list(uncertain_rows[0])); outputs.append(uncertain_path)
    shared_passage_groups = {passage: sorted(groups) for passage, groups in original_passage_groups.items() if len(groups) > 1}
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "stage": "Official test schema preparation only; no model generation or tuning",
        "dataset": "Pira 2.0", "revision": REVISION, "license": LICENSE,
        "script": file_record(Path(__file__)), "inputs": [file_record(path) for path in inputs.values()],
        "outputs": [file_record(path) for path in outputs],
        "counts": {"official_openqa": len(openqa), "official_mcqa": len(mcqa),
                   "unique_original_questions": len(test_ids), "corpus1_questions": sum(row["corpus"] == 1 for row in openqa),
                   "corpus2_questions": sum(row["corpus"] == 2 for row in openqa),
                   "known_article_eid_questions": sum(row["source_article_eid"] is not None for row in openqa),
                   "questions_without_known_eid": sum(row["source_article_eid"] is None for row in openqa),
                   "original_source_groups_before_text_merge": len(original_group_metadata),
                   "unique_source_groups": len(group_metadata), "unique_known_article_eids": len({row["source_article_eid"] for row in openqa if row["source_article_eid"]}),
                   "original_group_counts_by_unit": dict(collections.Counter(row["unit"] for row in original_group_metadata.values())),
                   "source_group_counts_by_unit": dict(collections.Counter(row["unit"] for row in group_metadata.values())),
                   "source_group_counts_by_corpus": {str(corpus): len({row["source_group"] for row in openqa if row["corpus"] == corpus}) for corpus in (1, 2)},
                   "source_group_conservative_count": len({row["source_group_conservative"] for row in openqa}),
                   "unique_expected_source_passages": len({row["expected_source_id"] for row in openqa}),
                   "existing_corpus_source_passages": len(passages), "existing_corpus_chunks": len(chunks)},
        "task_id_policy": "Both tasks use identical PIRA-TEST-{official original_id} in openQA official order; 227 paired question IDs, not 454 independent questions",
        "mcqa_question_differs_from_openqa_ids": question_differences,
        "question_choice_label_text_policy": "Each task's official question string, MC options and label copied verbatim; task-specific question differences preserved",
        "source_group_policy": "Start with known Scopus EID or original supporting passage ID when missing; merge connected components defined by same known EID OR same normalized support passage. source_group_original preserves the initial grouping. Corpus2 has no released unique report ID; passage groups are not whole-report independent groups.",
        "source_group_conservative_policy": "For a conservative sensitivity, all 46 corpus2 questions share PIRA-UNRESOLVED-CORPUS2; corpus1 retains the merged known-EID/text grouping, including its one unresolved missing-EID question. This is a dependence sensitivity assumption, not a claim that corpus2 is one actual report.",
        "source_group_definitions": group_metadata,
        "distinct_original_source_groups_sharing_identical_passage": shared_passage_groups,
        "source_group_limitation": "Different released EIDs sharing identical supporting text are merged. Missing EIDs still cannot certify distinct papers; the corpus1 fallback record may belong to another known article. Neither grouping proves full-document independence, especially for corpus2 without report IDs.",
        "expected_source_mapping": "Every expected_source_id equals an existing corpus chunk.section; retrieval corpus is reused unchanged, with no QA/label additions",
        "cross_split_known_overlap": overlap_summary,
        "selected_dev101_separation_scope": "Original question ID, released nonmissing article EID and normalized exact supporting passage only. Group labels also disjoint. This does not prove whole-report, semantic, hidden-source or base-pretraining independence.",
        "official_split_boundary": "Official question splits may share article EIDs or exact supporting passages; overlap is exported rather than removed. No official train/validation QA is added to this project's SFT.",
        "cross_split_question_id_overlap_details": {name: sorted(test_ids & index["ids"]) for name, index in indexes.items()},
        "historical_training_overlap_sensitivity_mask": {
            "original_ids": sorted(historical_flag_original_ids),
            "prepared_ids": ["PIRA-TEST-" + original_id for original_id in sorted(historical_flag_original_ids)],
            "prior_normalized_ids": [previous_openqa[original_id]["id"] for original_id in sorted(historical_flag_original_ids)],
            "match_rule": "Match original_id exactly in both tasks; never match only an old/new prefix",
            "status": "Prior automatic semantic review flags, not confirmed leakage; labels and primary 227 remain unchanged",
        },
        "answer_in_text": answer_counts,
        "answer_in_text_zero_ids": [row["original_id"] for row in openqa if row["original_assessments"]["answer_in_text"] == 0],
        "answer_in_text_missing_ids": [row["original_id"] for row in openqa if row["original_assessments"]["answer_in_text"] is None],
        "at_labels": dict(collections.Counter("missing" if row["original_assessments"]["at_labels"] is None
                          else str(int(row["original_assessments"]["at_labels"])) for row in openqa)),
        "annotation_policy": "Retain all 227. answer_in_text is preserved separately from at_labels; at_labels is not a refusal/abstention label and neither overrides original gold answers. No automatic deletion or inferred unanswerability.",
        "reference_policy": "reference_answer_origin is verbatim answer_en_origin; alternate validation answer remains optional and unverified; published source-field names retained",
        "human_review": "pending; all expert/reviewer cells blank", "original_inputs_and_corpus_unchanged": True,
        "official_model_run_performed": False, "official_model_performance_inspected": False,
    }
    write_json(HERE / "official_english_stage_manifest.json", manifest)
    lines = ["# Pirá官方227题英文评测入口准备", "",
        "本次仅适配数据字段和来源元数据；没有运行正式模型、查看正式模型表现或调整训练参数。", "",
        "## 可用文件", "",
        "- official_openqa_en_227.jsonl：开放问答227题，reference_answer_origin保留官方原答案。",
        "- official_mcqa_en_227.jsonl：五选一227题，原始选项及字母标签保留。",
        "- 两任务id完全一致且同序，original_id保留官方ID；题干分别沿用各任务原文，存在差异的ID列在manifest。它们不是454道独立题。",
        "- 检索库继续使用corpus_pira_chunks_en.jsonl：688段原始支撑文本、1507个source-only窗口，内容未修改。所有expected_source_id均存在于chunk.section。", "",
        "## 分组及已知跨split关系", "",
        f"官方test包含corpus1={manifest['counts']['corpus1_questions']}题、corpus2={manifest['counts']['corpus2_questions']}题；180题有已知Scopus EID，按EID分组。其余47题（corpus2全部46题及corpus1缺EID的1题）按支撑段落ID分组。",
        "corpus2发布元数据没有可用的唯一完整报告ID，**段落分组不能解释为完整报告相互独立**。混合EID/段落分组所得区间需保留这一限制。", "",
        f"初始分组184（145个EID组、39个段落fallback组）；三对不同EID共享相同规范化文本，按同EID或同文本连通分量合并后，主source_group为{manifest['counts']['unique_source_groups']}组。原184组保留在source_group_original；主组定义和合并关系见manifest。缺EID的corpus1那1题可能与其他记录同篇论文，无法确定。", "",
        f"source_group_conservative另将corpus2的46题全部放入PIRA-UNRESOLVED-CORPUS2，其余按主分组，共{manifest['counts']['source_group_conservative_count']}组，用作保守相关性敏感性分析；它不表示46题实际来自同一份报告，也不证明完整文档独立。", "",
        "| 对照范围 | test题已知EID重合数 | 重合EID数 | test题规范化原文重合数 | 重合支撑段落数 |", "|---|---:|---:|---:|---:|"]
    for name, summary in overlap_summary.items():
        lines.append(f"| {name} | {summary['test_questions_with_known_eid_overlap']} | {summary['unique_overlapping_known_eids']} | {summary['test_questions_with_normalized_passage_overlap']} | {summary['unique_overlapping_supporting_passage_ids']} |")
    lines += ["", "逐题匹配信息在official_test_cross_split_overlap.csv。选定dev101与test227仅在已知题ID、EID及规范化相同支撑文本范围内未重合；不声称未知全文、报告、语义或基座预训练独立。官方train/validation的来源重合按实际记录披露，不改官方题目划分。", "",
        "“与官方train重合的197题”是test题记录数；这些题对应155个不同的规范化支撑文本来源。155是unique来源数，197是题数，两者不是矛盾或不同版本。EID一栏同理，161个题记录对应128个重合EID。", "",
        "官方train/test还共享原题ID A173：题干相同、支撑文本不同；原始记录保留，来源关系另行披露。这不表示本项目使用了官方train问答进行SFT。", "",
        "## 旧重叠复核标记的稳定关联", "",
        "prior_normalized_id保留旧PIRA-B2585／PIRA-A1329等ID，prior_mcqa_normalized_id保留旧MC前缀。两任务都有historical_training_overlap_review_flag。旧训练数据相似性敏感性掩码必须按original_id恰好等于B2585或A1329匹配，不能因新PIRA-TEST前缀漏掉。", "",
        "这两题是先前自动相似度复核标记，尚非确认泄漏；本次不自动删除。主文件保留227题，另有原ID及新旧ID映射供预先规定的敏感性分析使用。", "",
        "## 原始标签与参考答案", "",
        "answer_in_text=1：206题；=0：10题；缺失：11题，全部保留。31条0/缺失记录单列official_test_support_uncertain.csv，人工栏为空。",
        "at_labels另行保留，**不能将at_labels解释为拒答标签，也不据此将问题自动改成不可回答或删题**。参考原答案仍是发布的answer_en_origin；另一份答案不默认语义等价。", "",
        "## 复现准备", "", "```powershell", ".venv\\Scripts\\python.exe paper_data/external_benchmarks/prepare_official_english_stage.py", "```", "",
        "准备脚本不加载模型。模型版本与提示/检索协议应先在开发集选定并冻结，之后才使用这些官方文件评测；运行目录应与开发集分开。输入、输出、原始版本及来源分组见official_english_stage_manifest.json。", "",
        "数据归属：Pirá作者／C4AI，数据集CC BY 4.0；底层支撑原文保留原作者权利。", "",
        "固定来源：https://huggingface.co/datasets/paulopirozelli/pira/tree/" + REVISION, ""]
    (HERE / "官方英文测试集准备说明.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"counts": manifest["counts"], "answer_in_text": answer_counts,
                      "cross_split_known_overlap": overlap_summary, "mcqa_question_differences": len(question_differences)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

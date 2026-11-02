"""Freeze completed development selection before any official model generation.

CPU/file preparation only. No model is loaded and no official output is read.
"""
from __future__ import annotations

import argparse
import ast
import copy
import json
from datetime import datetime, timezone
from pathlib import Path

from .common import CONDITIONS, ROOT, digest_file, digest_json, read_jsonl, resolve, write_json

BENCH = ROOT / "paper_data/external_benchmarks"
PLAN = BENCH / "pretest_review_plan_20261017"
WEIGHT_SUFFIXES = {".safetensors", ".bin"}
ARTIFACT_SUFFIXES = WEIGHT_SUFFIXES | {".json", ".txt", ".model", ".tiktoken"}


def record(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": digest_file(path)}


def inventory(folder: Path) -> dict[str, str]:
    return {str(path.resolve()): digest_file(path) for path in sorted(folder.rglob("*"))
            if path.is_file() and path.suffix in ARTIFACT_SUFFIXES and ".cache" not in path.parts}


def declared_inventory(manifest: dict) -> dict[str, str]:
    return {str(resolve(name).resolve()): value for name, value in manifest["signature"]["model_files"].items()}


def is_within(path: str, folder: Path) -> bool:
    try:
        Path(path).relative_to(folder)
        return True
    except ValueError:
        return False


def literal_constant(path: Path, name: str):
    for node in ast.parse(path.read_text(encoding="utf-8-sig")).body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError(f"Missing literal {name} in {path}")


def validate_dev(run: Path, task: str, adapter: Path, actual_adapter: dict[str, str], protocol: dict) -> dict:
    required = ("run_manifest.json", "answers.jsonl", "selected_questions.jsonl", "completion.json", "analysis.json")
    files = {name: record(run / name) for name in required}
    manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    signature = manifest["signature"]
    saved_protocol = signature["protocol"]
    if saved_protocol.get("task") != task or resolve(saved_protocol["adapter"]).resolve() != adapter:
        raise ValueError(f"Wrong development task or adapter: {run}")
    for key in ("prompt_profile", "retrieval", "seed", "mcqa_max_new_tokens"):
        if saved_protocol.get(key) != protocol.get(key):
            raise ValueError(f"Development {key} differs from selected official protocol: {run}")
    expected_generation = copy.deepcopy(protocol["generation"])
    if task == "mcqa":
        expected_generation["max_new_tokens"] = protocol["mcqa_max_new_tokens"]
    if saved_protocol["generation"] != expected_generation:
        raise ValueError(f"Development generation settings differ: {run}")
    selected, answers = read_jsonl(run / "selected_questions.jsonl"), read_jsonl(run / "answers.jsonl")
    ids = {question["id"] for question in selected}
    if len(selected) != 101 or len(ids) != 101 or len({q["original_id"] for q in selected}) != 101:
        raise ValueError(f"Development selection must contain 101 unique original questions: {run}")
    if any(q.get("split") != "dev" or bool(q.get("choices")) != (task == "mcqa") for q in selected):
        raise ValueError(f"Development split/task mismatch: {run}")
    conditions = manifest.get("conditions", list(CONDITIONS))
    if not {"lora", "lora_rag"} <= set(conditions) or any(c not in CONDITIONS for c in conditions) or len(set(conditions)) != len(conditions):
        raise ValueError(f"Development must request lora and lora_rag: {run}")
    if len({(r["condition"], r["id"]) for r in answers}) != len(answers):
        raise ValueError(f"Duplicate development result: {run}")
    if any(r["condition"] not in conditions or r["id"] not in ids or r.get("task") != task for r in answers):
        raise ValueError(f"Unexpected development result: {run}")
    counts = {c: len([r for r in answers if r["condition"] == c]) for c in conditions}
    if any({r["id"] for r in answers if r["condition"] == c} != ids for c in conditions):
        raise ValueError(f"Development run is incomplete: {run}")
    completion = json.loads((run / "completion.json").read_text(encoding="utf-8"))
    analysis = json.loads((run / "analysis.json").read_text(encoding="utf-8"))
    if not completion.get("complete") or not analysis.get("complete") or analysis.get("task") != task:
        raise ValueError(f"Development completion/metrics are not final: {run}")
    if completion.get("signature_sha256") != signature.get("signature_sha256"):
        raise ValueError(f"Development completion signature mismatch: {run}")
    for filename, field in (("answers.jsonl", "answers_sha256"), ("selected_questions.jsonl", "selected_questions_sha256")):
        if analysis["artifacts"].get(field) != files[filename]["sha256"]:
            raise ValueError(f"Development metrics refer to different raw artifacts: {run}")
    supplied = declared_inventory(manifest)
    adapter_saved = {path: value for path, value in supplied.items() if is_within(path, adapter)}
    if adapter_saved != actual_adapter:
        raise ValueError(f"Selected adapter files differ from those actually used on development: {run}")
    dataset_path, corpus_path = resolve(signature["dataset_path"]), resolve(signature["corpus_path"])
    if digest_file(dataset_path) != signature["dataset_sha256"] or digest_file(corpus_path) != signature["corpus_sha256"]:
        raise ValueError(f"Development input data changed: {run}")
    return {"directory": str(run), "task": task, "conditions": conditions, "counts": counts,
        "selected_original_ids": sorted(q["original_id"] for q in selected), "files": files,
        "signature": signature, "declared_inventory": supplied,
        "automatic_summary": analysis["automatic_overlap_or_mc_metrics"],
        "cost_summary": analysis.get("cost_descriptive"), "expert_status": "pending; no approval inferred",
        "answer_boundary": "All original generated development answers remain preserved; no extraction overwrites raw text"}


def training_lineage(adapter: Path) -> dict:
    layers, seen, limitations = [], set(), []
    current: Path | None = adapter
    while current is not None:
        if current in seen:
            raise ValueError("Cycle in adapter training lineage")
        seen.add(current)
        weights = inventory(current)
        config_path = next((folder / "training_config.json" for folder in (current, current.parent, current.parent.parent)
                            if (folder / "training_config.json").is_file()), None)
        layer = {"adapter": str(current), "adapter_files_sha256": weights}
        if config_path is None:
            limitations.append(f"Training config unavailable for ancestor {current}; lineage cannot be certified beyond it")
            layer["training_config_status"] = "missing"
            layers.append(layer)
            break
        config = json.loads(config_path.read_text(encoding="utf-8"))
        layer.update(training_config=record(config_path), training_configuration=config)
        source_records = []
        for key, hash_key in (("train_file", "train_sha256"), ("validation_file", "validation_sha256"), ("data", "data_sha256")):
            if not config.get(key):
                continue
            source = resolve(config[key]).resolve()
            item = record(source)
            if config.get(hash_key) and item["sha256"] != config[hash_key]:
                raise ValueError(f"Training source changed since checkpoint creation: {source}")
            item["configuration_field"] = key
            item["historical_hash_available"] = bool(config.get(hash_key))
            source_records.append(item)
            data_manifest = source.parent / "manifest.json"
            if data_manifest.is_file():
                layer.setdefault("data_manifests", {})[str(data_manifest)] = record(data_manifest)
                metadata = json.loads(data_manifest.read_text(encoding="utf-8"))
                for name in ("source_facts_60.jsonl", "family_split.json", "system_prompt.txt"):
                    path = source.parent / name
                    if path.is_file():
                        layer.setdefault("source_artifacts", {})[str(path)] = record(path)
                for field, path in (("source_pdf_sha256", ROOT / "docs/webb_intro_oceanography.pdf"),
                                    ("source_corpus_sha256", ROOT / "paper_data/corpus.jsonl")):
                    if metadata.get(field) and path.is_file():
                        actual = record(path)
                        if actual["sha256"] != metadata[field]:
                            raise ValueError(f"Training source provenance mismatch: {path}")
                        layer.setdefault("source_artifacts", {})[str(path)] = actual
        layer["training_source_files"] = source_records
        for name in ("training_metrics.json", "train_data.jsonl", "validation_data.jsonl"):
            path = config_path.parent / name
            if path.is_file():
                layer.setdefault("saved_training_artifacts", {})[name] = record(path)
        parent_adapter = resolve(config["start_adapter"]).resolve() if config.get("start_adapter") else None
        if parent_adapter is not None:
            parent_actual = inventory(parent_adapter)
            declared = config.get("starting_adapter_files_sha256")
            if declared and any(parent_actual.get(str((parent_adapter / name).resolve())) != value
                                for name, value in declared.items()):
                raise ValueError(f"Ancestor adapter changed since continued training: {parent_adapter}")
            layer["starting_adapter_hash_verified"] = bool(declared)
        layers.append(layer)
        current = parent_adapter
    return {"layers": layers, "limitations": limitations,
        "boundary": "Recorded checkpoint/data provenance only; no claim of independent expert correctness or unseen base pretraining"}


def freeze(args: argparse.Namespace) -> None:
    out = resolve(args.out).resolve()
    if out.exists():
        raise FileExistsError(f"Freeze output already exists: {out}")
    adapter = resolve(args.adapter).resolve()
    protocol_source = ROOT / "paper_eval/protocol_en_bm25_v2.json"
    protocol = json.loads(protocol_source.read_text(encoding="utf-8"))
    protocol["adapter"] = str(adapter)
    if protocol.get("prompt_profile") != "evidence_v2" or protocol["retrieval"].get("method") != "bm25":
        raise ValueError("Expected the selected evidence-v2 BM25 protocol")
    actual_adapter = inventory(adapter)
    if not actual_adapter or not any(Path(path).name in {"adapter_model.safetensors", "adapter_model.bin"} for path in actual_adapter):
        raise ValueError("A saved adapter with actual weight files is required")
    devs = {"openqa": validate_dev(resolve(args.open_dev_run).resolve(), "openqa", adapter, actual_adapter, protocol),
            "mcqa": validate_dev(resolve(args.mc_dev_run).resolve(), "mcqa", adapter, actual_adapter, protocol)}
    if devs["openqa"]["selected_original_ids"] != devs["mcqa"]["selected_original_ids"]:
        raise ValueError("Two development tasks must cover the same original 101 questions")
    base = resolve(protocol["base_model"]).resolve()
    base_inventories = [{path: value for path, value in dev["declared_inventory"].items() if is_within(path, base)} for dev in devs.values()]
    if not base_inventories[0] or base_inventories[0] != base_inventories[1]:
        raise ValueError("Two development runs did not use identical base-model files")
    base_record = {}
    for name, value in base_inventories[0].items():
        path = Path(name)
        if not path.is_file():
            raise FileNotFoundError(path)
        inherited = path.suffix in WEIGHT_SUFFIXES and path.stat().st_size > 64 * 1024**2
        if not inherited and digest_file(path) != value:
            raise ValueError(f"Base-model configuration/tokenizer changed: {path}")
        base_record[name] = {"sha256": value, "bytes_now": path.stat().st_size,
                             "verification": "Reused identical hashes from both completed dev manifests; large weights not rehashed at freeze" if inherited else "Recomputed actual file hash at freeze"}
    official_manifest_path = BENCH / "official_english_stage_manifest.json"
    official_manifest = json.loads(official_manifest_path.read_text(encoding="utf-8"))
    dataset_files = [BENCH / "official_openqa_en_227.jsonl", BENCH / "official_mcqa_en_227.jsonl"]
    prepared = [read_jsonl(path) for path in dataset_files]
    if any(len(rows) != 227 or len({r["original_id"] for r in rows}) != 227 for rows in prepared):
        raise ValueError("Both official datasets must contain 227 unique original questions")
    if [r["original_id"] for r in prepared[0]] != [r["original_id"] for r in prepared[1]]:
        raise ValueError("Official tasks must preserve identical ordered original IDs")
    declared_outputs = {item["file"]: item["sha256"] for item in official_manifest["outputs"]}
    for path in dataset_files:
        if digest_file(path) != declared_outputs[path.name]:
            raise ValueError(f"Prepared official dataset differs from its manifest: {path}")
    corpus = BENCH / "corpus_pira_chunks_en.jsonl"
    chunks = read_jsonl(corpus)
    allowed = {"id", "text", "source", "pdf_page", "section", "license", "url"}
    if len(chunks) != 1507 or any(set(row) != allowed for row in chunks):
        raise ValueError("Expected unchanged 1507 source-only retrieval chunks")
    sections = {row["section"] for row in chunks}
    if any(r["expected_source_id"] not in sections for rows in prepared for r in rows):
        raise ValueError("Official gold-source identity missing from existing source-only corpus")
    if any(digest_file(corpus) != dev["signature"]["corpus_sha256"] for dev in devs.values()):
        raise ValueError("Official retrieval corpus differs from completed development corpus")
    plan_manifest_path = PLAN / "manifest.json"
    plan = json.loads(plan_manifest_path.read_text(encoding="utf-8"))
    for label, value in plan["inputs_sha256"].items():
        if digest_file(resolve(label)) != value:
            raise ValueError(f"Pretest plan input changed: {label}")
    for label, value in plan["outputs_sha256"].items():
        if digest_file(PLAN / label) != value:
            raise ValueError(f"Pretest plan output changed: {label}")
    sample = read_jsonl(PLAN / "selected_review_questions_40.jsonl")
    if len(sample) != 40 or [r["original_id"] for r in sample] != plan["sample"]["selected_original_ids"]:
        raise ValueError("Preselected 40-question review plan is invalid")
    rules = plan["automatic_analysis_rules"]
    if literal_constant(ROOT / "paper_eval/english_report.py", "MC_PREFIX_PATTERN") != rules["mcqa"]["leading_option_regex"]:
        raise ValueError("MC scorer option parser differs from the pretest plan")
    if literal_constant(ROOT / "paper_eval/official_english_report.py", "SEED") != rules["uncertainty"]["bootstrap_seed"]:
        raise ValueError("Official source bootstrap seed differs from the pretest plan")
    exclude = rules["fixed_sensitivity_question_sets"][0]["exclude_original_ids"]
    if set(exclude) != {"A1329", "B2585"}:
        raise ValueError("Fixed 225-question mask changed")
    mask = [r["original_id"] for r in prepared[0] if r["original_id"] not in set(exclude)]
    if len(mask) != 225:
        raise ValueError("225-question sensitivity denominator changed")
    code_names = ("english.py", "inference.py", "common.py", "retrieval.py", "bm25_retrieval.py",
                  "english_report.py", "official_english_report.py", "report.py", "export_official_blind_subset.py", "freeze_english_experiment.py")
    code = {name: record(ROOT / "paper_eval" / name) for name in code_names}
    inference_names = ("english.py", "inference.py", "common.py", "retrieval.py", "bm25_retrieval.py")
    for dev in devs.values():
        if any(dev["signature"]["code_files"].get(name) != code[name]["sha256"] for name in inference_names):
            raise ValueError("Inference code changed since completed selected-model development")
    lineage = training_lineage(adapter)
    # Preserve the source protocol exactly except for the selected adapter path.
    frozen = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen before first official model generation, expert pending",
        "registration_status": "Local freeze only; no external preregistration or expert approval claimed",
        "official_model_outputs_read": False, "official_generation_performed_by_freezer": False,
        "expert_review": {"status": "pending", "expert_approval": False, "factual_correctness": None, "completeness": None, "citation_support": None},
        "protocol_source": record(protocol_source), "selected_adapter": str(adapter),
        "selected_adapter_actual_file_sha256": actual_adapter, "base_model": {"path": str(base), "files": base_record},
        "training_lineage": lineage, "code": code,
        "development": {task: {key: value for key, value in dev.items() if key not in {"declared_inventory", "signature"}}
                        for task, dev in devs.items()},
        "official_conditions": list(CONDITIONS), "official_tasks": ["openqa", "mcqa"],
        "official_dataset_files": [record(path) for path in dataset_files], "source_only_corpus": record(corpus),
        "official_preparation_manifest": record(official_manifest_path),
        "pretest_plan_manifest": record(plan_manifest_path), "pretest_plan_rules": rules,
        "human_review_rules": plan["human_review_rules"], "review40_file": record(PLAN / "selected_review_questions_40.jsonl"),
        "review40_original_ids": plan["sample"]["selected_original_ids"],
        "sensitivity_225": {"exclude_original_ids": sorted(exclude), "included_original_ids": mask,
                            "mask_sha256": digest_json(mask), "policy": "Identical original-ID mask in both tasks; prior review flags are not proven leakage"},
        "output_retention": "Preserve every raw answer, prompt/context fingerprint and actual provided passage. Do not replace raw text with extracted MC letters or edited citations.",
        "test_policy": "Official outputs are held out for final evaluation; do not select adapters, prompts, retrieval, exclusions or training on official test performance. Report any necessary post-freeze amendment separately.",
        "unresolved_limitations": ["Selected-model no-RAG development answers include invented/unprovided citation IDs (7 observed); this defect is retained, not silently repaired in official outputs.",
            "RAG development citation IDs being resolvable is not evidence that citations support the claims.",
            "Factual correctness, essential completeness and evidence support have no independent expert ratings or approval.",
            "The new English sources and benchmark references remain expert-review pending; automatic similarity screening does not prove no historical/pretraining contamination.",
            "Corpus2 report lineage is unresolved. Main/conservative grouping are uncertainty assumptions, not proof of complete-document independence.",
            "Natural-response latency is confounded by output length and device/model order; no inference-efficiency improvement claim is licensed."],
    }
    # Final race check on the selected weights and source protocol before writing.
    if inventory(adapter) != actual_adapter or digest_file(protocol_source) != frozen["protocol_source"]["sha256"]:
        raise ValueError("Selected adapter or protocol changed while freezing")
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "protocol.json", protocol)
    frozen["frozen_protocol_sha256"] = digest_file(out / "protocol.json")
    frozen["freeze_sha256"] = digest_json(frozen)
    write_json(out / "freeze_manifest.json", frozen)
    (out / "README.md").write_text("# 英文海洋问答正式实验冻结记录\n\n"
        "状态：frozen before first official model generation, expert pending。仅本地冻结，没有外部预注册或专家批准。\n\n"
        f"选定适配器：`{adapter}`。开放与选择题开发集均完整101题，实际包含LoRA及LoRA＋RAG；权重、原始回答和指标文件已核对。详细开发摘要与训练祖先/数据来源见freeze_manifest.json。\n\n"
        "protocol.json保持protocol_en_bm25_v2设置，仅替换adapter路径；正式条件固定base、base_rag、lora、lora_rag。两个题型共享227个原始ID，不能合计454个独立题。\n\n"
        "正式主集227、旧标记排除敏感性225、answer_in_text=1的206题及其他21题，规则与预先40题抽样、两位专家独立评价方案均已记录。225掩码按original_id匹配A1329/B2585。\n\n"
        "保留所有原始输出，不改引用或MC正文以改善成绩。正式测试不用于模型、提示、检索或训练调参。任何冻结后的必要变更须单独记录并保留旧版本。\n\n"
        "当前仍有无RAG虚构引用编号（开发观察7条）；RAG编号可解析不代表证据支持。事实、完整性和引用支持专家分均pending；没有专家批准。\n\n"
        "大型base权重SHA复用两个已完成开发manifest的一致记录，未再次读取全部大权重；选定adapter、配置、代码、数据和预案均重新计算实际哈希。冻结器没有加载模型、使用GPU或读取正式模型输出。\n", encoding="utf-8")
    print(f"Frozen selected English experiment: {out}; official generation not performed; expert review pending")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--open-dev-run", required=True)
    parser.add_argument("--mc-dev-run", required=True)
    parser.add_argument("--out", required=True)
    freeze(parser.parse_args())


if __name__ == "__main__":
    main()

"""Align v2 user-message postambles with the evaluator; tokenizer only, no GPU."""
from __future__ import annotations

import collections
import hashlib
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PARENT = HERE.with_name("english_training_v2")
DATA_FILES = ("candidates_300.jsonl", "train_candidates_240.jsonl", "validation_candidates_60.jsonl")
TRACE_FILES = ("facts.tsv", "source_facts_60.jsonl", "family_split.json", "source_anchor_and_expert_review.csv", "system_prompt.txt")
NO_CONTEXT = "\n\nNo reference passages were supplied. Do not add bracketed citations."
WITH_CONTEXT = "\n\nAnswer the question directly using the relevant passages, and cite their exact identifiers. If they do not support an answer, state that the supplied passages are insufficient."


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    from transformers import AutoTokenizer

    parent_manifest = json.loads((PARENT / "manifest.json").read_text(encoding="utf-8"))
    parent_files = [*DATA_FILES, *TRACE_FILES, "manifest.json", "README.md", "prepare_v2.py"]
    parent_hashes = {name: sha(PARENT / name) for name in parent_files}
    parent_hashes.update({p.relative_to(PARENT).as_posix(): sha(p) for p in sorted((PARENT / "revision_history").rglob("*")) if p.is_file()})
    existing = HERE / "manifest.json"
    if existing.exists() and json.loads(existing.read_text(encoding="utf-8"))["parent_v2_files_sha256"] != parent_hashes:
        raise ValueError("V2 parent changed; use a new output version")
    tokenizer = AutoTokenizer.from_pretrained(str(ROOT / "models/Qwen3-4B-Instruct"), local_files_only=True)
    outputs, summaries = {}, {}
    for name in DATA_FILES:
        originals = read_jsonl(PARENT / name)
        converted, lengths = [], []
        for original in originals:
            row = json.loads(json.dumps(original))
            suffix = NO_CONTEXT if row["example_type"] == "closed_book" else WITH_CONTEXT
            row["messages"][1]["content"] += suffix
            length = len(tokenizer.apply_chat_template(row["messages"], tokenize=True, add_generation_prompt=False, enable_thinking=False))
            if length > 1024:
                raise ValueError(f"Sample exceeds 1024-token budget: {row['id']} ({length})")
            row["token_length_qwen3_chat_template"] = length
            reverted = json.loads(json.dumps(row))
            reverted["messages"][1]["content"] = reverted["messages"][1]["content"][:-len(suffix)]
            reverted["token_length_qwen3_chat_template"] = original["token_length_qwen3_chat_template"]
            if reverted != original:
                raise ValueError("Non-postamble content changed")
            converted.append(row)
            lengths.append(length)
        (HERE / name).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in converted), encoding="utf-8")
        outputs[name] = converted
        summaries[name] = {"examples": len(converted), "families": len({row["family_id"] for row in converted}),
            "types": dict(collections.Counter(row["example_type"] for row in converted)),
            "full_chat_tokens": {"min": min(lengths), "max": max(lengths), "mean": sum(lengths) / len(lengths), "limit": 1024, "truncated_examples": 0}}
    train, validation = outputs[DATA_FILES[1]], outputs[DATA_FILES[2]]
    if {row["family_id"] for row in train} & {row["family_id"] for row in validation}:
        raise ValueError("Family leakage")
    if {row["source_group"] for row in train} & {row["source_group"] for row in validation}:
        raise ValueError("Section leakage")
    full = {row["id"]: row for row in outputs[DATA_FILES[0]]}
    if len(full) != 300 or any(row != full[row["id"]] for row in train + validation):
        raise ValueError("Full set and splits differ")
    for name in TRACE_FILES:
        shutil.copy2(PARENT / name, HERE / name)
    shutil.copytree(PARENT / "revision_history", HERE / "revision_history", dirs_exist_ok=True)
    provenance = HERE / "provenance_v2"
    provenance.mkdir(exist_ok=True)
    for name in ("manifest.json", "README.md", "prepare_v2.py"):
        shutil.copy2(PARENT / name, provenance / name)
    output_paths = [HERE / name for name in (*DATA_FILES, *TRACE_FILES)]
    output_paths += [p for directory in (HERE / "revision_history", provenance) for p in directory.rglob("*") if p.is_file()]
    manifest = {"version": "english_training_v3", "status": "Candidate message-format alignment only; no model training",
        "parent": "paper_data/english_training_v2", "parent_v2_manifest_sha256": parent_hashes["manifest.json"],
        "parent_v2_files_sha256": parent_hashes, "prompt_profile": "evidence_v2",
        "change": "Append evaluator-matching postamble to user messages only; system, question, assistant, facts, sources, labels and splits are unchanged. Token lengths are recomputed.",
        "closed_book_user_suffix": NO_CONTEXT, "context_user_suffix": WITH_CONTEXT,
        "system_prompt_text_sha256": parent_manifest["system_prompt_text_sha256"], "system_prompt_file_sha256": sha(HERE / "system_prompt.txt"),
        "source_pdf_sha256": parent_manifest["source_pdf_sha256"], "source_corpus_sha256": parent_manifest["source_corpus_sha256"],
        "license_and_attribution": parent_manifest["license_and_attribution"], "gold_status": parent_manifest["gold_status"],
        "summaries": summaries, "checks": {"only_user_postamble_changed": True, "no_facts_added_or_removed": True,
            "full_template_length_without_truncation": True, "family_and_section_train_validation_disjoint": True,
            "expert_approval": False, "model_training_started": False},
        "trainer_input_policy": parent_manifest["trainer_input_policy"], "evidence_scope_policy": parent_manifest["evidence_scope_policy"],
        "boundaries": parent_manifest["boundaries"],
        "output_files": {p.relative_to(HERE).as_posix(): {"sha256": sha(p), "bytes": p.stat().st_size} for p in output_paths},
        "builder_sha256": sha(Path(__file__))}
    (HERE / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"version": manifest["version"], "summaries": summaries}, ensure_ascii=False))


if __name__ == "__main__":
    main()

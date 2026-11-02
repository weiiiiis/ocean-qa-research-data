"""Copy the reviewed v1 candidates and replace only the system prompt with V2.

Loads the local tokenizer on CPU, never language-model weights or the GPU.
"""
from __future__ import annotations

import ast
import collections
import hashlib
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PARENT = HERE.with_name("english_training_v1")
DATA_FILES = ("candidates_300.jsonl", "train_candidates_240.jsonl", "validation_candidates_60.jsonl")
TRACE_FILES = ("facts.tsv", "source_facts_60.jsonl", "family_split.json", "source_anchor_and_expert_review.csv")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def prompt_constant() -> str:
    tree = ast.parse((ROOT / "paper_eval/english.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "OPEN_SYSTEM_V2" for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError("OPEN_SYSTEM_V2 constant is missing")


def main() -> None:
    from transformers import AutoTokenizer

    if HERE.resolve() == PARENT.resolve():
        raise ValueError("V2 output cannot be its parent")
    system = prompt_constant()
    parent_manifest = json.loads((PARENT / "manifest.json").read_text(encoding="utf-8"))
    parent_files = [*DATA_FILES, *TRACE_FILES, "manifest.json", "system_prompt.txt", "prepare_candidates.py", "README.md"]
    parent_hashes = {name: sha(PARENT / name) for name in parent_files}
    parent_hashes.update({p.relative_to(PARENT).as_posix(): sha(p) for p in sorted((PARENT / "revision_history").rglob("*")) if p.is_file()})
    old_manifest_path = HERE / "manifest.json"
    if old_manifest_path.exists():
        previous = json.loads(old_manifest_path.read_text(encoding="utf-8"))
        if previous["parent_v1_files_sha256"] != parent_hashes:
            raise ValueError("The reviewed V1 parent changed; use a new version directory")
        if previous["system_prompt_text_sha256"] != hashlib.sha256(system.encode("utf-8")).hexdigest():
            raise ValueError("OPEN_SYSTEM_V2 changed; use a new version directory")

    tokenizer = AutoTokenizer.from_pretrained(str(ROOT / "models/Qwen3-4B-Instruct"), local_files_only=True)
    all_outputs, summaries = {}, {}
    for name in DATA_FILES:
        original = read_jsonl(PARENT / name)
        output = []
        lengths, words = [], []
        for row in original:
            updated = json.loads(json.dumps(row))
            if [message["role"] for message in updated["messages"]] != ["system", "user", "assistant"]:
                raise ValueError(f"Unexpected conversation schema: {row['id']}")
            updated["messages"][0]["content"] = system
            length = len(tokenizer.apply_chat_template(updated["messages"], tokenize=True, add_generation_prompt=False, enable_thinking=False))
            if length > 1024:
                raise ValueError(f"V2 sample exceeds 1024 tokens: {row['id']}")
            updated["token_length_qwen3_chat_template"] = length
            if updated["messages"][1:] != row["messages"][1:]:
                raise ValueError("The V2 copy changed user/assistant text")
            # All fields except the system text and its derived token length
            # must remain byte-equivalent at the parsed JSON level.
            reverted = json.loads(json.dumps(updated))
            reverted["messages"][0]["content"] = row["messages"][0]["content"]
            reverted["token_length_qwen3_chat_template"] = row["token_length_qwen3_chat_template"]
            if reverted != row:
                raise ValueError("Unexpected non-prompt change in V2")
            output.append(updated)
            lengths.append(length)
            words.append(len(updated["messages"][-1]["content"].split()))
        (HERE / name).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output), encoding="utf-8")
        all_outputs[name] = output
        summaries[name] = {
            "examples": len(output), "families": len({row["family_id"] for row in output}),
            "source_groups": len({row["source_group"] for row in output}),
            "types": dict(collections.Counter(row["example_type"] for row in output)),
            "full_chat_tokens": {"min": min(lengths), "max": max(lengths), "mean": sum(lengths) / len(lengths), "limit": 1024, "truncated_examples": 0},
            "assistant_answer_max_whitespace_words": max(words),
        }
    full = {row["id"]: row for row in all_outputs[DATA_FILES[0]]}
    train, validation = all_outputs[DATA_FILES[1]], all_outputs[DATA_FILES[2]]
    if {row["family_id"] for row in train} & {row["family_id"] for row in validation}:
        raise ValueError("Family leakage across splits")
    if {row["source_group"] for row in train} & {row["source_group"] for row in validation}:
        raise ValueError("Section leakage across splits")
    if any(row != full[row["id"]] for row in train + validation):
        raise ValueError("Train/validation no longer match the full candidate set")
    for name in TRACE_FILES:
        shutil.copy2(PARENT / name, HERE / name)
    shutil.copytree(PARENT / "revision_history", HERE / "revision_history", dirs_exist_ok=True)
    provenance_dir = HERE / "provenance_v1"
    provenance_dir.mkdir(exist_ok=True)
    for name in ("manifest.json", "README.md", "prepare_candidates.py", "system_prompt.txt"):
        shutil.copy2(PARENT / name, provenance_dir / name)
    (HERE / "system_prompt.txt").write_text(system + "\n", encoding="utf-8")
    output_paths = [HERE / name for name in (*DATA_FILES, *TRACE_FILES, "system_prompt.txt")]
    output_paths += [p for directory in (HERE / "revision_history", provenance_dir) for p in directory.rglob("*") if p.is_file()]
    manifest = {
        "version": "english_training_v2", "status": "Candidate system-prompt adaptation only; no model training",
        "parent": "paper_data/english_training_v1", "parent_v1_manifest_sha256": parent_hashes["manifest.json"],
        "parent_v1_files_sha256": parent_hashes,
        "change": "Only messages[0].content is replaced by OPEN_SYSTEM_V2; user/assistant messages, IDs, families, splits, sources and labels are unchanged. Token lengths are recomputed.",
        "prompt_profile": "evidence_v2", "system_prompt_source": "paper_eval/english.py OPEN_SYSTEM_V2 (AST constant; no import)",
        "system_prompt_text_sha256": hashlib.sha256(system.encode("utf-8")).hexdigest(),
        "system_prompt_file_sha256": sha(HERE / "system_prompt.txt"),
        "source_pdf_sha256": parent_manifest["source_pdf_sha256"], "source_corpus_sha256": parent_manifest["source_corpus_sha256"],
        "license_and_attribution": parent_manifest["license_and_attribution"], "gold_status": parent_manifest["gold_status"],
        "summaries": summaries, "checks": {"only_system_prompt_changed": True, "full_template_length_without_truncation": True,
            "family_and_section_train_validation_disjoint": True, "expert_approval": False, "model_training_started": False},
        "trainer_input_policy": parent_manifest["trainer_input_policy"],
        "evidence_scope_policy": parent_manifest["evidence_scope_policy"],
        "boundaries": parent_manifest["boundaries"],
        "output_files": {p.relative_to(HERE).as_posix(): {"sha256": sha(p), "bytes": p.stat().st_size} for p in output_paths},
        "builder_sha256": sha(Path(__file__)),
    }
    write_json(HERE / "manifest.json", manifest)
    print(json.dumps({"version": manifest["version"], "prompt_profile": manifest["prompt_profile"], "summaries": summaries}, ensure_ascii=False))


if __name__ == "__main__":
    main()

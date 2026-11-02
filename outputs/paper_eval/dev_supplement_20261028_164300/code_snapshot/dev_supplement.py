"""Development-only budget/stage supplement; reuse the frozen English components.

python -m paper_eval.dev_supplement --out outputs/paper_eval/dev_supplement_TIMESTAMP
Resume only with identical inputs, code, packages, prompts and model hashes.
"""
from __future__ import annotations

import argparse
import collections
import gc
import json
import os
import platform
import random
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from .common import ROOT, digest_file, digest_json, read_jsonl, write_json, write_jsonl
from .english import _packages, fit_prompt, model_inventory, validate_dataset
from .inference import append_result, balanced_subset, generate, load_model

FROZEN = ROOT / "paper_data/frozen_english_v1_20261017"
DATASET = ROOT / "paper_data/external_benchmarks/dev_openqa_en.jsonl"
CORPUS = ROOT / "paper_data/external_benchmarks/corpus_pira_chunks_en.jsonl"
MODELS = {"base": None,
          "v1": ROOT / "outputs/ocean_qwen3_english_grounded_v1/epoch_1",
          "v2": ROOT / "outputs/ocean_qwen3_english_grounded_v2/epoch_1"}
CELLS = [(s, r, b) for s in ("base", "v2") for r in (False, True) for b in (160, 96)] + [
    ("v1", r, 160) for r in (False, True)]


def now():
    return datetime.now(timezone.utc).isoformat()


def cell(stage, rag, budget):
    return f"{stage}{'_rag' if rag else ''}_t{budget}"


def protected_inventory(out):
    # Preserve all pre-existing evaluation artifacts, training inputs, protocols and reports.
    paths = list((ROOT / "paper_data").rglob("*"))
    paths += list((ROOT / "outputs/paper_eval").rglob("*"))
    paths += list((ROOT / "outputs").glob("ocean_qwen3_english_grounded_*/*"))
    return {str(p.relative_to(ROOT)): digest_file(p) for p in sorted(set(paths))
            if p.is_file() and out not in p.parents}


def check_protected(out):
    before = json.loads((out / "protected_before.json").read_text(encoding="utf-8"))
    changed = [p for p, h in before.items() if not (ROOT / p).is_file() or digest_file(ROOT / p) != h]
    result = {"checked_at_utc": now(), "files_checked": len(before), "changed_or_missing": changed,
              "original_results_preserved": not changed}
    write_json(out / "protected_integrity.json", result)
    if changed:
        raise ValueError(f"Protected files changed: {changed}")
    return result


def prepare(out):
    from transformers import AutoTokenizer
    from .bm25_retrieval import Retriever
    rows = read_jsonl(DATASET)
    validate_dataset(rows, "openqa")
    if any(q.get("split") != "dev" for q in rows):
        raise ValueError("This supplement permits development data only")
    rows = balanced_subset(rows, 0)
    pilot = rows[:12]
    p = json.loads((FROZEN / "protocol.json").read_text(encoding="utf-8"))
    base = ROOT / p["base_model"]
    tokenizer = AutoTokenizer.from_pretrained(str(base), local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    retriever = Retriever(CORPUS)
    ret = p["retrieval"]
    for key in ("k1", "b", "query_term_frequency"):
        assert ret[key] == retriever.implementation_metadata[key]
    for key, value in retriever.implementation_metadata["lexical_settings"].items():
        assert ret[key] == value
    retrieval, prompts = {}, {}
    for q in rows:
        retrieval[q["id"]] = retriever.search(q["question"], ret["top_k"], ret["min_score"])[0]
        prompts[q["id"]] = {}
        for rag in (False, True):
            prompt, hits = fit_prompt(tokenizer, {"id": q["id"], "question": q["question"]},
                retrieval[q["id"]] if rag else [], "openqa", ret["max_context_tokens"],
                p["generation"]["max_input_tokens"], p["prompt_profile"])
            prompts[q["id"]][str(rag)] = {"prompt": prompt, "retrieved_context": hits,
                "prompt_sha256": digest_json(prompt), "context_sha256": digest_json(hits),
                "input_tokens": len(tokenizer.encode(prompt, add_special_tokens=False))}
    code = [ROOT / "paper_eval" / n for n in ("dev_supplement.py", "english.py", "inference.py",
            "common.py", "bm25_retrieval.py")]
    protocol = {"version": "dev-budget-stage-supplement-v1", "frozen_reference_protocol": p,
        "generation_budgets": [160, 96], "cells": [cell(*c) for c in CELLS],
        "dataset": str(DATASET), "corpus": str(CORPUS), "questions": len(rows),
        "source_groups": len({q["source_group"] for q in rows}),
        "pilot_ids": [q["id"] for q in pilot],
        "pilot_selection": "Existing balanced_source_group_round_robin_v1; first 12, before generation; retain all results",
        "pilot_gate": "All 12 x 8 records present, unique, exact prompt/context equality and termination metadata; truncation does not fail the gate",
        "sequence": "pilot: base,v2 x budgets160,96; gate; remaining: base,v2; all v1 at160; one GPU process",
        "order": "Alternate no-RAG/RAG and budget order by question index within each model block",
        "quantization": {"load_in_4bit": True, "type": "nf4", "double_quant": True,
                         "compute_dtype": "float16", "attention": "sdpa", "batch_size": 1},
        "statistics": {"seed": 20261028, "resamples": 10000,
            "method": "Paired percentile source-group cluster bootstrap; question-weighted mean difference; identical sampled clusters for all comparisons",
            "grouping": "Published source_group (known article EID); not proof of independent complete documents"},
        "scope": "English dev only; no training, no official test inference, no historical answers reused",
        "interpretation": "Generation budget sensitivity, not strict equal-length control. Stage comparison confounds second-stage design with extra training.",
        "termination": "Existing generate reports EOS or len(ids)>=budget without EOS; output_tokens includes EOS. Text is decoded with skip_special_tokens=True and whitespace strip only, no content cropping.",
        "words": "Whitespace-separated words in full decoded answer, citations included; also save citation-stripped words",
        "f1": "Existing SQuAD normalization and original published reference, citations removed; lexical overlap only",
        "citations": "Existing english_report regex; each comma/semicolon-separated identifier counted; resolution against supplied chunk IDs only"}
    signature = {"protocol": protocol, "dataset_sha256": digest_file(DATASET),
        "corpus_sha256": digest_file(CORPUS), "frozen_protocol_sha256": digest_file(FROZEN / "protocol.json"),
        "model_files": model_inventory([base, MODELS["v1"], MODELS["v2"]]),
        "code_files": {str(f.relative_to(ROOT)): digest_file(f) for f in code},
        "packages": _packages(), "python": platform.python_version(),
        "retrieval_sha256": digest_json(retrieval), "prompt_plan_sha256": digest_json(prompts),
        "selected_questions_sha256": digest_json(rows)}
    signature["signature_sha256"] = digest_json(signature)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "run_manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8"))["signature"] != signature:
            raise ValueError("Signature mismatch: refuse resume; use a new directory")
        for name, value in (("prompt_plan.json", prompts), ("retrieval_plan.json", retrieval), ("protocol.json", protocol)):
            if json.loads((out / name).read_text(encoding="utf-8")) != value:
                raise ValueError(f"Frozen {name} changed")
        if read_jsonl(out / "selected_questions.jsonl") != rows:
            raise ValueError("Frozen selected questions changed")
    else:
        if (out / "answers.jsonl").exists():
            raise ValueError("Results without signature cannot resume")
        write_json(out / "protected_before.json", protected_inventory(out))
        write_json(manifest_path, {"created_at_utc": now(), "signature": signature,
            "hardware": subprocess.check_output(["nvidia-smi"], text=True),
            "ancestry": "base -> v1/epoch_1 -> v2/epoch_1; v1/epoch_2 is not an ancestor",
            "old_results_reused": False})
        for name, value in (("protocol.json", protocol), ("prompt_plan.json", prompts),
                            ("retrieval_plan.json", retrieval), ("retrieval_implementation.json", retriever.implementation_metadata)):
            write_json(out / name, value)
        write_jsonl(out / "selected_questions.jsonl", rows)
        write_jsonl(out / "pilot_questions.jsonl", pilot)
        snap = out / "code_snapshot"
        snap.mkdir(exist_ok=True)
        for f in code:
            shutil.copy2(f, snap / f.name)
        shutil.copy2(FROZEN / "protocol.json", snap / "original_frozen_protocol.json")
        write_json(out / "preflight.json", {"created_at_utc": now(), "actual_questions": len(rows),
            "unique_ids": len({q["id"] for q in rows}), "source_groups": len({q["source_group"] for q in rows}),
            "source_group_sizes": dict(collections.Counter(collections.Counter(q["source_group"] for q in rows).values())),
            "pilot_source_groups": len({q["source_group"] for q in pilot}),
            "corpus_chunks": len(retriever.chunks), "corpus_sources": len({c["section"] for c in retriever.chunks}),
            "expected_101_matches": len(rows) == 101, "all_dev": True,
            "ancestors_agreed_with_model_card": True, "applicable_AGENTS": [],
            "base_model_label": "Qwen3-4B-Instruct-2507 (local directory Qwen3-4B-Instruct)",
            "prior_frozen_hash_match": {k: digest_file(ROOT / k) == h for k, h in
                json.loads((FROZEN / "freeze_manifest.json").read_text(encoding="utf-8"))["model_files_sha256"].items()
                if k in signature["model_files"]},
            "discrepancies": [] if len(rows) == 101 else ["Actual development count differs from 101; all actual rows retained"]})
    return rows, prompts, protocol, signature


def validate_answers(out, rows, prompts, sig, pilot_only=False):
    records = read_jsonl(out / "answers.jsonl") if (out / "answers.jsonl").exists() else []
    by_key = {}
    ids = {q["id"] for q in rows}
    expected_cells = {cell(*c) for c in CELLS}
    for r in records:
        key = (r["condition"], r["id"])
        if key in by_key or r["id"] not in ids or r["condition"] not in expected_cells:
            raise ValueError(f"Duplicate/unexpected answer {key}")
        p = prompts[r["id"]][str(r["rag"])]
        if r["signature_sha256"] != sig or any(r[k] != p[k] for k in
                ("prompt", "prompt_sha256", "context_sha256", "retrieved_context")):
            raise ValueError(f"Answer signature/prompt/context mismatch {key}")
        m = r["metrics"]
        if m["hit_token_limit"] == m["eos_terminated"] or m["output_tokens_including_eos"] > r["budget"]:
            raise ValueError(f"Invalid termination metadata {key}")
        if r["finish_reason"] != ("length" if m["hit_token_limit"] else "eos"):
            raise ValueError(f"Invalid finish reason {key}")
        by_key[key] = r
    subset = rows[:12] if pilot_only else rows
    cells = [cell(*c) for c in CELLS if not pilot_only or c[0] != "v1"]
    missing = [(c, q["id"]) for c in cells for q in subset if (c, q["id"]) not in by_key]
    report = {"checked_at_utc": now(), "scope": "pilot" if pilot_only else "full",
        "expected": len(cells) * len(subset), "records_in_file": len(records), "missing": missing,
        "unique": True, "signature_consistent": True, "prompt_context_consistent": True,
        "termination_metadata_valid": True, "complete": not missing,
        "by_condition": {c: {"answers": sum((c, q["id"]) in by_key for q in subset),
            "truncated": sum(by_key[(c,q["id"])]["metrics"]["hit_token_limit"] for q in subset if (c,q["id"]) in by_key)} for c in cells}}
    write_json(out / ("pilot_validation.json" if pilot_only else "integrity.json"), report)
    return by_key, report


def run_phase(out, rows, prompts, protocol, signature, phase):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Local CUDA GPU unavailable")
    torch.set_num_threads(4)
    random.seed(protocol["frozen_reference_protocol"]["seed"])
    torch.manual_seed(protocol["frozen_reference_protocol"]["seed"])
    torch.backends.cudnn.benchmark = False
    existing, _ = validate_answers(out, rows, prompts, signature["signature_sha256"])
    phase_rows = rows[:12] if phase == "pilot" else rows
    for stage in (["base", "v2"] if phase == "pilot" else ["base", "v2", "v1"]):
        stage_cells = [c for c in CELLS if c[0] == stage]
        if all((cell(*c), q["id"]) in existing for c in stage_cells for q in phase_rows):
            continue
        gpu = subprocess.check_output(["nvidia-smi"], text=True)
        append_result(out / "gpu_checks.jsonl", {"checked_at_utc": now(), "phase": phase, "stage": stage, "nvidia_smi": gpu})
        print(f"GPU pre-load check {stage}:\n{gpu}", flush=True)
        # No other jobs are stopped; external conflicts are recorded in the GPU log.
        model = tokenizer = None
        try:
            model, tokenizer, load_ms = load_model(ROOT / protocol["frozen_reference_protocol"]["base_model"], MODELS[stage])
            for q in rows:
                for rag in (False, True):
                    p = prompts[q["id"]][str(rag)]
                    rendered, hits = fit_prompt(tokenizer, {"id": q["id"], "question": q["question"]},
                        json.loads((out / "retrieval_plan.json").read_text(encoding="utf-8"))[q["id"]] if rag else [],
                        "openqa", protocol["frozen_reference_protocol"]["retrieval"]["max_context_tokens"],
                        protocol["frozen_reference_protocol"]["generation"]["max_input_tokens"], "evidence_v2")
                    if rendered != p["prompt"] or hits != p["retrieved_context"]:
                        raise ValueError("Model-loaded prompt differs from frozen shared plan")
            for rag in (False, True):
                generate(model, tokenizer, prompts[rows[0]["id"]][str(rag)]["prompt"],
                         protocol["frozen_reference_protocol"]["generation"], fixed_tokens=16)
            append_result(out / "model_loads.jsonl", {"stage": stage, "phase": phase,
                "load_ms": load_ms, "warmups_excluded": True, "timestamp_utc": now()})
            for i, q in enumerate(phase_rows):
                for s, rag, budget in (stage_cells if i % 2 == 0 else stage_cells[::-1]):
                    condition = cell(s, rag, budget)
                    key = (condition, q["id"])
                    if key in existing:
                        continue
                    p = prompts[q["id"]][str(rag)]
                    config = {**protocol["frozen_reference_protocol"]["generation"], "max_new_tokens": budget}
                    answer, metrics = generate(model, tokenizer, p["prompt"], config)
                    result = {"id": q["id"], "source_group": q["source_group"], "condition": condition,
                        "stage": s, "rag": rag, "budget": budget, "phase_generated": phase,
                        "task": "openqa", "question": q["question"], "answer": answer,
                        **{k: p[k] for k in ("prompt", "prompt_sha256", "context_sha256", "retrieved_context")},
                        "metrics": metrics, "finish_reason": "length" if metrics["hit_token_limit"] else "eos",
                        "signature_sha256": signature["signature_sha256"], "timestamp_utc": now()}
                    append_result(out / "answers.jsonl", result)
                    existing[key] = result
                    print(f"{phase} {condition} {i+1}/{len(phase_rows)} id={q['id']} tokens={metrics['output_tokens_including_eos']} finish={result['finish_reason']} saved={len(existing)}/{len(rows)*len(CELLS)}", flush=True)
        finally:
            del model, tokenizer
            gc.collect()
            torch.cuda.empty_cache()


class Tee:
    def __init__(self, original, log):
        self.original, self.log = original, log
    def write(self, text):
        self.log.write(text); self.log.flush()
        return self.original.write(text)
    def flush(self):
        self.original.flush(); self.log.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--phase", choices=["prepare", "pilot", "all", "validate"], default="all")
    args = parser.parse_args()
    out = args.out.resolve()
    if (ROOT / "outputs/paper_eval") not in out.parents or not out.name.startswith("dev_supplement_"):
        raise ValueError("Use a new outputs/paper_eval/dev_supplement_TIMESTAMP directory")
    out.mkdir(parents=True, exist_ok=True)
    log = (out / "run.log").open("a", encoding="utf-8", buffering=1)
    sys.stdout, sys.stderr = Tee(sys.stdout, log), Tee(sys.stderr, log)
    append_result(out / "commands.jsonl", {"timestamp_utc": now(), "cwd": str(ROOT),
        "argv": [sys.executable, "-m", "paper_eval.dev_supplement", *sys.argv[1:]],
        "command_windows": subprocess.list2cmdline([sys.executable, "-m", "paper_eval.dev_supplement", *sys.argv[1:]]),
        "environment": {k: os.environ.get(k) for k in ("PYTHONUTF8", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")}})
    try:
        rows, prompts, protocol, signature = prepare(out)
        print(f"Frozen: {len(rows)} questions; {protocol['source_groups']} groups; pilot={protocol['pilot_ids']}; signature={signature['signature_sha256']}", flush=True)
        if args.phase in ("all", "pilot"):
            run_phase(out, rows, prompts, protocol, signature, "pilot")
            _, gate = validate_answers(out, rows, prompts, signature["signature_sha256"], pilot_only=True)
            if not gate["complete"]:
                raise RuntimeError("Pilot gate incomplete")
            print(f"Pilot gate passed: {gate['by_condition']}; retaining truncated outputs and both budgets.", flush=True)
        if args.phase == "all":
            run_phase(out, rows, prompts, protocol, signature, "full")
        if args.phase in ("all", "validate"):
            _, integrity = validate_answers(out, rows, prompts, signature["signature_sha256"])
            check_protected(out)
            write_json(out / "completion.json", {**integrity, "signature_sha256": signature["signature_sha256"]})
            print(f"Full completion={integrity['complete']}: {integrity['records_in_file']}/{integrity['expected']}", flush=True)
    except Exception as exc:
        append_result(out / "failures.jsonl", {"timestamp_utc": now(), "type": type(exc).__name__, "error": str(exc)})
        raise


if __name__ == "__main__":
    main()

"""Build 300 traceable English training candidates from 60 textbook facts.

This preparation script never loads a language model or starts training. It
reads Paul Webb's local CC-BY textbook/corpus and a prompt constant only. It
does not read Pira questions, answers, or supporting passages, and does not
translate the project's Chinese candidate evaluation questions.
"""
from __future__ import annotations

import ast
import collections
import csv
import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CORPUS = ROOT / "paper_data/corpus.jsonl"
PDF = ROOT / "docs/webb_intro_oceanography.pdf"
URL = "https://rwu.pressbooks.pub/webboceanography/"
LICENSE = "Introduction to Oceanography by Paul Webb, CC BY 4.0 except where otherwise noted; portions credit Steven Earle's Physical Geology"
GOLD_STATUS = "source-grounded AI-authored candidate, local check performed where traceable; expert review pending"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def read_system() -> str:
    # Parse the constant without importing model/evaluation code.
    tree = ast.parse((ROOT / "paper_eval/english.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "OPEN_SYSTEM" for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError("Cannot locate the English evaluation OPEN_SYSTEM")


def clean_pdf_page(text: str) -> str:
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if re.fullmatch(r"\d+|\d+\s*\|.*|.*\|\s*\d+", line):
            continue
        if not line or line.startswith(("A YouTube element", "Additional links for more information")):
            continue
        lines.append(line)
    return re.sub(r"\s+", " ", " ".join(lines)).strip()


def source_records(facts: list[dict]) -> list[dict]:
    import fitz

    corpus = [json.loads(line) for line in CORPUS.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_page = collections.defaultdict(list)
    for chunk in corpus:
        by_page[(chunk["section"], chunk["pdf_page"])].append(chunk)
    reconstructed = {}
    for (section, page), chunks in by_page.items():
        chunks.sort(key=lambda row: row["id"])
        text = chunks[0]["text"]
        for chunk in chunks[1:]:
            start = (int(chunk["id"].split("_c")[1]) - 1) * 850
            if text[start:] != chunk["text"][:len(text) - start]:
                raise ValueError(f"Corpus window overlap mismatch at {chunk['id']}")
            text = text[:start] + chunk["text"]
        reconstructed[(section, page)] = (text, chunks)

    records = []
    with fitz.open(PDF) as document:
        for index, fact in enumerate(facts, 1):
            matches = []
            for (section, page), (text, chunks) in reconstructed.items():
                if section != fact["section"]:
                    continue
                start = text.find(fact["anchor_start"])
                stop = text.find(fact["anchor_end"], max(0, start))
                if start >= 0 and stop >= start:
                    stop += len(fact["anchor_end"])
                    matches.append((page, text[start:stop], start, stop, chunks))
            if len(matches) != 1:
                raise ValueError(f"Ambiguous/missing source anchor at family {index}: {fact['section']}")
            page, excerpt, start, stop, chunks = matches[0]
            if not 100 <= len(excerpt) <= 600:
                raise ValueError(f"Source excerpt must be compact (100-600 chars): {index}, {len(excerpt)}")
            actual_page = clean_pdf_page(document[page - 1].get_text(sort=True))
            if excerpt not in actual_page:
                raise ValueError(f"Source excerpt not found verbatim on PDF page {page}")
            relevant_chunks = [chunk["id"] for chunk in chunks
                               if (int(chunk["id"].split("_c")[1]) - 1) * 850 < stop
                               and (int(chunk["id"].split("_c")[1]) - 1) * 850 + len(chunk["text"]) > start]
            records.append({
                **fact, "family_id": f"WEBB-EN-F{index:03d}", "source_group": "webb_section_" + fact["section"],
                "pdf_page": page, "paragraph_ids": relevant_chunks, "excerpt": excerpt,
                "excerpt_sha256": hashlib.sha256(excerpt.encode("utf-8")).hexdigest(),
                "anchor_start_verified": True, "anchor_end_verified": True, "excerpt_pdf_verified": True,
            })
    return records


def cited(answer: str, passage_id: str) -> str:
    # All authored answers contain plain sentences without abbreviated stops.
    return " ".join(sentence.rstrip(".") + f" [{passage_id}]."
                    for sentence in re.split(r"(?<=\.)\s+", answer.strip()) if sentence)


def render_user(question: str, passages: list[dict]) -> str:
    if not passages:
        return "Question: " + question
    block = "\n\n".join("[" + passage["id"] + "]\n" + passage["text"] for passage in passages)
    evidence_scope = ("Use only these passages for this evidence-based answer. "
                      "If they do not support the requested information, state that the supplied evidence is insufficient.")
    return "Reference passages:\n" + block + "\n\n" + evidence_scope + "\n\nQuestion: " + question


def provenance(fact: dict, supplied: list[tuple[str, dict, str]]) -> dict:
    def origin(record: dict) -> dict:
        return {"book": "Introduction to Oceanography", "author": "Paul Webb", "url": URL, "license": LICENSE,
                "pdf_file": "docs/webb_intro_oceanography.pdf", "pdf_pages_one_based": [record["pdf_page"]],
                "section": record["section"], "paragraph_ids": record["paragraph_ids"],
                "anchor_start": record["anchor_start"], "anchor_end": record["anchor_end"],
                "excerpt_sha256": record["excerpt_sha256"]}
    return {"question_fact_source": origin(fact), "supplied_passages": [
        {"id_in_prompt": passage_id, "role": role, **origin(record)} for passage_id, record, role in supplied],
        "local_checks": ["English fact authored directly from the English textbook, not translated from candidate Chinese test questions",
                         "Both source anchors located; excerpt matched verbatim on the one-based PDF page",
                         "Answer manually checked against the excerpt by the AI author; not independent expert approval"]}


def main() -> None:
    facts_file = HERE / "facts.tsv"
    with facts_file.open(encoding="utf-8", newline="") as stream:
        facts = list(csv.DictReader(stream, delimiter="\t"))
    if len(facts) != 60 or len({fact["section"] for fact in facts}) != 60:
        raise ValueError("Expected 60 distinct section-level fact families")
    if len({fact["question"] for fact in facts}) != 60:
        raise ValueError("Duplicate primary question")
    records = source_records(facts)
    system = read_system()
    (HERE / "system_prompt.txt").write_text(system + "\n", encoding="utf-8")
    rows, audit_rows = [], []
    for index, fact in enumerate(records):
        # A half-cycle shift pairs early textbook topics with distant later
        # chapters, and vice versa. The pairing table remains reviewable.
        distractor = records[(index + 30) % len(records)]
        if abs(int(fact["section"].split(".")[0]) - int(distractor["section"].split(".")[0])) < 3:
            raise ValueError(f"Distractor chapter too close for {fact['family_id']}")
        source_marker = "WEBB_p" + f"{fact['pdf_page']:03d}" + "_" + "_".join(p.split("_", 2)[2] for p in fact["paragraph_ids"])
        variants = [
            ("closed_book", "no_context", fact["question"], [], fact["answer"]),
            ("grounded", "direct_DOC1", fact["question"], [("DOC1", fact, "support")], cited(fact["answer"], "DOC1")),
            ("grounded", "distractor_mixed_original_source_id", fact["question"],
             [(source_marker, fact, "support"), ("BACKGROUND_" + f"{index + 1:03d}", distractor, "distractor")]
             if index % 2 else [("BACKGROUND_" + f"{index + 1:03d}", distractor, "distractor"), (source_marker, fact, "support")],
             cited(fact["answer"], source_marker)),
            ("grounded", "rephrased_SRC_id", fact["question_rephrased"], [("SRC_F" + f"{index + 1:03d}", fact, "support")],
             cited(fact["answer"], "SRC_F" + f"{index + 1:03d}")),
        ]
        insufficient_id = ("DOC1", "SRC_BACKGROUND", "WEBB_CONTEXT")[index % 3]
        insufficient_answers = [
            "The supplied passage does not provide the information needed to answer this question [" + insufficient_id + "].",
            "The supplied passage does not establish an answer to this question; the required information is absent from this evidence [" + insufficient_id + "].",
            "There is insufficient evidence in the supplied passage to answer this question from that passage alone [" + insufficient_id + "].",
        ]
        variants.append(("insufficient_evidence", "unrelated_passage_only", fact["question"],
                         [(insufficient_id, distractor, "distractor")], insufficient_answers[index % 3]))
        for number, (kind, variant, question, supplied, answer) in enumerate(variants, 1):
            passages = [{"id": passage_id, "text": record["excerpt"]} for passage_id, record, _ in supplied]
            row = {
                "id": fact["family_id"] + f"-E{number}", "source_group": fact["source_group"], "family_id": fact["family_id"],
                "language": "en", "example_type": kind, "variant": variant,
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": render_user(question, passages)},
                             {"role": "assistant", "content": answer}],
                "provenance": provenance(fact, supplied), "gold_status": GOLD_STATUS,
            }
            supplied_ids = {passage["id"] for passage in passages}
            citations = set(re.findall(r"\[([^\[\]]+)\]", answer))
            if citations - supplied_ids or kind == "closed_book" and citations:
                raise ValueError(f"Invented answer citation: {row['id']}")
            if kind == "grounded" and citations != {passage_id for passage_id, _, role in supplied if role == "support"}:
                raise ValueError(f"Grounded citation not restricted to actual supporting document: {row['id']}")
            rows.append(row)
        audit_rows.append({
            "family_id": fact["family_id"], "source_group": fact["source_group"], "pdf_page_one_based": fact["pdf_page"],
            "paragraph_ids": ";".join(fact["paragraph_ids"]), "anchor_start": fact["anchor_start"], "anchor_end": fact["anchor_end"],
            "source_excerpt": fact["excerpt"], "source_excerpt_chars": len(fact["excerpt"]),
            "source_excerpt_sha256": fact["excerpt_sha256"], "anchor_and_pdf_check": "pass: literal source match",
            "question": fact["question"], "question_rephrased": fact["question_rephrased"], "candidate_answer": fact["answer"],
            "insufficient_distractor_family": distractor["family_id"], "insufficient_distractor_topic": distractor["topic"],
            "insufficient_distractor_excerpt": distractor["excerpt"],
            "expert_fact_correct": "", "expert_source_support": "", "expert_distractor_insufficient": "", "reviewer": "", "notes": "",
        })

    # Tokenizer only, CPU only. No model weights, inference or training is loaded.
    from transformers import AutoTokenizer
    tokenizer_path = ROOT / "models/Qwen3-4B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
    lengths = []
    for row in rows:
        length = len(tokenizer.apply_chat_template(row["messages"], tokenize=True, add_generation_prompt=False, enable_thinking=False))
        row["token_length_qwen3_chat_template"] = length
        row["max_sequence_length_checked"] = 1024
        lengths.append(length)
        if length > 1024:
            raise ValueError(f"Candidate exceeds 1024-token budget: {row['id']} ({length})")
    if len({row["id"] for row in rows}) != 300:
        raise ValueError("Expected 300 unique example IDs")
    write_jsonl(HERE / "candidates_300.jsonl", rows)
    write_jsonl(HERE / "source_facts_60.jsonl", records)

    # Offer a development holdout for the training process itself. The external
    # Pira development pool is unrelated and is not included in either split.
    held_out_families = {record["family_id"] for i, record in enumerate(records) if i % 5 == 4}
    train = [row for row in rows if row["family_id"] not in held_out_families]
    validation = [row for row in rows if row["family_id"] in held_out_families]
    if {row["family_id"] for row in train} & {row["family_id"] for row in validation}:
        raise ValueError("Knowledge family crossed candidate train/validation split")
    if {row["source_group"] for row in train} & {row["source_group"] for row in validation}:
        raise ValueError("Section crossed candidate train/validation split")
    def all_source_sections(examples):
        return {origin["section"] for row in examples
                for origin in [row["provenance"]["question_fact_source"], *row["provenance"]["supplied_passages"]]}
    if all_source_sections(train) & all_source_sections(validation):
        raise ValueError("A supplied support/distractor section crossed candidate train/validation split")
    write_jsonl(HERE / "train_candidates_240.jsonl", train)
    write_jsonl(HERE / "validation_candidates_60.jsonl", validation)
    write_json(HERE / "family_split.json", {
        "method": "Every fifth family in the authored source-section order is held out before model outputs are seen",
        "train_families": sorted({row["family_id"] for row in train}), "validation_families": sorted(held_out_families),
        "train_examples": len(train), "validation_examples": len(validation),
        "boundary": "Family/section separation within one textbook, not full-book independence. These are optional candidate training splits, not the external Pira evaluation set.",
        "supplied_passage_sections_also_disjoint": True,
    })
    with (HERE / "source_anchor_and_expert_review.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(audit_rows[0]))
        writer.writeheader()
        writer.writerows(audit_rows)
    manifest = {
        "status": "Candidate data prepared only; not expert-approved; no model trained",
        "gold_status": GOLD_STATUS, "language": "en", "examples": len(rows), "knowledge_families": len(records),
        "source_sections": len({row["source_group"] for row in rows}), "source_books": 1,
        "counts_by_type": dict(collections.Counter(row["example_type"] for row in rows)),
        "counts_by_variant": dict(collections.Counter(row["variant"] for row in rows)),
        "counts_by_chapter_examples": dict(collections.Counter(row["source_group"].split("_")[-1].split(".")[0] for row in rows)),
        "candidate_train_split": {"examples": len(train), "families": len({row["family_id"] for row in train})},
        "candidate_validation_split": {"examples": len(validation), "families": len(held_out_families)},
        "source_url": URL, "license_and_attribution": LICENSE,
        "source_pdf_sha256": sha(PDF), "source_corpus_sha256": sha(CORPUS), "authored_facts_sha256": sha(facts_file),
        "system_prompt_source": "paper_eval/english.py OPEN_SYSTEM (AST constant only)", "system_prompt_sha256": sha(HERE / "system_prompt.txt"),
        "tokenizer": "models/Qwen3-4B-Instruct; local-only tokenizer, no model loaded",
        "token_length_including_full_system_user_assistant_template": {"min": min(lengths), "max": max(lengths), "mean": sum(lengths) / len(lengths), "limit": 1024, "truncated_examples": 0},
        "source_excerpt_chars": {"min": min(len(record["excerpt"]) for record in records), "max": max(len(record["excerpt"]) for record in records)},
        "checks": {"all_60_anchor_pairs_located": True, "all_60_excerpts_matched_local_pdf": True,
                   "all_answer_citations_match_supplied_ids": True, "closed_book_answers_have_no_citations": True,
                   "grounded_answers_cite_only_relevant_supplied_document": True,
                   "all_samples_within_1024_tokens_without_truncation": True, "expert_approval": False,
                   "model_training_started": False},
        "boundaries": ["300 task-format examples represent 60 knowledge families, not 300 independent facts or questions.",
                       "Question paraphrases share a family and are kept in the same candidate split.",
                       "Sources are the local English Webb textbook only, not Pira QA or supporting passages.",
                       "No old Chinese candidate evaluation questions were read or translated by this builder.",
                       "Original text provenance differs from Pira's scientific-abstract/UN-report corpus; generic ocean concepts may still overlap.",
                       "The textbook was also used for a prior Chinese candidate benchmark; do not treat that benchmark as unseen evaluation after this training.",
                       "These short examples do not match the length or specificity of every external research-abstract question.",
                       "Use only if the independent development diagnosis justifies supplementation; no guaranteed improvement is asserted.",
                       "Insufficient-evidence examples assert only that the supplied passage is inadequate, not that the underlying question is globally unanswerable."],
        "trainer_input_policy": "Train only on messages. Provenance, type labels, reference fact metadata and review fields must never be concatenated into user/model input.",
        "evidence_scope_policy": "Every context-supplied positive and negative example has the same explicit passages-only instruction. Negative answers describe inadequacy of that supplied passage, not ignorance or global unanswerability of the ocean fact.",
    }
    output_names = ["candidates_300.jsonl", "source_facts_60.jsonl", "train_candidates_240.jsonl", "validation_candidates_60.jsonl", "family_split.json", "source_anchor_and_expert_review.csv", "system_prompt.txt"]
    manifest["output_files"] = {name: {"sha256": sha(HERE / name), "bytes": (HERE / name).stat().st_size} for name in output_names}
    review_path = HERE / "revision_history/semantic_review_2026-10-17.json"
    if review_path.exists():
        semantic_review = json.loads(review_path.read_text(encoding="utf-8"))
        manifest["semantic_review_revision"] = {
            "file": review_path.relative_to(HERE).as_posix(), "sha256": sha(review_path),
            "review_date_local": semantic_review["review_date_local"],
            "families_reviewed": semantic_review["families_reviewed"],
            "families_with_fact_or_anchor_changes": semantic_review["families_with_fact_or_anchor_changes"],
            "review_type": semantic_review["review_type"], "independent_expert_approval": False,
        }
    write_json(HERE / "manifest.json", manifest)
    print(json.dumps({key: manifest[key] for key in ["status", "examples", "knowledge_families", "counts_by_type", "candidate_train_split", "candidate_validation_split", "token_length_including_full_system_user_assistant_template"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()

from __future__ import annotations

import collections
import csv
import json
import re
from pathlib import Path
from typing import Any

from .common import DATA, ROOT, digest_file, normalize, read_jsonl, validate_questions, write_json, write_jsonl

SOURCE_URL = "https://rwu.pressbooks.pub/webboceanography/"
ATTRIBUTION = "Introduction to Oceanography by Paul Webb, CC BY 4.0, except where otherwise noted. Download this book for free at http://rwu.pressbooks.pub/webboceanography"
SOURCE_PDF = ROOT / "docs/webb_intro_oceanography.pdf"
HISTORICAL_DIR = ROOT.parent / "research/qwen2_ocean_qlora_project/data"
# Actual section starts, using one-based PDF pages, not printed page numbers.
SECTION_STARTS = {
    "1.1": 14, "1.2": 17, "1.3": 20, "1.4": 22,
    "2.1": 31, "2.2": 34, "2.3": 35,
    "3.1": 42, "3.2": 47, "3.3": 52,
    "4.1": 62, "4.2": 67, "4.3": 70, "4.4": 73, "4.5": 77,
    "4.6": 84, "4.7": 87, "4.8": 89, "4.9": 93, "4.10": 95, "4.11": 97,
    "5.1": 104, "5.2": 110, "5.3": 114, "5.4": 122, "5.5": 125, "5.6": 131, "5.7": 134,
    "6.1": 138, "6.2": 140, "6.3": 145, "6.4": 148, "6.5": 152,
    "7.1": 158, "7.2": 159, "7.3": 164, "7.4": 167,
    "8.1": 175, "8.2": 179, "8.3": 185, "8.4": 189, "8.5": 194,
    "9.1": 207, "9.2": 210, "9.3": 214, "9.4": 217, "9.5": 220, "9.6": 223, "9.7": 228, "9.8": 230,
    "10.1": 243, "10.2": 247, "10.3": 256, "10.4": 260,
    "11.1": 266, "11.2": 272, "11.3": 276,
    "12.1": 284, "12.2": 289, "12.3": 293, "12.4": 298, "12.5": 303, "12.6": 304,
    "13.1": 312, "13.2": 315, "13.3": 319, "13.4": 324, "13.5": 329, "13.6": 333, "13.7": 340,
    "14.1": 347, "14.2": 353, "14.3": 357,
}
DEV_SECTIONS = {"2.2", "4.10", "5.5", "6.4", "9.6", "14.3"}


def source_pages() -> dict[int, str]:
    import fitz
    with fitz.open(SOURCE_PDF) as doc:
        return {i + 1: page.get_text(sort=True) for i, page in enumerate(doc)}


def clean_page(text: str) -> str:
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if re.fullmatch(r"\d+|\d+\s*\|.*|.*\|\s*\d+", line):
            continue
        if not line or line.startswith(("A YouTube element", "Additional links for more information")):
            continue
        lines.append(line)
    return re.sub(r"\s+", " ", " ".join(lines)).strip()


def page_section(page: int) -> str:
    candidates = [(start, section) for section, start in SECTION_STARTS.items() if start <= page]
    return max(candidates)[1] if candidates else "front_matter"


def section_pages(section: str) -> list[int]:
    ordered = sorted(SECTION_STARTS.items(), key=lambda pair: pair[1])
    i = next(i for i, (s, _) in enumerate(ordered) if s == section)
    start = ordered[i][1]
    stop = ordered[i + 1][1] if i + 1 < len(ordered) else 361
    # Omit chapter title pages and long chapter summaries where the section ends.
    return list(range(start, stop))


def build_corpus(pages: dict[int, str]) -> list[dict[str, Any]]:
    chunks = []
    for page, text in pages.items():
        if not 14 <= page <= 360:
            continue
        text = clean_page(text)
        if len(text) < 120:
            continue
        # Sliding text windows preserve page provenance; overlap prevents sentence cuts.
        for j, start in enumerate(range(0, len(text), 850), start=1):
            piece = text[start:start + 1100]
            if len(piece) < 80:
                continue
            chunks.append({"id": f"WB_p{page:03d}_c{j:02d}", "text": piece,
                           "source": "Paul Webb, Introduction to Oceanography", "pdf_page": page,
                           "section": page_section(page), "license": "CC BY 4.0; see source-specific exceptions", "url": SOURCE_URL})
    return chunks


def training_inventory() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    local = sorted((ROOT / "data").rglob("*.jsonl"))
    historical = [HISTORICAL_DIR / name for name in ("ocean_qa_sample.jsonl", "mangrove_qa.jsonl", "black_current_train.jsonl", "seed_professional_single.jsonl", "seed_multiturn.jsonl")]
    inventory, examples, seen = [], [], set()
    for path in local + historical:
        if not path.exists() or "eval" in path.name:
            continue
        rows = read_jsonl(path)
        inventory.append({"path": str(path), "sha256": digest_file(path), "records": len(rows)})
        for line, row in enumerate(rows, start=1):
            messages = row.get("messages", [])
            questions = [m["content"] for m in messages if m.get("role") == "user" and isinstance(m.get("content"), str)]
            if not questions:
                questions = [str(row.get("question", row.get("instruction", "")))]
            answers = "\n".join(str(m.get("content", "")) for m in messages if m.get("role") == "assistant") or str(row.get("answer", row.get("output", "")))
            for question in questions:
                key = normalize(question + answers)
                if not question.strip() or key in seen:
                    continue
                seen.add(key)
                examples.append({"question": question, "answer": answers, "path": str(path), "line": line})
    return inventory, examples


def prepare() -> None:
    pages = source_pages()
    corpus = build_corpus(pages)
    by_section = collections.defaultdict(list)
    for chunk in corpus:
        by_section[chunk["section"]].append(chunk)
    tests, dev = [], []
    with (ROOT / "paper_eval/questions.tsv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    for index, row in enumerate(rows, start=1):
        section = row["section"]
        anchors = [x.strip() for x in row["anchors"].split(";") if x.strip()]
        candidates = by_section[section]
        evidence = []
        for anchor in anchors:
            matches = [c for c in candidates if normalize(anchor) in normalize(c["text"])]
            if not matches:
                raise ValueError(f"Source anchor missing in section {section}: {anchor}; {row['question']}")
            evidence.append({"chunk_id": matches[0]["id"], "pdf_page": matches[0]["pdf_page"], "anchor": anchor, "text": matches[0]["text"]})
        split = "dev" if section in DEV_SECTIONS else "test"
        record = {"id": f"OCEAN-{split.upper()}-{index:04d}", "source_group": f"webb_section_{section}", "section": section,
                  "task": row["task"], "mode": row["mode"], "question": row["question"],
                  "key_points": [p.strip() for p in row["key_points"].split(";") if p.strip()], "evidence": evidence,
                  "reference_pages": section_pages(section), "split": split, "language": "zh",
                  "gold_status": "AI-authored source-grounded draft; domain expert approval pending",
                  "source_attribution": ATTRIBUTION}
        (dev if split == "dev" else tests).append(record)
    validate_questions(tests + dev)
    if {r["source_group"] for r in tests} & {r["source_group"] for r in dev}:
        raise ValueError("Development and test source groups overlap")
    write_jsonl(DATA / "corpus.jsonl", corpus)
    write_jsonl(DATA / "test.jsonl", tests)
    write_jsonl(DATA / "dev.jsonl", dev)
    inventory, training = training_inventory()
    write_json(DATA / "manifest.json", {
        "version": "0.1-draft", "source_pdf": str(SOURCE_PDF), "source_pdf_sha256": digest_file(SOURCE_PDF),
        "source_attribution": ATTRIBUTION, "source_url": SOURCE_URL,
        "test_questions": len(tests), "dev_questions": len(dev), "source_groups_test": len({r['source_group'] for r in tests}),
        "corpus_chunks": len(corpus), "corpus_sha256": digest_file(DATA / "corpus.jsonl"),
        "test_sha256": digest_file(DATA / "test.jsonl"), "dev_sha256": digest_file(DATA / "dev.jsonl"),
        "training_inventory": inventory, "unique_training_pairs_audited": len(training),
        "exclusions": ["docs/seed_from_training.jsonl", "all training QA files", "all test/dev questions, reference answers and scoring rubrics"],
        "independence_boundary": "Development/test source sections are disjoint. Known SFT questions and answers are audited. Historical source-document lineage and base-model pretraining exposure cannot be reconstructed; do not claim unseen-source or contamination-free generalization.",
        "human_gold_review": "pending", "sampling": "purposive source-grounded coverage, not a random sample of all ocean questions",
        "rebuild_rule": "Any edit after freezing changes the dataset hash; never tune on locked test outputs."
    })
    (DATA / "ATTRIBUTION.txt").write_text(ATTRIBUTION + "\n\nThe evidence passages are English textbook extracts. Questions and draft reference points are newly authored in Chinese. Original textbook figures are not redistributed.\n", encoding="utf-8")
    with (DATA / "gold_review.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "split", "section", "question", "draft_key_points", "pdf_pages_1_based", "decision", "corrected_key_points", "reviewer", "notes"])
        for row in tests + dev:
            writer.writerow([row["id"], row["split"], row["section"], row["question"], ";".join(row["key_points"]), ",".join(map(str, row["reference_pages"])), "", "", "", ""])
    print(f"Prepared {len(tests)} test / {len(dev)} dev questions, {len(corpus)} source-only chunks; expert gold review pending.")


def audit(semantic: bool = True) -> None:
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from .common import load_protocol
    protocol = load_protocol()
    questions = read_jsonl(DATA / "test.jsonl")
    inventory, training = training_inventory()
    train_questions = [r["question"] for r in training]
    train_content = [r["question"] + "\n" + r["answer"] for r in training]
    test_questions = [r["question"] for r in questions]
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), sublinear_tf=True)
    vectors = vectorizer.fit_transform(train_questions + train_content + test_questions)
    offset = 2 * len(training)
    sims = (vectors[offset:] @ vectors[:len(training)].T).toarray()
    content_sims = (vectors[offset:] @ vectors[len(training):offset].T).toarray()
    embed_sims = None
    if semantic:
        from .retrieval import Retriever
        retriever = Retriever(DATA / "corpus.jsonl", ROOT / "models/embedding_minilm", ROOT / "cache/paper_eval")
        # Long training answers are windowed; audit windows cannot enter the RAG corpus.
        windows = [(i, text[start:start + 700]) for i, text in enumerate(train_content) for start in range(0, len(text), 500)]
        train_vectors = retriever.encode([text for _, text in windows])
        test_vectors = retriever.encode(test_questions)
        embed_sims = test_vectors @ train_vectors.T
    exact = {normalize(t): i for i, t in enumerate(train_questions)}
    rows = []
    for i, question in enumerate(questions):
        nearest = int(np.argmax(sims[i]))
        content_nearest = int(np.argmax(content_sims[i]))
        semantic_nearest = int(np.argmax(embed_sims[i])) if embed_sims is not None else None
        reasons = []
        if normalize(question["question"]) in exact:
            reasons.append("exact_training_question")
        if float(sims[i, nearest]) >= protocol["training_audit"]["char_tfidf_review_threshold"]:
            reasons.append("lexical_near_training_question")
        if float(content_sims[i, content_nearest]) >= protocol["training_audit"]["char_tfidf_review_threshold"]:
            reasons.append("lexical_near_training_content")
        if embed_sims is not None and float(embed_sims[i, semantic_nearest]) >= protocol["training_audit"]["embedding_review_threshold"]:
            reasons.append("semantic_near_training_question")
        rows.append({"id": question["id"], "source_group": question["source_group"], "question": question["question"],
                     "automatic_status": "exclude" if "exact_training_question" in reasons else ("review_required" if reasons else "no_automatic_flag"),
                     "flags": reasons, "lexical_similarity": float(sims[i, nearest]), "nearest_training_question": training[nearest],
                     "content_similarity": float(content_sims[i, content_nearest]), "nearest_training_content": training[content_nearest],
                     "semantic_similarity": float(embed_sims[i, semantic_nearest]) if embed_sims is not None else None,
                     "nearest_semantic_training_question": training[windows[semantic_nearest][0]] if semantic_nearest is not None else None,
                     "human_independence_review": "pending"})
    # Near-duplicates inside the test set are warnings, not extra independent units.
    test_matrix = vectors[offset:]
    internal = (test_matrix @ test_matrix.T).toarray()
    pairs = [{"a": questions[i]["id"], "b": questions[j]["id"], "similarity": float(internal[i, j])}
             for i in range(len(questions)) for j in range(i) if internal[i, j] >= 0.65]
    write_json(DATA / "overlap_audit.json", {"inventory": inventory, "audited_unique_training_pairs": len(training),
               "status_counts": dict(collections.Counter(r["automatic_status"] for r in rows)), "semantic_audit": semantic,
               "test_sha256": digest_file(DATA / "test.jsonl"), "test_near_duplicate_pairs": pairs, "items": rows,
               "warning": "No automated flag is not proof of unseen knowledge. Human semantic/source review remains necessary. Historical source lineage is incomplete."})
    with (DATA / "overlap_review.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "question", "automatic_status", "lexical_similarity", "semantic_similarity", "nearest_training_question", "human_decision", "reviewer", "reason"])
        for row in rows:
            nearest = row["nearest_semantic_training_question"] or row["nearest_training_question"]
            writer.writerow([row["id"], row["question"], row["automatic_status"], row["lexical_similarity"], row["semantic_similarity"], nearest["question"], "", "", ""])
    print(json.dumps({"audited_pairs": len(training), "status_counts": dict(collections.Counter(r['automatic_status'] for r in rows)), "test_near_duplicate_pairs": len(pairs)}, ensure_ascii=False))

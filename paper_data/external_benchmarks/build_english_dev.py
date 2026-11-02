"""Select an English development pool using known article and passage separation."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent


def normalized(text: str) -> str:
    return re.sub(r"[^\w]", "", text.casefold())


def main() -> None:
    raw = HERE / "raw/pira/data"
    test_path = next(raw.glob("test-*.parquet"))
    dev_path = next(raw.glob("validation-*.parquet"))
    test = pd.read_parquet(test_path)
    dev = pd.read_parquet(dev_path)
    test_eids = set(test.eid_article_scopus.dropna().astype(str))
    test_texts = set(test["abstract"].map(normalized))
    selected = dev[(dev["corpus"] == 1) & dev.eid_article_scopus.notna()
                   & ~dev.eid_article_scopus.astype(str).isin(test_eids)
                   & ~dev["abstract"].map(normalized).isin(test_texts)]
    records = []
    for row in selected.to_dict("records"):
        validated = row["answer_en_validate"]
        records.append({"id": "PIRA-DEV-" + row["id_qa"], "original_id": row["id_qa"], "split": "development", "language": "en",
            "question": row["question_en_origin"], "reference_answer_origin": row["answer_en_origin"],
            "reference_answer_validate": validated if isinstance(validated, str) else None,
            "supporting_text": row["abstract"], "source_article_eid": row["eid_article_scopus"],
            "source_group": "PIRA-EID-" + row["eid_article_scopus"],
            "review_status": "Published reference retained; local factual and suitability review pending"})
    target = HERE / "pira_dev_article_disjoint_en_101.jsonl"
    target.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {"questions": len(records), "unique_known_article_eids": selected.eid_article_scopus.nunique(),
        "unique_normalized_passages": selected["abstract"].map(normalized).nunique(),
        "selection_rules": ["official validation split", "corpus 1 only (scientific abstracts)", "article EID present",
                            "article EID absent from official test", "normalized supporting passage absent from official test"],
        "reason_for_excluding_corpus_2": "UN report-level source identity cannot be separated reliably from released metadata",
        "input_validation_sha256": sha(dev_path), "held_out_test_sha256": sha(test_path), "output_sha256": sha(target),
        "boundary": "Disjoint only by published EID and normalized passage. Not proof of unseen pretraining or complete source-document independence. This is a development pool, never add its QA to this run's SFT data. Do not select training settings on the official test."}
    (HERE / "english_dev_selection_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ["questions", "unique_known_article_eids", "unique_normalized_passages"]}))


if __name__ == "__main__":
    main()

"""Download pinned official public data for inspection; does not train or run models."""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCES = {
    "pira": {"repo": "paulopirozelli/pira", "revision": "756f73c8a52e2b0fbdc8f42569e1e8c6f3865a49", "license_card": "CC BY 4.0",
             "files": ["README.md", "data/test-00000-of-00001-41cee92681c49a98.parquet", "data/train-00000-of-00001-7cb516c53d508ffa.parquet", "data/validation-00000-of-00001-ce5e7ad4b67feb65.parquet", "mcqa/test-00000-of-00001-1c607c8177ebc2d9.parquet"]},
    "oceanbenchmark_2026": {"repo": "zjunlp/OceanBenchmark", "revision": "640b00c5f84c4bb42425ce8214762fd2e5745216", "license_card": "MIT (dataset card)",
                            "files": ["README.md", "Science_Text/data.csv", "Science_Text/qa.json"]},
    "oceanbench_2024": {"repo": "zjunlp/OceanBench", "revision": "7571d9879da80fb34d0e53be7469599e6158a184", "license_card": "MIT (dataset card)",
                          "files": ["README.md", "OceanBench.txt"]},
}


def fetch(name: str, config: dict, file: str) -> dict:
    url = f"https://huggingface.co/datasets/{config['repo']}/resolve/{config['revision']}/{urllib.parse.quote(file, safe='/')}"
    path = ROOT / "raw" / name / file
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        request = urllib.request.Request(url, headers={"User-Agent": "OceanQA-academic-dataset-inspection"})
        with urllib.request.urlopen(request, timeout=90) as response:
            content = response.read(20 * 1024 * 1024 + 1)
        if len(content) > 20 * 1024 * 1024:
            raise ValueError("Unexpectedly large file")
        path.write_bytes(content)
    return {"dataset": name, "repo": config["repo"], "revision": config["revision"], "url": url,
            "local_file": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "license_card": config["license_card"]}


def main():
    records, errors = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        jobs = {pool.submit(fetch, name, config, file): (name, file) for name, config in SOURCES.items() for file in config["files"]}
        for future in concurrent.futures.as_completed(jobs):
            name, file = jobs[future]
            try:
                row = future.result()
                records.append(row)
                print(name, file, row["bytes"], flush=True)
            except Exception as exc:
                errors.append({"dataset": name, "file": file, "error_type": type(exc).__name__})
                print("FAILED", name, file, type(exc).__name__, flush=True)
    (ROOT / "download_manifest.json").write_text(json.dumps({"retrieved_at_utc": datetime.now(timezone.utc).isoformat(), "files": records, "errors": errors}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

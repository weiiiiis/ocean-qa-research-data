"""Finish the already frozen local experiment after interrupted desktop turns.

This supervisor waits for an existing run and never launches a second GPU job.
It changes no model, data, prompt, parsing rule or evaluation setting.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
RUN_NAME = "en_test_openqa_frozen_v1_20261017"
RUN = ROOT / "outputs/paper_eval" / RUN_NAME
REPORT = ROOT / "outputs/paper_eval/en_test_openqa_frozen_v1_report_20261017"
REVIEW = ROOT / "outputs/paper_eval/en_test_blind40_frozen_v1_20261017"
STATUS = ROOT / "outputs/frozen_english_eval_job_status.json"


def status(phase: str, **extra) -> None:
    item = {"phase": phase, "pid": os.getpid(), "updated_at_utc": datetime.now(timezone.utc).isoformat(), **extra}
    temporary = STATUS.with_suffix(".tmp")
    temporary.write_text(json.dumps(item, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(STATUS)


def other_pipeline_processes() -> list[int]:
    active = []
    for process in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            command = " ".join(process.info["cmdline"] or [])
            if process.pid != os.getpid() and "python" in (process.info["name"] or "").lower():
                if RUN_NAME in command and "paper_eval." in command:
                    active.append(process.pid)
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            pass
    return active


def complete(path: Path) -> bool:
    return path.exists() and bool(json.loads(path.read_text(encoding="utf-8")).get("complete"))


def execute(phase: str, arguments: list[str], log_name: str) -> None:
    status(phase, arguments=arguments)
    environment = dict(os.environ, PYTHONIOENCODING="utf-8", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
    with (ROOT / "outputs" / log_name).open("a", encoding="utf-8") as log:
        result = subprocess.run([sys.executable, "-u", *arguments], cwd=str(ROOT), env=environment,
                                stdout=log, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(f"{phase} failed with exit {result.returncode}; preserved log: {log_name}")


def main() -> None:
    # Windows releases this advisory lock when the process exits or is killed.
    import msvcrt
    lock_path = ROOT / "outputs/frozen_english_eval_job.lock"
    with lock_path.open("a+b") as lock:
        if lock.tell() == 0:
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            return
        try:
            status("waiting_for_existing_pipeline")
            quiet_since = None
            while True:
                active = other_pipeline_processes()
                if active:
                    quiet_since = None
                    status("waiting_for_existing_pipeline", active_pids=active)
                elif quiet_since is None:
                    quiet_since = time.monotonic()
                elif time.monotonic() - quiet_since >= 15:
                    break
                time.sleep(3)
            if not complete(RUN / "completion.json"):
                execute("resuming_frozen_openqa", ["-m", "paper_eval.english", "--dataset",
                    "paper_data/external_benchmarks/official_openqa_en_227.jsonl", "--corpus",
                    "paper_data/external_benchmarks/corpus_pira_chunks_en.jsonl", "--out", str(RUN),
                    "--task", "openqa", "--protocol", "paper_data/frozen_english_v1_20261017/protocol.json"],
                    "en_test_openqa_frozen_v1_detached.log")
            if not complete(RUN / "completion.json"):
                raise RuntimeError("Raw answers are incomplete; statistics were not published")
            if not REPORT.exists():
                execute("reporting_frozen_openqa", ["-m", "paper_eval.official_english_report",
                    "--run", str(RUN), "--out", str(REPORT)], "en_test_openqa_frozen_v1_detached_report.log")
            if not complete(REPORT / "analysis.json"):
                raise RuntimeError("Report incomplete; existing artifacts retained")
            if not REVIEW.exists():
                execute("exporting_blind40", ["-m", "paper_eval.export_official_blind_subset",
                    "--run", str(RUN), "--plan", "paper_data/external_benchmarks/pretest_review_plan_20261017",
                    "--out", str(REVIEW)], "en_test_blind40_frozen_v1_detached.log")
            if not (REVIEW / "review_export_manifest.json").exists():
                raise RuntimeError("Blind export incomplete; existing artifacts retained")
            status("complete", run=str(RUN), report=str(REPORT), expert_review=str(REVIEW / "expert_review"))
        except Exception as error:
            status("failed", error=str(error))
            raise
        finally:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


if __name__ == "__main__":
    main()

"""Finish delivery for the current finite experiment when its artifacts exist."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "outputs/paper_eval"
STATUS = ROOT / "outputs/frozen_delivery_status.json"


def status(phase: str, **extra) -> None:
    value = {"phase": phase, "pid": os.getpid(), "updated_at_utc": datetime.now(timezone.utc).isoformat(), **extra}
    temporary = STATUS.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(STATUS)


def complete(path: Path, key: str = "complete") -> bool:
    try:
        return bool(json.loads(path.read_text(encoding="utf-8")).get(key))
    except (FileNotFoundError, json.JSONDecodeError):
        return False


def main() -> None:
    deadline = time.monotonic() + 3 * 60 * 60
    try:
        status("waiting_for_current_frozen_experiment")
        while time.monotonic() < deadline:
            ready = (complete(EVAL / "en_test_mcqa_frozen_v1_report_20261017/analysis.json")
                     and complete(EVAL / "en_test_openqa_frozen_v1_report_20261017/analysis.json")
                     and complete(EVAL / "en_test_blind40_frozen_v1_20261017/review_export_manifest.json", "subset_complete"))
            if ready:
                break
            time.sleep(5)
        else:
            raise RuntimeError("Current experiment did not finish within the finite three-hour delivery window")
        summary = ROOT / "paper_data/正式实验结果汇总.md"
        packet = EVAL / "teacher_review_packets_frozen_v1_20261017/package_manifest.json"
        if not (summary.exists() and packet.exists()):
            status("packaging_completed_experiment")
            result = subprocess.run([sys.executable, "-u", "-m", "paper_eval.package_frozen_delivery"],
                                    cwd=str(ROOT), check=False)
            if result.returncode:
                raise RuntimeError(f"Delivery helper failed: exit={result.returncode}")
        if not summary.exists() or not packet.exists():
            raise RuntimeError("Delivery did not produce both expected outputs")
        status("complete", summary=str(summary), packets=str(packet.parent))
    except Exception as exc:
        status("failed", error=str(exc))
        raise


if __name__ == "__main__":
    main()

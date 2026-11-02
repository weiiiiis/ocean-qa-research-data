from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Source-grounded ocean QA paper evaluation; human judgments remain blank until reviewed.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare", help="Build draft questions, source-only corpus and gold review sheet")
    audit = commands.add_parser("audit", help="Audit known training questions AND answers")
    audit.add_argument("--no-semantic", action="store_true")
    run = commands.add_parser("run", help="Run four controlled local conditions")
    run.add_argument("--out", required=True, type=Path)
    run.add_argument("--split", choices=["dev", "test"], default="test")
    run.add_argument("--limit", type=int, default=0, help="0 means all; source-group-balanced selection")
    run.add_argument("--protocol", type=Path)
    run.add_argument("--skip-probes", action="store_true")
    blind = commands.add_parser("blind", help="Create two independent offline blind-review packets")
    blind.add_argument("--run", required=True, type=Path)
    report = commands.add_parser("report", help="Summarize real outputs; absent human scores stay missing")
    report.add_argument("--run", required=True, type=Path)
    report.add_argument("--ratings", nargs="*", type=Path, default=[])
    args = parser.parse_args()
    if args.command == "prepare":
        from .data import prepare
        prepare()
    elif args.command == "audit":
        from .data import audit as do_audit
        do_audit(not args.no_semantic)
    elif args.command == "run":
        from .inference import run
        run(args.out.resolve(), args.split, args.limit, args.protocol, args.skip_probes)
    elif args.command == "blind":
        from .review import make_packets
        make_packets(args.run.resolve())
    elif args.command == "report":
        from .report import report as make_report
        make_report(args.run.resolve(), args.ratings)


if __name__ == "__main__":
    main()

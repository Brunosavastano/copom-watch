from __future__ import annotations

import argparse
import logging
import sys

from copom_tone_index.econometric_review import run_econometric_review
from copom_tone_index.focus import (
    fetch_focus_command,
    focus_coverage_command,
    import_focus_snapshot_command,
    rebuild_focus_revisions_command,
)
from copom_tone_index.pipeline import run_pipeline, validate_existing_outputs


def configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="COPOM Tone Index pipeline.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run-pipeline", help="Run full ingestion, scoring and reporting pipeline.")
    run_parser.add_argument("--months", type=int, default=None, help="Operational window in months.")
    run_parser.add_argument("--copom-quantity", type=int, default=None, help="Number of meetings requested from BCB lists.")
    run_parser.add_argument("--refresh-focus", action="store_true", help="Refresh Focus observations from OData.")
    run_parser.add_argument(
        "--use-llm",
        choices=["auto", "never", "always"],
        default="auto",
        help="Use optional LLM layer when available.",
    )

    for command in ["ingest", "score", "build-notes"]:
        command_parser = subparsers.add_parser(command, help=f"Alias for run-pipeline ({command}).")
        command_parser.add_argument("--months", type=int, default=None)
        command_parser.add_argument("--copom-quantity", type=int, default=None)
        command_parser.add_argument("--refresh-focus", action="store_true")
        command_parser.add_argument("--use-llm", choices=["auto", "never", "always"], default="auto")

    subparsers.add_parser("validate", help="Validate existing DuckDB outputs.")

    review_parser = subparsers.add_parser("review-econometrics", help="Run econometric accuracy review.")
    review_parser.add_argument("--manual-sample-size", type=int, default=80, help="Sentence sample size for manual audit.")
    review_parser.add_argument("--min-formal-obs", type=int, default=30, help="Minimum observations for formal inference flags.")
    review_parser.add_argument("--min-exploratory-obs", type=int, default=8, help="Minimum observations to estimate a diagnostic regression.")

    fetch_focus_parser = subparsers.add_parser("fetch-focus", help="Fetch event-oriented Focus observations from BCB OData.")
    fetch_focus_parser.add_argument("--months", type=int, default=None, help="Operational window in months.")

    rebuild_focus_parser = subparsers.add_parser("rebuild-focus-revisions", help="Rebuild Focus revisions from stored observations.")
    rebuild_focus_parser.add_argument("--months", type=int, default=None, help="Optional operational window in months.")

    subparsers.add_parser("focus-coverage", help="Write Focus coverage diagnostics.")

    import_focus_parser = subparsers.add_parser("import-focus-snapshot", help="Import official Focus snapshot fallback.")
    import_focus_parser.add_argument("--path", required=True, help="CSV/TXT/XLS/XLSX snapshot path.")
    import_focus_parser.add_argument("--source-date", required=True, help="Snapshot observation date (YYYY-MM-DD).")

    args = parser.parse_args(argv)
    configure_logging(args.verbose)
    if args.command in {"run-pipeline", "ingest", "score", "build-notes"}:
        result = run_pipeline(
            months=args.months,
            use_llm=args.use_llm,
            copom_quantity=args.copom_quantity,
            refresh_focus=args.refresh_focus,
        )
        print(f"Database: {result.database}")
        print(f"Validation errors: {result.validation_errors}; warnings: {result.validation_warnings}")
        return 1 if result.validation_errors else 0
    if args.command == "validate":
        result = validate_existing_outputs()
        print(f"Database: {result.database}")
        print(f"Validation errors: {result.validation_errors}; warnings: {result.validation_warnings}")
        return 1 if result.validation_errors else 0
    if args.command == "review-econometrics":
        result = run_econometric_review(
            manual_sample_size=args.manual_sample_size,
            min_formal_obs=args.min_formal_obs,
            min_exploratory_obs=args.min_exploratory_obs,
        )
        print(f"Status: {result.status}")
        print(f"Report: {result.report_path}")
        print(f"Output dir: {result.output_dir}")
        print(f"Estimated regressions: {result.valid_regressions}; blocked: {result.blocked_regressions}")
        return 0
    if args.command == "fetch-focus":
        result = fetch_focus_command(months=args.months)
        print(f"Focus observations: {result.observations}")
        print(f"Focus revisions: {result.revisions}")
        print(f"Coverage status: {result.coverage_status}")
        print(f"Output: {result.output_path}")
        return 0
    if args.command == "rebuild-focus-revisions":
        result = rebuild_focus_revisions_command(months=args.months)
        print(f"Focus observations: {result.observations}")
        print(f"Focus revisions: {result.revisions}")
        print(f"Coverage status: {result.coverage_status}")
        print(f"Output: {result.output_path}")
        return 0
    if args.command == "focus-coverage":
        result = focus_coverage_command()
        print(f"Focus revisions: {result.revisions}")
        print(f"Coverage status: {result.coverage_status}")
        print(f"Output: {result.output_path}")
        return 0
    if args.command == "import-focus-snapshot":
        result = import_focus_snapshot_command(args.path, args.source_date)
        print(f"Focus observations: {result.observations}")
        print(f"Focus revisions: {result.revisions}")
        print(f"Coverage status: {result.coverage_status}")
        print(f"Output: {result.output_path}")
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())

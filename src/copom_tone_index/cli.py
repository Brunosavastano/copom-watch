from __future__ import annotations

import argparse
import logging
import sys

from copom_tone_index.econometric_review import run_econometric_review
from copom_tone_index.focus import (
    fetch_focus_command,
    focus_coverage_command,
    focus_v21_audit_command,
    focus_v21_refresh_command,
    import_focus_snapshot_command,
    rebuild_focus_revisions_command,
)
from copom_tone_index.market import (
    derive_decision_expectations_command,
    fetch_public_market_command,
    import_decision_expectations_command,
    import_market_csv_command,
    public_market_coverage_command,
    run_event_study_command,
)
from copom_tone_index.pipeline import run_pipeline, validate_existing_outputs
from copom_tone_index.semantic import build_semantic_index_command, semantic_ask_command, semantic_search_command
from copom_tone_index.v2 import (
    v2_audit_command,
    v2_backfill_command,
    v2_benchmark_baseline_command,
    v2_calibrate_command,
    v2_freeze_release_command,
    v2_import_labels_command,
    v2_redline_command,
    v2_report_command,
    v2_review_remaining_errors_command,
    v2_run_all_command,
    v2_score_command,
    v2_train_supervised_command,
)
from copom_tone_index.v2_health import export_label_sample_command, v2_health_check_command
from copom_tone_index.v21 import v21_build_event_panel_command, v21_freeze_release_command, v21_health_command
from copom_tone_index.v22 import package_public_data_command, v22_freeze_release_command, v22_health_command


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

    v2_parser = subparsers.add_parser("v2", help="COPOM Watch V2 commands.")
    v2_subparsers = v2_parser.add_subparsers(dest="v2_command")
    v2_backfill = v2_subparsers.add_parser("backfill", help="Fetch and version full COPOM document history.")
    v2_backfill.add_argument("--quantity", type=int, default=None, help="Number of meetings requested from BCB lists.")
    v2_subparsers.add_parser("score", help="Build V2 sentence, document, meeting and subindex scores.")
    v2_subparsers.add_parser("calibrate", help="Rebuild fixed calibration windows.")
    v2_redline = v2_subparsers.add_parser("redline", help="Build V2 textual redline.")
    v2_redline.add_argument("--force", action="store_true", help="Recompute redline even when current coverage is already available.")
    v2_subparsers.add_parser("audit", help="Build V2 model audit and manual review sample.")
    v2_subparsers.add_parser("benchmark-baseline", help="Benchmark the deterministic V2 baseline against accepted human labels.")
    v2_review_errors = v2_subparsers.add_parser("review-remaining-errors", help="Classify remaining likely baseline errors without tuning rules.")
    v2_review_errors.add_argument("--limit", type=int, default=177, help="Maximum likely baseline errors to review.")
    v2_subparsers.add_parser("train-supervised", help="Train an experimental supervised benchmark model from accepted human labels.")
    v2_freeze = v2_subparsers.add_parser("freeze-release", help="Freeze V2.0 release artifacts and manifest.")
    v2_freeze.add_argument("--version", default="v2.0.4-holdout-stance-hardened", help="Release version name.")
    v2_import_labels = v2_subparsers.add_parser("import-labels", help="Import reviewed V2 labels from CSV.")
    v2_import_labels.add_argument("--path", required=True, help="CSV path exported from the manual label sample.")
    v2_import_labels.add_argument("--label-source", default="human", help="Label source metadata.")
    v2_export_sample = v2_subparsers.add_parser("export-label-sample", help="Export a stratified V2 human review sample.")
    v2_export_sample.add_argument("--n", type=int, default=300, help="Maximum number of sentences.")
    v2_export_sample.add_argument("--out", default="data/labels/review_sample_001.csv", help="Output CSV path.")
    v2_subparsers.add_parser("health-check", help="Build V2 acceptance health-check report.")
    v2_focus_refresh = v2_subparsers.add_parser("focus-refresh", help="Refresh Focus V2.1 vintages and event features.")
    v2_focus_refresh.add_argument("--months", type=int, default=400, help="Focus window in months.")
    v2_subparsers.add_parser("focus-audit", help="Build Focus V2.1 coverage report from stored observations.")
    v2_subparsers.add_parser("build-event-panel", help="Build V2.1 decision, Focus and market event panel.")
    v2_subparsers.add_parser("v21-health", help="Build V2.1 analytical acceptance report.")
    v21_freeze = v2_subparsers.add_parser("freeze-v21", help="Freeze V2.1 analytical release artifacts and manifest.")
    v21_freeze.add_argument("--version", default="v2.1-public-focus-market-acceptance", help="V2.1 release version name.")
    v2_subparsers.add_parser("package-public-data", help="Build reduced public DuckDB package for Streamlit deploy.")
    v2_subparsers.add_parser("v22-health", help="Build V2.2 product/deploy acceptance report.")
    v22_freeze = v2_subparsers.add_parser("freeze-v22", help="Freeze V2.2 product/RAG/deploy release artifacts and manifest.")
    v22_freeze.add_argument("--version", default="v2.2-product-rag-deploy", help="V2.2 release version name.")
    v2_report = v2_subparsers.add_parser("report", help="Generate a V2 HTML report.")
    v2_report.add_argument("--meeting", type=int, default=None, help="Meeting number. Defaults to latest.")
    v2_run_all = v2_subparsers.add_parser("run-all", help="Run V2 backfill, score, redline, audit and report.")
    v2_run_all.add_argument("--quantity", type=int, default=None, help="Number of meetings requested from BCB lists.")
    v2_run_all.add_argument("--refresh-backfill", action="store_true", help="Force a fresh COPOM document fetch instead of reusing an existing broad V2 backfill.")

    market_parser = subparsers.add_parser("market", help="Optional market data commands.")
    market_subparsers = market_parser.add_subparsers(dest="market_command")
    market_import = market_subparsers.add_parser("import-csv", help="Import market observations from CSV.")
    market_import.add_argument("--path", required=True, help="CSV path with asset, timestamp/date and value columns.")
    market_import.add_argument("--source", default="user_csv")
    market_import.add_argument("--data-access-tier", default="USER_CSV")
    market_import.add_argument("--license-note", default="")
    decision_import = market_subparsers.add_parser("import-decision-expectations", help="Import decision expectations from CSV.")
    decision_import.add_argument("--path", required=True, help="CSV path with meeting_id, as_of_timestamp and expected_selic_change_bps.")
    decision_import.add_argument("--source", default="user_csv")
    decision_import.add_argument("--data-access-tier", default="USER_CSV")
    decision_import.add_argument("--license-note", default="")
    market_fetch_public = market_subparsers.add_parser("fetch-public", help="Fetch public market data from BCB/PTAX/ANBIMA adapters.")
    market_fetch_public.add_argument("--sources", default="bcb-sgs,ptax,anbima", help="Comma-separated sources: bcb-sgs, ptax, anbima, b3.")
    market_fetch_public.add_argument("--months", type=int, default=400, help="Historical window in months.")
    decision_derive = market_subparsers.add_parser("derive-decision-expectations", help="Derive public/proxy decision expectations.")
    decision_derive.add_argument("--method", default="public", help="Derivation method. Default: public.")
    market_subparsers.add_parser("public-coverage", help="Report public market and decision expectation coverage.")
    market_subparsers.add_parser("event-study", help="Build market event-study windows.")

    semantic_parser = subparsers.add_parser("semantic", help="Local semantic search support.")
    semantic_subparsers = semantic_parser.add_subparsers(dest="semantic_command")
    semantic_build = semantic_subparsers.add_parser("build-index", help="Build local semantic chunks with citations.")
    semantic_build.add_argument("--method", choices=["token", "tfidf"], default="tfidf", help="Retrieval method stored in semantic chunks.")
    semantic_search = semantic_subparsers.add_parser("search", help="Search local semantic chunks with mandatory citations.")
    semantic_search.add_argument("--query", required=True, help="Portuguese search query.")
    semantic_search.add_argument("--top-n", type=int, default=10, help="Maximum number of cited matches.")
    semantic_search.add_argument("--method", choices=["token", "tfidf"], default=None, help="Override retrieval method.")
    semantic_ask = semantic_subparsers.add_parser("ask", help="Answer using local extractive RAG with mandatory citations.")
    semantic_ask.add_argument("--query", required=True, help="Portuguese question.")
    semantic_ask.add_argument("--top-n", type=int, default=8, help="Maximum citations.")
    semantic_ask.add_argument("--method", choices=["token", "tfidf"], default="tfidf", help="Retrieval method.")

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
    if args.command == "v2":
        if args.v2_command == "backfill":
            result = v2_backfill_command(quantity=args.quantity)
        elif args.v2_command == "score":
            result = v2_score_command()
        elif args.v2_command == "calibrate":
            result = v2_calibrate_command()
        elif args.v2_command == "redline":
            result = v2_redline_command(force=args.force)
        elif args.v2_command == "audit":
            result = v2_audit_command()
        elif args.v2_command == "benchmark-baseline":
            benchmark = v2_benchmark_baseline_command()
            print(f"Status: {benchmark.status}")
            print(f"Database: {benchmark.database}")
            print(f"Rows: {benchmark.rows}")
            print(f"Warnings: {benchmark.warnings}")
            print(f"Run ID: {benchmark.run_id}")
            print(f"Output dir: {benchmark.output_dir}")
            print(f"Report: {benchmark.report_path}")
            return 1 if benchmark.status == "fail" else 0
        elif args.v2_command == "review-remaining-errors":
            result = v2_review_remaining_errors_command(limit=args.limit)
        elif args.v2_command == "train-supervised":
            result = v2_train_supervised_command()
        elif args.v2_command == "freeze-release":
            result = v2_freeze_release_command(version=args.version)
        elif args.v2_command == "import-labels":
            result = v2_import_labels_command(args.path, label_source=args.label_source)
        elif args.v2_command == "export-label-sample":
            sample = export_label_sample_command(n=args.n, out=args.out)
            print(f"Rows: {sample.rows}")
            print(f"Output: {sample.output_path}")
            print(f"Codebook: {sample.codebook_path}")
            return 0
        elif args.v2_command == "health-check":
            health = v2_health_check_command()
            print(f"Status: {health.status}")
            print(f"Warnings: {health.warnings}; errors: {health.errors}")
            print(f"JSON: {health.json_path}")
            print(f"HTML: {health.html_path}")
            return 1 if health.status == "fail" else 0
        elif args.v2_command == "focus-refresh":
            focus = focus_v21_refresh_command(months=args.months)
            print(f"Focus vintages: {focus.vintages}")
            print(f"Focus event features: {focus.event_features}")
            print(f"Coverage status: {focus.coverage_status}")
            print(f"Output: {focus.output_path}")
            return 0
        elif args.v2_command == "focus-audit":
            focus = focus_v21_audit_command()
            print(f"Focus vintages: {focus.vintages}")
            print(f"Focus event features: {focus.event_features}")
            print(f"Coverage status: {focus.coverage_status}")
            print(f"Output: {focus.output_path}")
            return 0
        elif args.v2_command == "build-event-panel":
            result = v21_build_event_panel_command()
        elif args.v2_command == "v21-health":
            health = v21_health_command()
            print(f"Status: {health.status}")
            print(f"Warnings: {health.warnings}; errors: {health.errors}")
            print(f"Rows: {health.rows}")
            print(f"JSON: {health.json_path}")
            print(f"HTML: {health.html_path}")
            return 1 if health.status == "fail" else 0
        elif args.v2_command == "freeze-v21":
            result = v21_freeze_release_command(version=args.version)
        elif args.v2_command == "package-public-data":
            result = package_public_data_command()
        elif args.v2_command == "v22-health":
            health = v22_health_command()
            print(f"Status: {health.status}")
            print(f"Warnings: {health.warnings}; errors: {health.errors}")
            print(f"JSON: {health.json_path}")
            print(f"HTML: {health.html_path}")
            return 1 if health.status == "fail" else 0
        elif args.v2_command == "freeze-v22":
            result = v22_freeze_release_command(version=args.version)
        elif args.v2_command == "report":
            result = v2_report_command(meeting=args.meeting)
        elif args.v2_command == "run-all":
            result = v2_run_all_command(quantity=args.quantity, refresh_backfill=args.refresh_backfill)
        else:
            v2_parser.print_help()
            return 0
        print(f"Status: {result.status}")
        print(f"Database: {result.database}")
        print(f"Rows: {result.rows}")
        print(f"Run ID: {result.run_id}")
        print(f"Output: {result.output_path}")
        return 1 if result.status == "fail" else 0
    if args.command == "market":
        if args.market_command == "import-csv":
            result = import_market_csv_command(
                args.path,
                source=args.source,
                data_access_tier=args.data_access_tier,
                license_note=args.license_note,
            )
        elif args.market_command == "import-decision-expectations":
            result = import_decision_expectations_command(
                args.path,
                source=args.source,
                data_access_tier=args.data_access_tier,
                license_note=args.license_note,
            )
        elif args.market_command == "fetch-public":
            result = fetch_public_market_command(sources=args.sources, months=args.months)
        elif args.market_command == "derive-decision-expectations":
            result = derive_decision_expectations_command(method=args.method)
        elif args.market_command == "public-coverage":
            result = public_market_coverage_command()
        elif args.market_command == "event-study":
            result = run_event_study_command()
        else:
            market_parser.print_help()
            return 0
        print(f"Status: {result.status}")
        print(f"Database: {result.database}")
        print(f"Rows: {result.rows}")
        print(f"Output: {result.output_path}")
        return 0
    if args.command == "semantic":
        if args.semantic_command == "build-index":
            result = build_semantic_index_command(method=args.method)
        elif args.semantic_command == "search":
            result = semantic_search_command(args.query, top_n=args.top_n, method=args.method)
            print(f"Status: {result.status}")
            print(f"Database: {result.database}")
            print(f"Rows: {result.rows}")
            print(f"Output: {result.output_path}")
            for match in result.top_matches:
                print(f"[{match.get('rank')}] {match.get('score')} - {match.get('citation')}")
                print(str(match.get("text", ""))[:500])
            return 0
        elif args.semantic_command == "ask":
            result = semantic_ask_command(args.query, top_n=args.top_n, method=args.method)
            print(f"Status: {result.status}")
            print(f"Database: {result.database}")
            print(f"Rows: {result.rows}")
            print(f"Output: {result.output_path}")
            print(f"Report: {result.report_path}")
            if result.answer:
                print(result.answer)
            return 0
        else:
            semantic_parser.print_help()
            return 0
        print(f"Status: {result.status}")
        print(f"Database: {result.database}")
        print(f"Rows: {result.rows}")
        print(f"Output: {result.output_path}")
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())

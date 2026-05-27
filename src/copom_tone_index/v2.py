from __future__ import annotations

import hashlib
import html
import json
import logging
import re
import shutil
from dataclasses import dataclass
from datetime import timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from copom_tone_index.bcb import (
    attach_selic_to_meetings,
    copom_detail_url,
    date_window_for_sources,
    fetch_copom_documents,
    fetch_selic,
)
from copom_tone_index.config import get_paths, load_lexicon, load_settings, load_topics, load_v2_settings
from copom_tone_index.http_client import CachedHttpClient
from copom_tone_index.scoring import classify_index
from copom_tone_index.storage import export_tables, write_tables
from copom_tone_index.text import clean_copom_html, contains_any, normalize_whitespace, split_sentences, strip_accents

LOGGER = logging.getLogger(__name__)
DEFAULT_RULE_ENGINE_VERSION = "taxonomy-rules-v2.0.4"

V2_TABLES = [
    "event_calendar",
    "v2_runs",
    "v2_meetings",
    "v2_documents",
    "v2_sentences",
    "v2_sentence_scores",
    "v2_document_scores",
    "v2_meeting_scores",
    "v2_subindices",
    "v2_calibration",
    "v2_taxonomy_versions",
    "v2_lexicon_versions",
    "v2_redline",
    "v2_evidence",
    "v2_labels",
    "v2_model_predictions",
    "v2_supervised_predictions",
    "v2_supervised_model_audit",
    "v2_model_audit",
    "v2_model_audit_details",
    "v2_model_audit_error_analysis",
    "v2_model_audit_error_classification",
    "v2_reviewer_disagreements",
    "focus_vintages",
    "focus_event_features",
    "market_observations",
    "market_event_windows",
    "decision_expectations",
    "public_market_source_audit",
    "decision_expectation_source_audit",
    "public_market_coverage",
    "v21_event_panel",
]


@dataclass(frozen=True)
class V2CommandResult:
    database: Path
    output_path: Path | None
    rows: int
    status: str
    run_id: str | None = None


@dataclass(frozen=True)
class V2BenchmarkResult:
    database: Path
    report_path: Path
    output_dir: Path
    status: str
    rows: int
    warnings: int
    run_id: str


def v2_backfill_command(quantity: int | None = None) -> V2CommandResult:
    settings = load_settings()
    v2_settings = load_v2_settings()
    paths = get_paths(settings)
    run_id = make_run_id("v2_backfill")
    client = CachedHttpClient(
        cache_dir=paths.raw,
        timeout_seconds=int(settings["pipeline"]["request_timeout_seconds"]),
        retries=int(settings["pipeline"]["retries"]),
        retry_backoff_seconds=float(settings["pipeline"]["retry_backoff_seconds"]),
    )
    quantity = int(quantity or v2_settings["backfill"]["default_quantity"])
    meetings, documents = fetch_copom_documents(client, quantity=quantity)
    start_date, end_date = date_window_for_sources(meetings)
    selic = fetch_selic(client, start_date, end_date)
    meetings = attach_selic_to_meetings(meetings, selic)
    meetings, documents = prepare_v2_backfill(meetings, documents, run_id, v2_settings)
    sentences = build_v2_sentences(documents, run_id)
    event_calendar = build_event_calendar(documents)
    taxonomy_versions, lexicon_versions = build_v2_version_tables(load_topics(), load_lexicon(), v2_settings, run_id)
    runs = append_run(paths.database, run_row(run_id, "backfill", "completed", {"quantity": quantity}))
    write_tables(
        paths.database,
        {
            "v2_runs": runs,
            "v2_meetings": meetings,
            "v2_documents": documents,
            "v2_sentences": sentences,
            "event_calendar": event_calendar,
            "v2_taxonomy_versions": taxonomy_versions,
            "v2_lexicon_versions": lexicon_versions,
        },
    )
    export_tables(paths.database, paths.processed, ["v2_runs", "v2_meetings", "v2_documents", "v2_sentences", "event_calendar"])
    return V2CommandResult(paths.database, paths.processed / "v2_documents.csv", len(documents), "completed", run_id)


def v2_reuse_backfill_command(quantity: int | None = None) -> V2CommandResult:
    paths = get_paths()
    run_id = make_run_id("v2_backfill")
    documents = read_optional_table(paths.database, "v2_documents", pd.DataFrame())
    documents = ensure_v2_document_source_urls(documents)
    event_calendar = build_event_calendar(documents)
    runs = append_run(paths.database, run_row(run_id, "backfill", "reused", {"quantity": quantity, "documents": len(documents)}))
    write_tables(paths.database, {"v2_runs": runs, "v2_documents": documents, "event_calendar": event_calendar})
    export_tables(paths.database, paths.processed, ["v2_runs", "v2_documents", "event_calendar"])
    return V2CommandResult(paths.database, paths.processed / "v2_documents.csv", len(documents), "reused", run_id)


def v2_score_command() -> V2CommandResult:
    settings = load_settings()
    v2_settings = load_v2_settings()
    paths = get_paths(settings)
    run_id = make_run_id("v2_score")
    meetings = read_optional_table(paths.database, "v2_meetings", pd.DataFrame())
    documents = read_optional_table(paths.database, "v2_documents", pd.DataFrame())
    if documents.empty:
        v2_backfill_command()
        documents = read_optional_table(paths.database, "v2_documents", pd.DataFrame())
    if meetings.empty:
        meetings = read_optional_table(paths.database, "copom_meetings", pd.DataFrame())
    if meetings.empty:
        meetings = _meetings_from_v2_documents(documents)
    sentences = build_v2_sentences(documents, run_id)
    sentence_scores = score_v2_sentences(
        sentences,
        topics=load_topics(),
        lexicon=load_lexicon(),
        v2_settings=v2_settings,
        run_id=run_id,
    )
    document_scores, meeting_scores_raw = aggregate_v2_scores(
        meetings,
        documents,
        sentence_scores,
        v2_settings,
        run_id,
    )
    calibration = build_v2_calibration(meeting_scores_raw, v2_settings, run_id)
    meeting_scores = apply_v2_calibration(meeting_scores_raw, calibration, v2_settings)
    subindices = build_v2_subindices(sentence_scores, meeting_scores, v2_settings, run_id)
    evidence = build_v2_evidence(sentence_scores, meeting_scores, top_n=5)
    predictions = build_v2_model_predictions(sentence_scores)
    labels = read_optional_table(paths.database, "v2_labels", empty_v2_labels())
    audit = build_v2_model_audit(labels, predictions, run_id)
    audit_details = build_v2_model_audit_details(labels, predictions, run_id)
    taxonomy_versions, lexicon_versions = build_v2_version_tables(load_topics(), load_lexicon(), v2_settings, run_id)
    runs = append_run(paths.database, run_row(run_id, "score", "completed", {"documents": len(documents)}))
    tables = {
        "v2_runs": runs,
        "v2_sentences": sentences,
        "v2_sentence_scores": sentence_scores,
        "v2_document_scores": document_scores,
        "v2_meeting_scores": meeting_scores,
        "v2_subindices": subindices,
        "v2_calibration": calibration,
        "v2_taxonomy_versions": taxonomy_versions,
        "v2_lexicon_versions": lexicon_versions,
        "v2_evidence": evidence,
        "v2_labels": labels,
        "v2_model_predictions": predictions,
        "v2_model_audit": audit,
        "v2_model_audit_details": audit_details,
    }
    write_tables(paths.database, tables)
    export_tables(paths.database, paths.processed, list(tables))
    return V2CommandResult(paths.database, paths.processed / "v2_meeting_scores.csv", len(meeting_scores), "completed", run_id)


def v2_calibrate_command() -> V2CommandResult:
    settings = load_settings()
    v2_settings = load_v2_settings()
    paths = get_paths(settings)
    run_id = make_run_id("v2_calibrate")
    scores = read_optional_table(paths.database, "v2_meeting_scores", pd.DataFrame())
    if scores.empty:
        return v2_score_command()
    calibration = build_v2_calibration(scores, v2_settings, run_id)
    calibrated_scores = apply_v2_calibration(scores, calibration, v2_settings)
    runs = append_run(paths.database, run_row(run_id, "calibrate", "completed", {"scores": len(scores)}))
    write_tables(paths.database, {"v2_runs": runs, "v2_calibration": calibration, "v2_meeting_scores": calibrated_scores})
    export_tables(paths.database, paths.processed, ["v2_calibration", "v2_meeting_scores"])
    return V2CommandResult(paths.database, paths.processed / "v2_calibration.csv", len(calibration), "completed", run_id)


def v2_redline_command(force: bool = False) -> V2CommandResult:
    settings = load_settings()
    v2_settings = load_v2_settings()
    paths = get_paths(settings)
    run_id = make_run_id("v2_redline")
    documents = read_optional_table(paths.database, "v2_documents", pd.DataFrame())
    sentence_scores = read_optional_table(paths.database, "v2_sentence_scores", pd.DataFrame())
    if documents.empty or sentence_scores.empty:
        v2_score_command()
        documents = read_optional_table(paths.database, "v2_documents", pd.DataFrame())
        sentence_scores = read_optional_table(paths.database, "v2_sentence_scores", pd.DataFrame())
    existing = read_optional_table(paths.database, "v2_redline", pd.DataFrame())
    if not force and redline_is_current(existing, documents, sentence_scores):
        runs = append_run(
            paths.database,
            run_row(run_id, "redline", "reused", {"rows": len(existing), "reason": "coverage_current"}),
        )
        write_tables(paths.database, {"v2_runs": runs})
        export_tables(paths.database, paths.processed, ["v2_redline"])
        return V2CommandResult(paths.database, paths.processed / "v2_redline.csv", len(existing), "reused", run_id)
    redline = build_v2_redline(documents, sentence_scores, v2_settings, run_id)
    runs = append_run(paths.database, run_row(run_id, "redline", "completed", {"rows": len(redline)}))
    write_tables(paths.database, {"v2_runs": runs, "v2_redline": redline})
    export_tables(paths.database, paths.processed, ["v2_redline"])
    return V2CommandResult(paths.database, paths.processed / "v2_redline.csv", len(redline), "completed", run_id)


def v2_audit_command(sample_size: int = 120) -> V2CommandResult:
    settings = load_settings()
    paths = get_paths(settings)
    run_id = make_run_id("v2_audit")
    labels = read_optional_table(paths.database, "v2_labels", empty_v2_labels())
    predictions = read_optional_table(paths.database, "v2_model_predictions", pd.DataFrame())
    sentence_scores = read_optional_table(paths.database, "v2_sentence_scores", pd.DataFrame())
    if predictions.empty or sentence_scores.empty:
        v2_score_command()
        predictions = read_optional_table(paths.database, "v2_model_predictions", pd.DataFrame())
        sentence_scores = read_optional_table(paths.database, "v2_sentence_scores", pd.DataFrame())
    audit = build_v2_model_audit(labels, predictions, run_id)
    audit_details = build_v2_model_audit_details(labels, predictions, run_id)
    error_analysis = build_v2_model_audit_error_analysis(labels, predictions, sentence_scores, run_id)
    reviewer_disagreements = build_v2_reviewer_disagreements(labels, predictions, sentence_scores, run_id)
    error_classification = build_v2_model_audit_error_classification(error_analysis, reviewer_disagreements, run_id)
    error_summary = build_v2_model_audit_error_summary(error_analysis, run_id)
    sample = build_manual_label_sample(sentence_scores, sample_size=sample_size)
    output_dir = paths.processed.parent / "v2"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir = paths.reports.parent / "v2"
    report_dir.mkdir(parents=True, exist_ok=True)
    sample.to_csv(output_dir / "manual_label_sample.csv", index=False)
    audit.to_csv(output_dir / "model_audit.csv", index=False)
    audit_details.to_csv(output_dir / "model_audit_details.csv", index=False)
    error_analysis.to_csv(output_dir / "model_audit_error_analysis.csv", index=False)
    error_classification.to_csv(output_dir / "model_audit_error_classification.csv", index=False)
    error_summary.to_csv(output_dir / "model_audit_error_summary.csv", index=False)
    reviewer_disagreements.to_csv(output_dir / "reviewer_disagreements.csv", index=False)
    audit_html = build_v2_model_audit_report_html(audit, error_analysis, error_summary, run_id)
    (report_dir / "model_audit.html").write_text(audit_html, encoding="utf-8")
    disagreement_html = build_v2_validation_disagreement_report_html(
        audit,
        reviewer_disagreements,
        error_classification,
        run_id,
    )
    (report_dir / "validation_disagreement_report.html").write_text(disagreement_html, encoding="utf-8")
    runs = append_run(paths.database, run_row(run_id, "audit", "completed", {"sample_size": sample_size}))
    write_tables(
        paths.database,
        {
            "v2_runs": runs,
            "v2_model_audit": audit,
            "v2_model_audit_details": audit_details,
            "v2_model_audit_error_analysis": error_analysis,
            "v2_model_audit_error_classification": error_classification,
            "v2_reviewer_disagreements": reviewer_disagreements,
        },
    )
    export_tables(
        paths.database,
        paths.processed,
        [
            "v2_model_audit",
            "v2_model_audit_details",
            "v2_model_audit_error_analysis",
            "v2_model_audit_error_classification",
            "v2_reviewer_disagreements",
        ],
    )
    return V2CommandResult(paths.database, output_dir / "model_audit.csv", len(audit), "completed", run_id)


def v2_benchmark_baseline_command() -> V2BenchmarkResult:
    settings = load_settings()
    paths = get_paths(settings)
    run_id = make_run_id("v2_benchmark")
    labels = read_optional_table(paths.database, "v2_labels", empty_v2_labels())
    predictions = read_optional_table(paths.database, "v2_model_predictions", pd.DataFrame())
    sentence_scores = read_optional_table(paths.database, "v2_sentence_scores", pd.DataFrame())
    if predictions.empty or sentence_scores.empty:
        v2_score_command()
        predictions = read_optional_table(paths.database, "v2_model_predictions", pd.DataFrame())
        sentence_scores = read_optional_table(paths.database, "v2_sentence_scores", pd.DataFrame())
    output_dir = paths.processed.parent / "v2"
    report_dir = paths.reports.parent / "v2"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    archive_existing_benchmark_snapshot(output_dir, report_dir, "v2.0.3")
    previous_dir = output_dir / "benchmarks" / "v2.0.3"
    previous_sample_path = previous_dir / "baseline_benchmark_by_sample.csv"
    previous_stance_path = previous_dir / "baseline_benchmark_by_stance.csv"
    previous_sample = pd.read_csv(previous_sample_path) if previous_sample_path.exists() else pd.DataFrame()
    previous_stance = pd.read_csv(previous_stance_path) if previous_stance_path.exists() else pd.DataFrame()
    by_sample, by_topic, by_stance = build_baseline_benchmark_tables(labels, predictions, sentence_scores, run_id)
    gates = evaluate_benchmark_gates(by_sample, by_stance, previous_sample, previous_stance)
    by_sample.to_csv(output_dir / "baseline_benchmark_by_sample.csv", index=False)
    by_topic.to_csv(output_dir / "baseline_benchmark_by_topic.csv", index=False)
    by_stance.to_csv(output_dir / "baseline_benchmark_by_stance.csv", index=False)
    report_html = build_baseline_benchmark_report_html(by_sample, by_topic, by_stance, gates, run_id)
    report_path = report_dir / "baseline_benchmark_report.html"
    report_path.write_text(report_html, encoding="utf-8")
    runs = append_run(
        paths.database,
        run_row(run_id, "benchmark_baseline", gates["status"], {"warnings": len(gates["warnings"])}),
    )
    write_tables(paths.database, {"v2_runs": runs})
    status = str(gates["status"])
    return V2BenchmarkResult(
        database=paths.database,
        report_path=report_path,
        output_dir=output_dir,
        status=status,
        rows=len(by_sample),
        warnings=len(gates["warnings"]),
        run_id=run_id,
    )


def v2_review_remaining_errors_command(limit: int = 177) -> V2CommandResult:
    settings = load_settings()
    paths = get_paths(settings)
    run_id = make_run_id("v2_remaining_errors")
    output_dir = paths.processed.parent / "v2"
    report_dir = paths.reports.parent / "v2"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    error_classification = read_remaining_error_source(paths.database, output_dir)
    if error_classification.empty:
        v2_audit_command()
        error_classification = read_remaining_error_source(paths.database, output_dir)
    review = build_remaining_error_review(error_classification, run_id, limit=limit)
    output_path = output_dir / "remaining_error_review.csv"
    report_path = report_dir / "remaining_error_review.html"
    review.to_csv(output_path, index=False)
    report_path.write_text(build_remaining_error_review_html(review, run_id, limit), encoding="utf-8")
    runs = append_run(paths.database, run_row(run_id, "review_remaining_errors", "completed", {"limit": limit, "rows": len(review)}))
    write_tables(paths.database, {"v2_runs": runs})
    return V2CommandResult(paths.database, output_path, len(review), "completed", run_id)


def v2_train_supervised_command() -> V2CommandResult:
    settings = load_settings()
    paths = get_paths(settings)
    run_id = make_run_id("v2_train_supervised")
    labels = read_optional_table(paths.database, "v2_labels", empty_v2_labels())
    predictions = read_optional_table(paths.database, "v2_model_predictions", pd.DataFrame())
    sentence_scores = read_optional_table(paths.database, "v2_sentence_scores", pd.DataFrame())
    if predictions.empty or sentence_scores.empty:
        v2_score_command()
        predictions = read_optional_table(paths.database, "v2_model_predictions", pd.DataFrame())
        sentence_scores = read_optional_table(paths.database, "v2_sentence_scores", pd.DataFrame())
    supervised_predictions, audit = build_v2_supervised_model_artifacts(labels, predictions, sentence_scores, run_id)
    output_dir = paths.processed.parent / "v2"
    report_dir = paths.reports.parent / "v2"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    supervised_predictions.to_csv(output_dir / "supervised_model_predictions.csv", index=False)
    audit.to_csv(output_dir / "supervised_model_audit.csv", index=False)
    report_path = report_dir / "supervised_model_report.html"
    report_path.write_text(build_supervised_model_report_html(audit, supervised_predictions, run_id), encoding="utf-8")
    status = "insufficient_labels" if not audit.empty and (audit["status"] == "insufficient_labels").all() else "completed"
    runs = append_run(paths.database, run_row(run_id, "train_supervised", status, {"predictions": len(supervised_predictions)}))
    write_tables(
        paths.database,
        {
            "v2_runs": runs,
            "v2_supervised_predictions": supervised_predictions,
            "v2_supervised_model_audit": audit,
        },
    )
    export_tables(paths.database, paths.processed, ["v2_supervised_predictions", "v2_supervised_model_audit"])
    return V2CommandResult(paths.database, output_dir / "supervised_model_audit.csv", len(audit), status, run_id)


def v2_freeze_release_command(version: str = "v2.0.4-holdout-stance-hardened") -> V2CommandResult:
    settings = load_settings()
    v2_settings = load_v2_settings()
    paths = get_paths(settings)
    run_id = make_run_id("v2_freeze")
    benchmark = v2_benchmark_baseline_command()
    from copom_tone_index.v2_health import v2_health_check_command

    health = v2_health_check_command()
    output_dir = paths.processed.parent / "v2"
    report_dir = paths.reports.parent / "v2"
    release_dir = report_dir / "releases" / version
    report_dir.mkdir(parents=True, exist_ok=True)
    release_dir.mkdir(parents=True, exist_ok=True)
    by_sample = pd.read_csv(output_dir / "baseline_benchmark_by_sample.csv") if (output_dir / "baseline_benchmark_by_sample.csv").exists() else pd.DataFrame()
    total_metrics = (
        by_sample[by_sample["sample"] == "total_consensus"].iloc[0].to_dict()
        if not by_sample.empty and (by_sample["sample"] == "total_consensus").any()
        else {}
    )
    status = "fail" if benchmark.status == "fail" or health.errors > 0 else "completed"
    methodology_path = report_dir / "v2_0_methodology_report.html"
    methodology_path.write_text(
        build_v2_methodology_report_html(version, total_metrics, benchmark.status, health.status, health.warnings, health.errors, v2_settings),
        encoding="utf-8",
    )
    artifact_paths = collect_release_artifacts(output_dir, report_dir, methodology_path)
    copied_artifacts: list[dict[str, object]] = []
    for path in artifact_paths:
        if not path.exists() or not path.is_file():
            continue
        target = release_dir / path.name
        shutil.copy2(path, target)
        copied_artifacts.append({"path": str(target), "sha256": file_sha256(target), "bytes": target.stat().st_size})
    manifest = {
        "version": version,
        "status": status,
        "generated_at": utc_now_naive().isoformat(),
        "run_id": run_id,
        "database_path": str(paths.database),
        "rule_engine_version": rule_engine_version(v2_settings),
        "calibration_version": str(v2_settings.get("scoring", {}).get("default_calibration", "")),
        "benchmark_status": benchmark.status,
        "health_status": health.status,
        "health_warnings": health.warnings,
        "health_errors": health.errors,
        "metrics": json_safe_record(total_metrics),
        "artifacts": copied_artifacts,
        "release_dir": str(release_dir),
        "methodology_report": str(methodology_path),
        "official_index": "deterministic_v2_baseline",
        "supervised_model_status": "experimental_not_official",
    }
    manifest_path = report_dir / "release_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copy2(manifest_path, release_dir / manifest_path.name)
    runs = append_run(paths.database, run_row(run_id, "freeze_release", status, {"version": version, "artifacts": len(copied_artifacts)}))
    write_tables(paths.database, {"v2_runs": runs})
    return V2CommandResult(paths.database, manifest_path, len(copied_artifacts), status, run_id)


def archive_existing_benchmark_snapshot(output_dir: Path, report_dir: Path, version: str) -> Path | None:
    """Preserve the latest benchmark once as a fixed comparison baseline."""

    snapshot_dir = output_dir / "benchmarks" / version
    marker = snapshot_dir / ".snapshot_complete"
    if marker.exists():
        return snapshot_dir
    required = [
        output_dir / "baseline_benchmark_by_sample.csv",
        output_dir / "baseline_benchmark_by_topic.csv",
        output_dir / "baseline_benchmark_by_stance.csv",
    ]
    if not all(path.exists() for path in required):
        return None
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    for path in required:
        shutil.copy2(path, snapshot_dir / path.name)
    report_path = report_dir / "baseline_benchmark_report.html"
    if report_path.exists():
        shutil.copy2(report_path, snapshot_dir / report_path.name)
    marker.write_text("snapshot_complete\n", encoding="utf-8")
    return snapshot_dir


def read_remaining_error_source(database: Path, output_dir: Path) -> pd.DataFrame:
    csv_path = output_dir / "model_audit_error_classification.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return read_optional_table(database, "v2_model_audit_error_classification", pd.DataFrame())


def build_remaining_error_review(error_classification: pd.DataFrame, run_id: str, limit: int = 177) -> pd.DataFrame:
    columns = [
        "run_id",
        "sentence_id",
        "meeting_id",
        "nro_reuniao",
        "document_type",
        "priority_score",
        "topic_label",
        "predicted_topic",
        "stance_label",
        "predicted_stance",
        "is_informative_label",
        "predicted_is_informative",
        "taxonomy_boundary_flag",
        "error_interpretation",
        "review_category",
        "pattern_key",
        "candidate_rank",
        "review_reason",
        "text",
    ]
    if error_classification.empty:
        return pd.DataFrame(columns=columns)
    frame = error_classification.copy()
    for column in ["priority_score", "issue_count"]:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    if "priority_score" in frame:
        frame = frame.sort_values("priority_score", ascending=False)
    if "error_interpretation" not in frame:
        frame["error_interpretation"] = ""
    likely = frame[frame["error_interpretation"].astype(str).eq("likely_baseline_error")].copy()
    if limit and limit > 0:
        likely = likely.head(limit)
    topic_transition = (likely["topic_label"].astype(str) + " <- " + likely["predicted_topic"].astype(str)) if not likely.empty else pd.Series(dtype=str)
    stance_transition = (likely["stance_label"].astype(str) + " <- " + likely["predicted_stance"].astype(str)) if not likely.empty else pd.Series(dtype=str)
    topic_counts = topic_transition.value_counts().to_dict()
    stance_counts = stance_transition.value_counts().to_dict()
    candidate_counts: dict[str, int] = {}
    rows: list[dict[str, object]] = []
    for _, row in likely.iterrows():
        topic_key = f"{row.get('topic_label', '')} <- {row.get('predicted_topic', '')}"
        stance_key = f"{row.get('stance_label', '')} <- {row.get('predicted_stance', '')}"
        category, reason = classify_remaining_error(row, int(topic_counts.get(topic_key, 0)), int(stance_counts.get(stance_key, 0)))
        pattern_key = stance_key if str(row.get("stance_label", "")) != str(row.get("predicted_stance", "")) else topic_key
        if category == "generalizable_rule_candidate":
            candidate_counts[pattern_key] = candidate_counts.get(pattern_key, 0) + 1
            candidate_rank = candidate_counts[pattern_key] if len(candidate_counts) <= 10 else 0
        else:
            candidate_rank = 0
        rows.append(
            {
                "run_id": run_id,
                "sentence_id": row.get("sentence_id", ""),
                "meeting_id": row.get("meeting_id", ""),
                "nro_reuniao": row.get("nro_reuniao", ""),
                "document_type": row.get("document_type", ""),
                "priority_score": row.get("priority_score", np.nan),
                "topic_label": row.get("topic_label", ""),
                "predicted_topic": row.get("predicted_topic", ""),
                "stance_label": row.get("stance_label", ""),
                "predicted_stance": row.get("predicted_stance", ""),
                "is_informative_label": row.get("is_informative_label", ""),
                "predicted_is_informative": row.get("predicted_is_informative", ""),
                "taxonomy_boundary_flag": row.get("taxonomy_boundary_flag", ""),
                "error_interpretation": row.get("error_interpretation", ""),
                "review_category": category,
                "pattern_key": pattern_key,
                "candidate_rank": candidate_rank,
                "review_reason": reason,
                "text": row.get("text", ""),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def classify_remaining_error(row: pd.Series, topic_count: int, stance_count: int) -> tuple[str, str]:
    raw_boundary = row.get("taxonomy_boundary_flag", "")
    if pd.isna(raw_boundary) or str(raw_boundary).strip() == "":
        raw_boundary = row.get("taxonomy_boundary", "")
    boundary = "" if pd.isna(raw_boundary) else str(raw_boundary).strip()
    if boundary:
        return "taxonomy_boundary", f"Explicit taxonomy boundary: {boundary}."
    raw_conflict = row.get("human_conflict_types", "")
    conflict = "" if pd.isna(raw_conflict) else str(raw_conflict).strip()
    if conflict:
        return "label_review_needed", "Reviewer disagreement exists around this sentence."
    text = normalized_rule_text(str(row.get("text", "")))
    mixed_terms = ["embora", "apesar", "contudo", "mas ", "por outro lado", "ao mesmo tempo", "nao obstante", "a despeito"]
    if contains_rule_term(text, mixed_terms):
        return "contextual_limit", "Sentence has mixed-signal connective terms; avoid automatic rule tuning."
    if topic_count >= 3 or stance_count >= 8 or float(row.get("priority_score", 0) or 0) >= 3.5:
        return "generalizable_rule_candidate", "Repeated or high-priority transition that may justify a future general rule."
    return "do_not_tune", "Low-frequency residual error; keep for documentation rather than rule tuning."


def build_remaining_error_review_html(review: pd.DataFrame, run_id: str, limit: int) -> str:
    summary = review.groupby("review_category", dropna=False).size().reset_index(name="count") if not review.empty else pd.DataFrame()
    candidates = (
        review[review["review_category"] == "generalizable_rule_candidate"]
        .groupby("pattern_key", dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(10)
        if not review.empty
        else pd.DataFrame()
    )
    top_rows = review.sort_values("priority_score", ascending=False).head(40) if not review.empty else review
    return f"""<!doctype html>
<html lang="pt-BR">
<head><meta charset="utf-8"><title>COPOM Watch V2 Remaining Error Review</title>
<style>body{{font-family:Arial,sans-serif;margin:32px;color:#1f2933}}table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{border:1px solid #d9e2ec;padding:8px;vertical-align:top}}th{{background:#f0f4f8;text-align:left}}.badge{{display:inline-block;padding:4px 8px;background:#e0f2fe;color:#0b4f71;border-radius:4px}}</style></head>
<body>
<h1>COPOM Watch V2 Remaining Error Review</h1>
<p><span class="badge">No automatic tuning</span></p>
<p>Run ID: <code>{html.escape(run_id)}</code> | Limit: <strong>{limit}</strong> | Rows: <strong>{len(review)}</strong></p>
<h2>Resumo Por Categoria</h2>
{dataframe_to_html_table(summary)}
<h2>Top 10 Candidatos A Regras Futuras</h2>
{dataframe_to_html_table(candidates)}
<h2>Erros Priorizados</h2>
{dataframe_to_html_table(top_rows)}
</body></html>"""


def build_v2_supervised_model_artifacts(
    labels: pd.DataFrame,
    baseline_predictions: pd.DataFrame,
    sentence_scores: pd.DataFrame,
    run_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_version = "experimental-supervised-tfidf-logreg-v1"
    prediction_columns = [
        "run_id",
        "sentence_id",
        "target",
        "split",
        "truth_label",
        "predicted_label",
        "baseline_label",
        "confidence",
        "model_version",
        "model_status",
    ]
    audit_columns = ["run_id", "target", "metric", "value", "status", "detail", "model_version", "model_status"]
    accepted = accepted_v2_labels(labels)
    if accepted.empty:
        return pd.DataFrame(columns=prediction_columns), supervised_insufficient_audit(run_id, "all", "No accepted human labels.", model_version)
    slices = benchmark_label_slices(accepted)
    sample_002_ids = set(slices.get("sample_002_consensus", pd.DataFrame()).get("sentence_id", pd.Series(dtype=str)).astype(str))
    consensus = consensus_v2_labels(accepted)
    if consensus.empty or not sample_002_ids:
        return pd.DataFrame(columns=prediction_columns), supervised_insufficient_audit(
            run_id,
            "all",
            "Need sample 001 plus sample 002 holdout labels.",
            model_version,
        )
    data = consensus.merge(
        sentence_scores[["sentence_id", "text"]].drop_duplicates("sentence_id"),
        on="sentence_id",
        how="inner",
    ).merge(
        baseline_predictions[
            ["sentence_id", "predicted_topic", "predicted_stance", "predicted_is_informative"]
        ].drop_duplicates("sentence_id"),
        on="sentence_id",
        how="left",
    )
    data["split"] = np.where(data["sentence_id"].astype(str).isin(sample_002_ids), "holdout_sample_002", "train_sample_001")
    train = data[data["split"] == "train_sample_001"].copy()
    holdout = data[data["split"] == "holdout_sample_002"].copy()
    if train.empty or holdout.empty:
        return pd.DataFrame(columns=prediction_columns), supervised_insufficient_audit(
            run_id,
            "all",
            "Train or holdout split is empty.",
            model_version,
        )
    prediction_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    targets = {
        "stance": ("stance_label", "predicted_stance"),
        "topic": ("topic_label", "predicted_topic"),
        "is_informative": ("is_informative_label", "predicted_is_informative"),
    }
    for target, (truth_col, baseline_col) in targets.items():
        target_predictions, target_audit = train_single_supervised_target(
            train,
            holdout,
            target,
            truth_col,
            baseline_col,
            run_id,
            model_version,
        )
        prediction_rows.extend(target_predictions)
        audit_rows.extend(target_audit)
    return pd.DataFrame(prediction_rows, columns=prediction_columns), pd.DataFrame(audit_rows, columns=audit_columns)


def train_single_supervised_target(
    train: pd.DataFrame,
    holdout: pd.DataFrame,
    target: str,
    truth_col: str,
    baseline_col: str,
    run_id: str,
    model_version: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    audit_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    y_train = train[truth_col].astype(str)
    y_holdout = holdout[truth_col].astype(str)
    if len(train) < 2 or len(holdout) < 1 or y_train.nunique() < 2:
        audit_rows.append(
            supervised_audit_row(
                run_id,
                target,
                "status",
                np.nan,
                "insufficient_labels",
                f"Need at least two train labels and two classes; got train={len(train)}, classes={y_train.nunique()}, holdout={len(holdout)}.",
                model_version,
            )
        )
        return prediction_rows, audit_rows
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
    except Exception as exc:  # pragma: no cover - dependency failure is environment-specific.
        audit_rows.append(supervised_audit_row(run_id, target, "status", np.nan, "unavailable", str(exc), model_version))
        return prediction_rows, audit_rows
    model = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=5000)),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
        ]
    )
    model.fit(train["text"].astype(str), y_train)
    predicted = pd.Series(model.predict(holdout["text"].astype(str)), index=holdout.index).astype(str)
    probabilities = model.predict_proba(holdout["text"].astype(str)) if hasattr(model, "predict_proba") else None
    confidence = probabilities.max(axis=1) if probabilities is not None else np.full(len(holdout), np.nan)
    for idx, (_, row) in enumerate(holdout.iterrows()):
        prediction_rows.append(
            {
                "run_id": run_id,
                "sentence_id": row["sentence_id"],
                "target": target,
                "split": "holdout_sample_002",
                "truth_label": str(row[truth_col]),
                "predicted_label": str(predicted.loc[row.name]),
                "baseline_label": str(row.get(baseline_col, "")),
                "confidence": float(confidence[idx]) if not pd.isna(confidence[idx]) else np.nan,
                "model_version": model_version,
                "model_status": "experimental",
            }
        )
    metric_frame = pd.DataFrame({"truth": y_holdout.astype(str), "pred": predicted.astype(str), "baseline": holdout[baseline_col].astype(str)})
    audit_rows.extend(
        [
            supervised_audit_row(run_id, target, "holdout_observations", len(holdout), "ok", "", model_version),
            supervised_audit_row(run_id, target, "train_observations", len(train), "ok", "", model_version),
            supervised_audit_row(run_id, target, "accuracy", accuracy_metric(metric_frame, "truth", "pred"), "ok", "", model_version),
            supervised_audit_row(run_id, target, "f1_macro", macro_f1_metric(metric_frame["truth"], metric_frame["pred"]), "ok", "", model_version),
            supervised_audit_row(
                run_id,
                target,
                "baseline_accuracy",
                accuracy_metric(metric_frame, "truth", "baseline"),
                "ok",
                "Baseline deterministic benchmark on the same holdout.",
                model_version,
            ),
        ]
    )
    for record in confusion_matrix_records(metric_frame, "truth", "pred"):
        audit_rows.append(
            supervised_audit_row(run_id, target, "confusion", float(record["count"]), "ok", json.dumps(record, ensure_ascii=False), model_version)
        )
    return prediction_rows, audit_rows


def supervised_audit_row(
    run_id: str,
    target: str,
    metric: str,
    value: object,
    status: str,
    detail: str,
    model_version: str,
) -> dict[str, object]:
    return {
        "run_id": run_id,
        "target": target,
        "metric": metric,
        "value": value,
        "status": status,
        "detail": detail,
        "model_version": model_version,
        "model_status": "experimental",
    }


def supervised_insufficient_audit(run_id: str, target: str, detail: str, model_version: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            supervised_audit_row(
                run_id,
                target,
                "status",
                np.nan,
                "insufficient_labels",
                detail,
                model_version,
            )
        ]
    )


def build_supervised_model_report_html(audit: pd.DataFrame, predictions: pd.DataFrame, run_id: str) -> str:
    metrics = audit[audit["metric"].isin(["accuracy", "f1_macro", "baseline_accuracy", "holdout_observations", "train_observations"])] if not audit.empty else audit
    return f"""<!doctype html>
<html lang="pt-BR"><head><meta charset="utf-8"><title>COPOM Watch V2 Supervised Model</title>
<style>body{{font-family:Arial,sans-serif;margin:32px;color:#1f2933}}table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{border:1px solid #d9e2ec;padding:8px}}th{{background:#f0f4f8;text-align:left}}.badge{{display:inline-block;padding:4px 8px;background:#fff3cd;color:#513c06;border-radius:4px}}</style></head>
<body>
<h1>COPOM Watch V2 Supervised Model Report</h1>
<p><span class="badge">Experimental benchmark only; deterministic V2 baseline remains official.</span></p>
<p>Run ID: <code>{html.escape(run_id)}</code></p>
<h2>Métricas</h2>{dataframe_to_html_table(metrics)}
<h2>Predições Holdout</h2>{dataframe_to_html_table(predictions.head(80))}
</body></html>"""


def build_v2_methodology_report_html(
    version: str,
    metrics: dict[str, object],
    benchmark_status: str,
    health_status: str,
    health_warnings: int,
    health_errors: int,
    v2_settings: dict[str, Any],
) -> str:
    metric_frame = pd.DataFrame([metrics]) if metrics else pd.DataFrame()
    version_frame = pd.DataFrame(
        [
            {
                "version": version,
                "rule_engine_version": rule_engine_version(v2_settings),
                "calibration_version": v2_settings.get("scoring", {}).get("default_calibration", ""),
                "benchmark_status": benchmark_status,
                "health_status": health_status,
                "health_warnings": health_warnings,
                "health_errors": health_errors,
                "official_index": "deterministic_v2_baseline",
                "supervised_model": "experimental_not_official",
            }
        ]
    )
    return f"""<!doctype html>
<html lang="pt-BR"><head><meta charset="utf-8"><title>COPOM Watch V2.0 Methodology</title>
<style>body{{font-family:Arial,sans-serif;margin:32px;line-height:1.45;color:#1f2933}}table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{border:1px solid #d9e2ec;padding:8px;vertical-align:top}}th{{background:#f0f4f8;text-align:left}}</style></head>
<body>
<h1>COPOM Watch V2.0 Methodology Report</h1>
<p>A V2.0 fecha o core metodológico com baseline determinístico versionado, validação humana, redline textual, subíndices e benchmark anti-overfit. Mercado, Focus expandido e RAG ficam para V2.1/V2.2.</p>
<h2>Versão E Status</h2>{dataframe_to_html_table(version_frame)}
<h2>Métricas De Aceite</h2>{dataframe_to_html_table(metric_frame)}
<h2>Decisão Metodológica</h2>
<p>O índice oficial continua sendo o baseline determinístico V2.0.4. O modelo supervisionado leve, quando treinado, é benchmark experimental e não substitui os scores oficiais.</p>
<h2>Limitações</h2>
<p>Os erros remanescentes devem ser tratados como fila qualitativa. Novas regras só devem ser aceitas quando forem gerais, auditáveis e não piorarem o holdout.</p>
</body></html>"""


def collect_release_artifacts(output_dir: Path, report_dir: Path, methodology_path: Path) -> list[Path]:
    return [
        report_dir / "acceptance_report.html",
        report_dir / "acceptance_report.json",
        report_dir / "baseline_benchmark_report.html",
        report_dir / "model_audit.html",
        report_dir / "validation_disagreement_report.html",
        report_dir / "remaining_error_review.html",
        report_dir / "supervised_model_report.html",
        report_dir / "copom_watch_v2_278.html",
        methodology_path,
        output_dir / "baseline_benchmark_by_sample.csv",
        output_dir / "baseline_benchmark_by_topic.csv",
        output_dir / "baseline_benchmark_by_stance.csv",
        output_dir / "model_audit.csv",
        output_dir / "model_audit_error_classification.csv",
        output_dir / "remaining_error_review.csv",
        output_dir / "supervised_model_audit.csv",
        output_dir / "supervised_model_predictions.csv",
    ]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_safe_record(record: dict[str, object]) -> dict[str, object]:
    safe: dict[str, object] = {}
    for key, value in record.items():
        if isinstance(value, (np.integer, np.floating)):
            safe[key] = value.item()
        elif pd.isna(value):
            safe[key] = None
        else:
            safe[key] = value
    return safe


def v2_import_labels_command(path: str | Path, label_source: str = "human") -> V2CommandResult:
    settings = load_settings()
    paths = get_paths(settings)
    run_id = make_run_id("v2_import_labels")
    imported = normalize_v2_labels(pd.read_csv(path), label_source=label_source)
    existing = read_optional_table(paths.database, "v2_labels", empty_v2_labels())
    labels = pd.concat([existing, imported], ignore_index=True).drop_duplicates(
        ["sentence_id", "label_source", "annotator_id"],
        keep="last",
    )
    predictions = read_optional_table(paths.database, "v2_model_predictions", pd.DataFrame())
    audit = build_v2_model_audit(labels, predictions, run_id)
    audit_details = build_v2_model_audit_details(labels, predictions, run_id)
    runs = append_run(paths.database, run_row(run_id, "import_labels", "completed", {"path": str(path)}))
    write_tables(
        paths.database,
        {"v2_runs": runs, "v2_labels": labels, "v2_model_audit": audit, "v2_model_audit_details": audit_details},
    )
    export_tables(paths.database, paths.processed, ["v2_labels", "v2_model_audit", "v2_model_audit_details"])
    return V2CommandResult(paths.database, paths.processed / "v2_labels.csv", len(labels), "completed", run_id)


def v2_report_command(meeting: int | None = None) -> V2CommandResult:
    settings = load_settings()
    v2_settings = load_v2_settings()
    paths = get_paths(settings)
    run_id = make_run_id("v2_report")
    scores = read_optional_table(paths.database, "v2_meeting_scores", pd.DataFrame())
    redline = read_optional_table(paths.database, "v2_redline", pd.DataFrame())
    if scores.empty:
        v2_score_command()
        scores = read_optional_table(paths.database, "v2_meeting_scores", pd.DataFrame())
    if redline.empty:
        v2_redline_command()
        redline = read_optional_table(paths.database, "v2_redline", pd.DataFrame())
    evidence = read_optional_table(paths.database, "v2_evidence", pd.DataFrame())
    subindices = read_optional_table(paths.database, "v2_subindices", pd.DataFrame())
    output_path = write_v2_html_report(scores, subindices, evidence, redline, v2_settings, meeting=meeting)
    runs = append_run(paths.database, run_row(run_id, "report", "completed", {"meeting": meeting}))
    write_tables(paths.database, {"v2_runs": runs})
    return V2CommandResult(paths.database, output_path, 1, "completed", run_id)


def v2_run_all_command(quantity: int | None = None, refresh_backfill: bool = False) -> V2CommandResult:
    paths = get_paths()
    if not refresh_backfill and existing_v2_backfill_is_sufficient(paths.database, quantity):
        backfill = v2_reuse_backfill_command(quantity=quantity)
    else:
        backfill = v2_backfill_command(quantity=quantity)
    score = v2_score_command()
    v2_redline_command()
    v2_audit_command()
    v2_report_command()
    return V2CommandResult(score.database, score.output_path, score.rows, "completed", backfill.run_id)


def prepare_v2_backfill(
    meetings: pd.DataFrame,
    documents: pd.DataFrame,
    run_id: str,
    v2_settings: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    documents = documents.copy()
    min_minutes = int(v2_settings["backfill"]["min_minutes_meeting"])
    min_statement = int(v2_settings["backfill"]["min_statement_meeting"])
    keep = (
        ((documents["document_type"] == "ata") & (documents["nro_reuniao"] >= min_minutes))
        | ((documents["document_type"] == "comunicado") & (documents["nro_reuniao"] >= min_statement))
    )
    documents = documents[keep].copy()
    now = utc_now_naive()
    documents = ensure_v2_document_source_urls(documents)
    documents["raw_text"] = documents["raw_text"].fillna("")
    documents["clean_text"] = documents["raw_text"].map(clean_copom_html)
    documents["source_hash"] = documents["raw_text"].map(stable_text_hash)
    documents["document_version"] = documents["source_hash"]
    documents["collected_at"] = now
    documents["run_id"] = run_id
    documents["known_at_timestamp"] = documents.apply(_known_at_timestamp, axis=1)
    meeting_ids = sorted(documents["meeting_id"].dropna().unique())
    meetings = meetings[meetings["meeting_id"].isin(meeting_ids)].copy()
    meetings["run_id"] = run_id
    meetings["collected_at"] = now
    return meetings.reset_index(drop=True), documents.reset_index(drop=True)


def ensure_v2_document_source_urls(documents: pd.DataFrame) -> pd.DataFrame:
    if documents.empty:
        return documents
    frame = documents.copy()
    if "source_url" not in frame:
        frame["source_url"] = ""
    if "url" not in frame:
        frame["url"] = ""
    frame["source_url"] = frame["source_url"].fillna("").astype(str)
    frame["url"] = frame["url"].fillna("").astype(str)
    missing = frame["source_url"].str.strip() == ""
    frame.loc[missing & frame["url"].str.strip().ne(""), "source_url"] = frame.loc[
        missing & frame["url"].str.strip().ne(""),
        "url",
    ]
    still_missing = frame["source_url"].str.strip() == ""
    if "document_type" in frame and "nro_reuniao" in frame:
        frame.loc[still_missing, "source_url"] = frame.loc[still_missing].apply(
            lambda row: copom_detail_url(str(row["document_type"]), int(row["nro_reuniao"])),
            axis=1,
        )
    return frame


def build_v2_sentences(documents: pd.DataFrame, run_id: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    now = utc_now_naive()
    for _, doc in documents.iterrows():
        clean_text = doc.get("clean_text")
        if not isinstance(clean_text, str) or not clean_text.strip():
            clean_text = clean_copom_html(str(doc.get("raw_text", "")))
        for order, sentence in enumerate(split_sentences(clean_text), start=1):
            rows.append(
                {
                    "sentence_id": f"{doc['document_id']}_{order:03d}",
                    "document_id": doc["document_id"],
                    "meeting_id": doc["meeting_id"],
                    "nro_reuniao": int(doc["nro_reuniao"]),
                    "document_type": doc["document_type"],
                    "sentence_order": order,
                    "text": sentence,
                    "sentence_hash": stable_text_hash(sentence),
                    "source_hash": doc.get("source_hash", stable_text_hash(str(doc.get("raw_text", "")))),
                    "run_id": run_id,
                    "created_at": now,
                }
            )
    return pd.DataFrame(rows)


def score_v2_sentences(
    sentences: pd.DataFrame,
    topics: dict[str, Any],
    lexicon: dict[str, Any],
    v2_settings: dict[str, Any],
    run_id: str,
) -> pd.DataFrame:
    if sentences.empty:
        return empty_v2_sentence_scores()
    neutral_band = float(v2_settings["scoring"]["neutral_band"])
    low_information_threshold = float(v2_settings["scoring"]["low_information_threshold"])
    rule_version = rule_engine_version(v2_settings)
    rows: list[dict[str, Any]] = []
    for _, sentence in sentences.iterrows():
        text = str(sentence["text"])
        taxonomy_boundary_flag = ""
        if is_institutional_or_header_sentence(text):
            matched_topics = ["institutional"]
            primary_topic = "institutional"
            stance = "neutral"
            stance_score = 0.0
            evidence_terms: list[str] = []
        else:
            matched_topics = topic_candidate_detection(text, topics)
            primary_topic = topic_priority_resolution(text, matched_topics, topics)
            if primary_topic not in matched_topics:
                matched_topics.append(primary_topic)
            taxonomy_boundary_flag = ambiguity_flagging(text, primary_topic, matched_topics)
            hawkish_score, hawkish_terms = score_terms(text, lexicon.get("hawkish", []))
            dovish_score, dovish_terms = score_terms(text, lexicon.get("dovish", []))
            total = hawkish_score + dovish_score
            stance_score = 0.0 if total == 0 else (hawkish_score - dovish_score) / total
            stance_score = float(max(-1.0, min(1.0, stance_score)))
            if abs(stance_score) < neutral_band:
                stance = "neutral"
                stance_score = 0.0
            elif stance_score > 0:
                stance = "hawkish"
            else:
                stance = "dovish"
            evidence_terms = (hawkish_terms + dovish_terms)[:10]
            stance_override = stance_direction_detection(text, primary_topic, matched_topics)
            if stance_override is not None:
                stance, stance_score, rule_term = stance_override
                evidence_terms = [rule_term, *evidence_terms][:10]
            guard_override = negation_and_reversal_guards(text, stance, primary_topic)
            if guard_override is not None:
                stance, stance_score, rule_term = guard_override
                evidence_terms = [rule_term, *evidence_terms][:10]
        confidence = confidence_score(evidence_terms, stance_score, matched_topics)
        information_weight = infer_information_weight(text, primary_topic, matched_topics, stance, low_information_threshold)
        is_informative = information_weight >= low_information_threshold
        topic_weight = float(topics.get(primary_topic, {}).get("weight", 1.0))
        denominator_weight = topic_weight * confidence * information_weight if is_informative else 0.0
        tone_level = stance_score * denominator_weight
        rows.append(
            {
                **sentence.to_dict(),
                "primary_topic": primary_topic,
                "topics": "|".join(matched_topics),
                "topic_count": len(matched_topics),
                "stance": stance,
                "stance_score": round(stance_score, 6),
                "confidence": round(confidence, 6),
                "is_informative": bool(is_informative),
                "information_weight": round(information_weight, 6),
                "topic_weight": round(topic_weight, 6),
                "denominator_weight": round(denominator_weight, 6),
                "tone_level": round(tone_level, 6),
                "novelty_score": np.nan,
                "taxonomy_boundary_flag": taxonomy_boundary_flag,
                "rationale": v2_rationale(stance, primary_topic, evidence_terms, is_informative),
                "evidence_terms": json.dumps(evidence_terms, ensure_ascii=False),
                "model_version": v2_settings["project"]["default_model_version"],
                "prompt_version": v2_settings["project"]["prompt_version"],
                "taxonomy_version": v2_settings["project"]["taxonomy_version"],
                "lexicon_version": v2_settings["project"]["lexicon_version"],
                "rule_engine_version": rule_version,
                "calibration_version": v2_settings["scoring"]["default_calibration"],
                "run_id": run_id,
            }
        )
    return pd.DataFrame(rows)


def aggregate_v2_scores(
    meetings: pd.DataFrame,
    documents: pd.DataFrame,
    sentence_scores: pd.DataFrame,
    v2_settings: dict[str, Any],
    run_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if sentence_scores.empty:
        return pd.DataFrame(), pd.DataFrame()
    grouped = sentence_scores.groupby(["document_id", "meeting_id", "document_type"], as_index=False).agg(
        weighted_sum=("tone_level", "sum"),
        weight_sum=("denominator_weight", "sum"),
        sentence_count=("sentence_id", "count"),
        informative_sentence_count=("is_informative", "sum"),
        directional_weighted_sum=("stance_score", lambda values: np.nan),
    )
    directional = (
        sentence_scores.assign(direction_abs=lambda frame: frame["stance_score"].abs() * frame["denominator_weight"])
        .groupby("document_id", as_index=False)
        .agg(direction_abs_sum=("direction_abs", "sum"))
    )
    grouped = grouped.drop(columns=["directional_weighted_sum"]).merge(directional, on="document_id", how="left")
    grouped["document_tone_raw"] = np.where(grouped["weight_sum"] > 0, grouped["weighted_sum"] / grouped["weight_sum"], np.nan)
    grouped["directional_intensity"] = np.where(
        grouped["weight_sum"] > 0,
        grouped["direction_abs_sum"] / grouped["weight_sum"],
        np.nan,
    )
    grouped["informative_share"] = np.where(
        grouped["sentence_count"] > 0,
        grouped["informative_sentence_count"] / grouped["sentence_count"],
        0.0,
    )
    grouped["score_status"] = np.where(grouped["informative_sentence_count"] > 0, "ok", "low_information")
    metadata_cols = [
        "document_id",
        "nro_reuniao",
        "publication_date",
        "title",
        "source",
        "source_hash",
        "model_version",
        "prompt_version",
        "taxonomy_version",
        "lexicon_version",
        "rule_engine_version",
        "calibration_version",
        "run_id",
    ]
    doc_meta = documents.copy()
    for col in ["model_version", "prompt_version", "taxonomy_version", "lexicon_version", "rule_engine_version", "calibration_version"]:
        if col not in doc_meta:
            doc_meta[col] = ""
    doc_meta["model_version"] = v2_settings["project"]["default_model_version"]
    doc_meta["prompt_version"] = v2_settings["project"]["prompt_version"]
    doc_meta["taxonomy_version"] = v2_settings["project"]["taxonomy_version"]
    doc_meta["lexicon_version"] = v2_settings["project"]["lexicon_version"]
    doc_meta["rule_engine_version"] = rule_engine_version(v2_settings)
    doc_meta["calibration_version"] = v2_settings["scoring"]["default_calibration"]
    document_scores = grouped.merge(doc_meta[[col for col in metadata_cols if col in doc_meta]], on="document_id", how="left")
    meetings = meetings.copy()
    if "data_referencia" in meetings:
        meetings["data_referencia"] = pd.to_datetime(meetings["data_referencia"], errors="coerce")
    rows: list[dict[str, Any]] = []
    communication_weight = float(v2_settings["scoring"]["communication_weight"])
    minutes_weight = float(v2_settings["scoring"]["minutes_weight"])
    for _, meeting in meetings.sort_values("data_referencia").iterrows():
        meeting_docs = document_scores[document_scores["meeting_id"] == meeting["meeting_id"]]
        comunicado = select_doc_score(meeting_docs, "comunicado")
        ata = select_doc_score(meeting_docs, "ata")
        if pd.notna(comunicado) and pd.notna(ata):
            tone_raw = communication_weight * comunicado + minutes_weight * ata
        elif pd.notna(comunicado):
            tone_raw = comunicado
        else:
            tone_raw = ata
        informative_count = int(meeting_docs["informative_sentence_count"].sum()) if not meeting_docs.empty else 0
        rows.append(
            {
                "meeting_id": meeting["meeting_id"],
                "nro_reuniao": int(meeting["nro_reuniao"]) if pd.notna(meeting.get("nro_reuniao")) else np.nan,
                "data_referencia": meeting.get("data_referencia", pd.NaT),
                "tone_comunicado": comunicado,
                "tone_ata": ata,
                "tone_raw": tone_raw,
                "tone_change": np.nan,
                "communication_surprise_naive": np.nan,
                "directional_intensity": float(meeting_docs["directional_intensity"].mean()) if not meeting_docs.empty else np.nan,
                "informative_sentence_count": informative_count,
                "score_status": "ok" if informative_count > 0 and pd.notna(tone_raw) else "low_information",
                "run_id": run_id,
                "model_version": v2_settings["project"]["default_model_version"],
                "prompt_version": v2_settings["project"]["prompt_version"],
                "taxonomy_version": v2_settings["project"]["taxonomy_version"],
                "lexicon_version": v2_settings["project"]["lexicon_version"],
                "rule_engine_version": rule_engine_version(v2_settings),
                "calibration_version": v2_settings["scoring"]["default_calibration"],
                "calibration_window": "",
            }
        )
    meeting_scores = pd.DataFrame(rows).sort_values("data_referencia").reset_index(drop=True)
    meeting_scores["tone_change"] = meeting_scores["tone_raw"].diff()
    meeting_scores["communication_surprise_naive"] = meeting_scores["tone_change"]
    return document_scores, meeting_scores


def build_v2_calibration(meeting_scores: pd.DataFrame, v2_settings: dict[str, Any], run_id: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    min_obs = int(v2_settings["scoring"]["min_calibration_observations"])
    scores = meeting_scores.copy()
    if "data_referencia" in scores:
        scores["data_referencia"] = pd.to_datetime(scores["data_referencia"], errors="coerce")
    for version, config in v2_settings["calibrations"].items():
        start = pd.Timestamp(config["start"])
        end = pd.Timestamp(config["end"])
        sample = scores[
            scores["tone_raw"].notna()
            & scores["data_referencia"].notna()
            & (scores["data_referencia"] >= start)
            & (scores["data_referencia"] <= end)
        ]["tone_raw"]
        status = "ok" if len(sample) >= min_obs and sample.std(ddof=0) > 0 else "insufficient_data"
        mean = float(sample.mean()) if len(sample) else np.nan
        std = float(sample.std(ddof=0)) if len(sample) > 1 else np.nan
        rows.append(
            {
                "calibration_version": version,
                "calibration_window": f"{start.date()}:{end.date()}",
                "start_date": start,
                "end_date": end,
                "mean": mean,
                "std": std,
                "q05": float(sample.quantile(0.05)) if len(sample) else np.nan,
                "q50": float(sample.quantile(0.50)) if len(sample) else np.nan,
                "q95": float(sample.quantile(0.95)) if len(sample) else np.nan,
                "observations": int(len(sample)),
                "official": bool(config.get("official", False)),
                "status": status,
                "run_id": run_id,
                "generated_at": utc_now_naive(),
            }
        )
    return pd.DataFrame(rows)


def apply_v2_calibration(
    meeting_scores: pd.DataFrame,
    calibration: pd.DataFrame,
    v2_settings: dict[str, Any],
) -> pd.DataFrame:
    scores = meeting_scores.copy()
    version = v2_settings["scoring"]["default_calibration"]
    row = calibration[calibration["calibration_version"] == version]
    if row.empty:
        mean = np.nan
        std = np.nan
        window = ""
        status = "missing_calibration"
    else:
        selected = row.iloc[0]
        mean = selected["mean"]
        std = selected["std"]
        window = selected["calibration_window"]
        status = selected["status"]
    if pd.isna(mean) or pd.isna(std) or float(std) == 0.0 or status != "ok":
        scores["tone_z"] = np.nan
        scores["copom_tone_index_v2"] = np.nan
        scores["classification_v2"] = "unavailable"
        scores["calibration_status"] = status
    else:
        scores["tone_z"] = (scores["tone_raw"] - float(mean)) / float(std)
        scores["copom_tone_index_v2"] = 50 + 10 * scores["tone_z"]
        scores["classification_v2"] = scores["copom_tone_index_v2"].map(classify_index)
        scores["calibration_status"] = "ok"
    scores["calibration_version"] = version
    scores["calibration_window"] = window
    return scores


def build_v2_subindices(
    sentence_scores: pd.DataFrame,
    meeting_scores: pd.DataFrame,
    v2_settings: dict[str, Any],
    run_id: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if sentence_scores.empty:
        return pd.DataFrame()
    for subindex, config in v2_settings["subindices"].items():
        topic_set = set(config["topics"])
        subset = sentence_scores[
            sentence_scores["is_informative"]
            & sentence_scores["topics"].fillna("").map(lambda value: bool(topic_set.intersection(str(value).split("|"))))
        ].copy()
        for meeting_id, group in subset.groupby("meeting_id"):
            weight = group["denominator_weight"].sum()
            tone = group["tone_level"].sum() / weight if weight > 0 else np.nan
            rows.append(
                {
                    "meeting_id": meeting_id,
                    "subindex": subindex,
                    "label": config["label"],
                    "tone_raw": tone,
                    "sentence_count": int(len(group)),
                    "weight_sum": float(weight),
                    "formula_version": "topic-weighted-v1",
                    "run_id": run_id,
                }
            )
    frame = pd.DataFrame(rows)
    reaction_rows = build_reaction_function_subindex(frame, meeting_scores, v2_settings, run_id)
    if reaction_rows.empty:
        return frame
    return pd.concat([frame, reaction_rows], ignore_index=True)


def build_reaction_function_subindex(
    subindices: pd.DataFrame,
    meeting_scores: pd.DataFrame,
    v2_settings: dict[str, Any],
    run_id: str,
) -> pd.DataFrame:
    weights = {key: float(value) for key, value in v2_settings["reaction_function"]["weights"].items()}
    rows: list[dict[str, Any]] = []
    if meeting_scores.empty:
        return pd.DataFrame()
    for meeting_id in meeting_scores["meeting_id"]:
        group = subindices[subindices["meeting_id"] == meeting_id]
        numerator = 0.0
        denominator = 0.0
        used = 0
        for subindex, weight in weights.items():
            value = group.loc[group["subindex"] == subindex, "tone_raw"]
            if value.empty or pd.isna(value.iloc[0]):
                continue
            numerator += weight * float(value.iloc[0])
            denominator += abs(weight)
            used += 1
        rows.append(
            {
                "meeting_id": meeting_id,
                "subindex": "text_implied_reaction_function",
                "label": v2_settings["reaction_function"]["label"],
                "tone_raw": numerator / denominator if denominator else np.nan,
                "sentence_count": int(used),
                "weight_sum": denominator,
                "formula_version": v2_settings["reaction_function"]["version"],
                "run_id": run_id,
            }
        )
    return pd.DataFrame(rows)


def build_v2_redline(
    documents: pd.DataFrame,
    sentence_scores: pd.DataFrame,
    v2_settings: dict[str, Any],
    run_id: str,
) -> pd.DataFrame:
    if documents.empty or sentence_scores.empty:
        return pd.DataFrame()
    maintained_similarity = float(v2_settings["redline"]["maintained_similarity"])
    related_similarity = float(v2_settings["redline"]["related_similarity"])
    tone_change_threshold = float(v2_settings["redline"]["tone_change_threshold"])
    rows: list[dict[str, Any]] = []
    docs = documents.sort_values(["document_type", "nro_reuniao"])
    scores_by_document = {str(document_id): group.copy() for document_id, group in sentence_scores.groupby("document_id")}
    for document_type, group in docs.groupby("document_type"):
        previous_doc_id: str | None = None
        for _, doc in group.iterrows():
            current = scores_by_document.get(str(doc["document_id"]), pd.DataFrame())
            if previous_doc_id is None:
                previous_doc_id = doc["document_id"]
                continue
            previous = scores_by_document.get(str(previous_doc_id), pd.DataFrame())
            rows.extend(
                redline_pair_rows(
                    current=current,
                    previous=previous,
                    document_type=document_type,
                    meeting_id=doc["meeting_id"],
                    nro_reuniao=int(doc["nro_reuniao"]),
                    maintained_similarity=maintained_similarity,
                    related_similarity=related_similarity,
                    tone_change_threshold=tone_change_threshold,
                    run_id=run_id,
                )
            )
            previous_doc_id = doc["document_id"]
    return pd.DataFrame(rows)


def redline_pair_rows(
    current: pd.DataFrame,
    previous: pd.DataFrame,
    document_type: str,
    meeting_id: str,
    nro_reuniao: int,
    maintained_similarity: float,
    related_similarity: float,
    tone_change_threshold: float,
    run_id: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    used_previous: set[str] = set()
    previous_index = build_previous_redline_index(previous)
    for _, current_row in current.iterrows():
        best = best_previous_match_from_index(str(current_row["text"]), previous_index, used_previous)
        previous_row = best[0]
        similarity = best[1]
        if previous_row is None or similarity < related_similarity:
            change_type = "added"
            previous_sentence_id = ""
            previous_text = ""
            previous_score = np.nan
            tone_delta = np.nan
        else:
            previous_sentence_id = previous_row["sentence_id"]
            used_previous.add(previous_sentence_id)
            previous_text = previous_row["text"]
            previous_score = previous_row["stance_score"]
            tone_delta = float(current_row["stance_score"]) - float(previous_score)
            if similarity >= maintained_similarity and abs(tone_delta) < tone_change_threshold:
                change_type = "maintained"
            elif abs(tone_delta) >= tone_change_threshold:
                change_type = "tone_changed"
            else:
                change_type = "rewritten"
        rows.append(
            {
                "meeting_id": meeting_id,
                "nro_reuniao": nro_reuniao,
                "document_type": document_type,
                "change_type": change_type,
                "current_sentence_id": current_row["sentence_id"],
                "previous_sentence_id": previous_sentence_id,
                "similarity": round(float(similarity), 6),
                "tone_delta": tone_delta,
                "current_stance_score": current_row["stance_score"],
                "previous_stance_score": previous_score,
                "current_topic": current_row["primary_topic"],
                "current_text": current_row["text"],
                "previous_text": previous_text,
                "run_id": run_id,
            }
        )
    for previous_row in previous_index["records"]:
        if previous_row["sentence_id"] in used_previous:
            continue
        rows.append(
            {
                "meeting_id": meeting_id,
                "nro_reuniao": nro_reuniao,
                "document_type": document_type,
                "change_type": "removed",
                "current_sentence_id": "",
                "previous_sentence_id": previous_row["sentence_id"],
                "similarity": np.nan,
                "tone_delta": np.nan,
                "current_stance_score": np.nan,
                "previous_stance_score": previous_row["stance_score"],
                "current_topic": previous_row["primary_topic"],
                "current_text": "",
                "previous_text": previous_row["text"],
                "run_id": run_id,
            }
        )
    return rows


def build_v2_evidence(sentence_scores: pd.DataFrame, meeting_scores: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    if sentence_scores.empty:
        return pd.DataFrame()
    informative = sentence_scores[sentence_scores["is_informative"]].copy()
    informative["abs_tone_level"] = informative["tone_level"].abs()
    hawkish = (
        informative[informative["tone_level"] > 0]
        .sort_values(["meeting_id", "tone_level", "confidence"], ascending=[True, False, False])
        .groupby("meeting_id")
        .head(top_n)
        .assign(evidence_type="hawkish")
    )
    dovish = (
        informative[informative["tone_level"] < 0]
        .sort_values(["meeting_id", "tone_level", "confidence"], ascending=[True, True, False])
        .groupby("meeting_id")
        .head(top_n)
        .assign(evidence_type="dovish")
    )
    evidence = pd.concat([hawkish, dovish], ignore_index=True) if not hawkish.empty or not dovish.empty else pd.DataFrame()
    if evidence.empty:
        return evidence
    evidence["citation"] = evidence.apply(
        lambda row: f"Reuniao {row.get('nro_reuniao', '')}, {row.get('document_type', '')}, sentenca {row.get('sentence_id', '')}",
        axis=1,
    )
    return evidence[
        [
            "meeting_id",
            "nro_reuniao",
            "document_type",
            "sentence_id",
            "citation",
            "evidence_type",
            "primary_topic",
            "topics",
            "stance",
            "stance_score",
            "confidence",
            "information_weight",
            "tone_level",
            "text",
            "rationale",
            "model_version",
            "prompt_version",
            "taxonomy_version",
            "lexicon_version",
            "rule_engine_version",
            "calibration_version",
            "run_id",
        ]
    ]


def build_v2_model_predictions(sentence_scores: pd.DataFrame) -> pd.DataFrame:
    if sentence_scores.empty:
        return pd.DataFrame(
            columns=[
                "sentence_id",
                "predicted_topic",
                "predicted_stance",
                "predicted_is_informative",
                "taxonomy_boundary_flag",
                "prediction_source",
                "model_version",
                "rule_engine_version",
            ]
        )
    frame = sentence_scores.copy()
    for column in ["taxonomy_boundary_flag", "rule_engine_version"]:
        if column not in frame:
            frame[column] = ""
    return frame[
        [
            "sentence_id",
            "primary_topic",
            "stance",
            "is_informative",
            "taxonomy_boundary_flag",
            "model_version",
            "prompt_version",
            "taxonomy_version",
            "lexicon_version",
            "rule_engine_version",
            "run_id",
        ]
    ].rename(
        columns={
            "primary_topic": "predicted_topic",
            "stance": "predicted_stance",
            "is_informative": "predicted_is_informative",
        }
    ).assign(prediction_source="baseline")


def build_v2_model_audit(labels: pd.DataFrame, predictions: pd.DataFrame, run_id: str) -> pd.DataFrame:
    columns = ["run_id", "metric", "value", "status", "detail"]
    if labels.empty or predictions.empty:
        return pd.DataFrame(
            [
                {
                    "run_id": run_id,
                    "metric": "label_coverage",
                    "value": 0.0,
                    "status": "needs_labels",
                    "detail": "No accepted human labels are available; LLM bootstrap labels are not treated as ground truth.",
                }
            ],
            columns=columns,
        )
    accepted = accepted_v2_labels(labels)
    if accepted.empty:
        return pd.DataFrame(
            [
                {
                    "run_id": run_id,
                    "metric": "label_coverage",
                    "value": 0.0,
                    "status": "needs_human_acceptance",
                    "detail": "Labels exist but none are accepted or human reviewed.",
                }
            ],
            columns=columns,
        )
    consensus = consensus_v2_labels(accepted)
    merged = consensus.merge(predictions, on="sentence_id", how="inner")
    if merged.empty:
        return pd.DataFrame(
            [
                {
                    "run_id": run_id,
                    "metric": "matched_label_coverage",
                    "value": 0.0,
                    "status": "invalid_for_audit",
                    "detail": "Accepted labels do not match current predictions.",
                }
            ],
            columns=columns,
        )
    rows: list[dict[str, object]] = [
        {"run_id": run_id, "metric": "accepted_labels", "value": float(len(accepted)), "status": "ok", "detail": ""},
        {
            "run_id": run_id,
            "metric": "unique_accepted_sentences",
            "value": float(len(consensus)),
            "status": "ok",
            "detail": "Model accuracy metrics use one consensus label per sentence, so double review does not overweight a sentence.",
        },
        {
            "run_id": run_id,
            "metric": "label_conflict_sentences",
            "value": float(consensus["label_conflict"].sum()) if "label_conflict" in consensus else 0.0,
            "status": "ok",
            "detail": "Number of accepted sentences with reviewer disagreement in topic, stance or informativeness.",
        },
        {"run_id": run_id, "metric": "matched_labels", "value": float(len(merged)), "status": "ok", "detail": ""},
        {
            "run_id": run_id,
            "metric": "stance_accuracy",
            "value": accuracy_metric(merged, "stance_label", "predicted_stance"),
            "status": "ok",
            "detail": "",
        },
        {
            "run_id": run_id,
            "metric": "topic_accuracy",
            "value": accuracy_metric(merged, "topic_label", "predicted_topic"),
            "status": "ok",
            "detail": "",
        },
        {
            "run_id": run_id,
            "metric": "informativeness_accuracy",
            "value": accuracy_metric(merged, "is_informative_label", "predicted_is_informative"),
            "status": "ok",
            "detail": "",
        },
        {
            "run_id": run_id,
            "metric": "stance_f1_macro",
            "value": macro_f1_metric(merged["stance_label"], merged["predicted_stance"]),
            "status": "ok",
            "detail": "",
        },
    ]
    agreement = human_stance_agreement(accepted)
    if agreement is not None:
        rows.append(
            {
                "run_id": run_id,
                "metric": "human_stance_agreement",
                "value": agreement,
                "status": "ok",
                "detail": "Share of double-reviewed sentences with identical stance labels.",
            }
        )
    for item in confusion_matrix_records(merged, "stance_label", "predicted_stance"):
        rows.append(
            {
                "run_id": run_id,
                "metric": "stance_confusion",
                "value": float(item["count"]),
                "status": "ok",
                "detail": json.dumps({"actual": item["actual"], "predicted": item["predicted"]}, ensure_ascii=False, sort_keys=True),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def build_v2_model_audit_details(labels: pd.DataFrame, predictions: pd.DataFrame, run_id: str) -> pd.DataFrame:
    columns = [
        "run_id",
        "sentence_id",
        "label_source",
        "label_status",
        "annotator_id",
        "reviewer_count",
        "label_conflict",
        "topic_label",
        "predicted_topic",
        "topic_correct",
        "stance_label",
        "predicted_stance",
        "stance_correct",
        "is_informative_label",
        "predicted_is_informative",
        "informativeness_correct",
    ]
    if labels.empty or predictions.empty:
        return pd.DataFrame(columns=columns)
    accepted = accepted_v2_labels(labels)
    if accepted.empty:
        return pd.DataFrame(columns=columns)
    consensus = consensus_v2_labels(accepted)
    merged = consensus.merge(predictions, on="sentence_id", how="inner")
    if merged.empty:
        return pd.DataFrame(columns=columns)
    details = merged.copy()
    details["run_id"] = run_id
    details["topic_correct"] = details["topic_label"].astype(str) == details["predicted_topic"].astype(str)
    details["stance_correct"] = details["stance_label"].astype(str) == details["predicted_stance"].astype(str)
    details["informativeness_correct"] = (
        details["is_informative_label"].astype(bool) == details["predicted_is_informative"].astype(bool)
    )
    return details[columns]


def build_v2_model_audit_error_analysis(
    labels: pd.DataFrame,
    predictions: pd.DataFrame,
    sentence_scores: pd.DataFrame,
    run_id: str,
) -> pd.DataFrame:
    columns = [
        "run_id",
        "sentence_id",
        "meeting_id",
        "nro_reuniao",
        "document_type",
        "sentence_order",
        "issue_types",
        "issue_count",
        "priority_score",
        "suggested_action",
        "topic_label",
        "predicted_topic",
        "stance_label",
        "predicted_stance",
        "is_informative_label",
        "predicted_is_informative",
        "tone_level",
        "confidence",
        "information_weight",
        "evidence_terms",
        "taxonomy_boundary_flag",
        "text",
    ]
    if labels.empty or predictions.empty:
        return pd.DataFrame(columns=columns)
    consensus = consensus_v2_labels(labels)
    if consensus.empty:
        return pd.DataFrame(columns=columns)
    merged = consensus.merge(predictions, on="sentence_id", how="inner")
    if merged.empty:
        return pd.DataFrame(columns=columns)
    context_columns = [
        "sentence_id",
        "meeting_id",
        "nro_reuniao",
        "document_type",
        "sentence_order",
        "text",
        "tone_level",
        "confidence",
        "information_weight",
        "evidence_terms",
        "taxonomy_boundary_flag",
    ]
    context = sentence_scores[[col for col in context_columns if col in sentence_scores]].drop_duplicates("sentence_id")
    if not context.empty:
        merged = merged.merge(context, on="sentence_id", how="left")
    rows: list[dict[str, object]] = []
    for _, row in merged.iterrows():
        issue_types = audit_issue_types(row)
        if not issue_types:
            continue
        priority = audit_error_priority(row, issue_types)
        rows.append(
            {
                "run_id": run_id,
                "sentence_id": row.get("sentence_id", ""),
                "meeting_id": row.get("meeting_id", ""),
                "nro_reuniao": row.get("nro_reuniao", np.nan),
                "document_type": row.get("document_type", ""),
                "sentence_order": row.get("sentence_order", np.nan),
                "issue_types": "|".join(issue_types),
                "issue_count": len(issue_types),
                "priority_score": round(priority, 6),
                "suggested_action": suggested_audit_action(row, issue_types),
                "topic_label": row.get("topic_label", ""),
                "predicted_topic": row.get("predicted_topic", ""),
                "stance_label": row.get("stance_label", ""),
                "predicted_stance": row.get("predicted_stance", ""),
                "is_informative_label": bool(row.get("is_informative_label", False)),
                "predicted_is_informative": bool(row.get("predicted_is_informative", False)),
                "tone_level": row.get("tone_level", np.nan),
                "confidence": row.get("confidence", np.nan),
                "information_weight": row.get("information_weight", np.nan),
                "evidence_terms": row.get("evidence_terms", ""),
                "taxonomy_boundary_flag": row.get("taxonomy_boundary_flag", ""),
                "text": row.get("text", ""),
            }
        )
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["priority_score", "issue_count", "sentence_id"],
        ascending=[False, False, True],
    )


def build_v2_model_audit_error_summary(error_analysis: pd.DataFrame, run_id: str) -> pd.DataFrame:
    columns = ["run_id", "issue_type", "truth", "prediction", "count", "share", "suggested_action"]
    if error_analysis.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, object]] = []
    total = len(error_analysis)
    summary_specs = [
        ("topic_mismatch", "topic_label", "predicted_topic"),
        ("stance_mismatch", "stance_label", "predicted_stance"),
        ("informativeness_mismatch", "is_informative_label", "predicted_is_informative"),
    ]
    for issue_type, truth_col, pred_col in summary_specs:
        subset = error_analysis[error_analysis["issue_types"].astype(str).str.contains(issue_type, regex=False)]
        if subset.empty:
            continue
        grouped = subset.groupby([truth_col, pred_col, "suggested_action"], dropna=False).size().reset_index(name="count")
        for _, row in grouped.sort_values("count", ascending=False).iterrows():
            rows.append(
                {
                    "run_id": run_id,
                    "issue_type": issue_type,
                    "truth": str(row[truth_col]),
                    "prediction": str(row[pred_col]),
                    "count": int(row["count"]),
                    "share": round(float(row["count"]) / total, 6),
                    "suggested_action": row["suggested_action"],
                }
            )
    return pd.DataFrame(rows, columns=columns)


def build_v2_reviewer_disagreements(
    labels: pd.DataFrame,
    predictions: pd.DataFrame,
    sentence_scores: pd.DataFrame,
    run_id: str,
) -> pd.DataFrame:
    columns = [
        "run_id",
        "sentence_id",
        "meeting_id",
        "nro_reuniao",
        "document_type",
        "sentence_order",
        "reviewer_count",
        "conflict_types",
        "topic_labels",
        "stance_labels",
        "informativeness_labels",
        "annotator_ids",
        "consensus_topic",
        "consensus_stance",
        "consensus_is_informative",
        "predicted_topic",
        "predicted_stance",
        "predicted_is_informative",
        "text",
    ]
    accepted = accepted_v2_labels(labels)
    if accepted.empty:
        return pd.DataFrame(columns=columns)
    consensus = consensus_v2_labels(accepted)
    context_columns = [
        "sentence_id",
        "meeting_id",
        "nro_reuniao",
        "document_type",
        "sentence_order",
        "text",
    ]
    context = sentence_scores[[col for col in context_columns if col in sentence_scores]].drop_duplicates("sentence_id")
    prediction_context = predictions.drop_duplicates("sentence_id") if not predictions.empty else pd.DataFrame()
    rows: list[dict[str, object]] = []
    for sentence_id, group in accepted.groupby("sentence_id", sort=True):
        conflict_types = human_conflict_types(group)
        if not conflict_types:
            continue
        consensus_row = consensus[consensus["sentence_id"] == sentence_id]
        row = consensus_row.iloc[0].to_dict() if not consensus_row.empty else {}
        context_row = context[context["sentence_id"] == sentence_id]
        pred_row = prediction_context[prediction_context["sentence_id"] == sentence_id] if not prediction_context.empty else pd.DataFrame()
        context_data = context_row.iloc[0].to_dict() if not context_row.empty else {}
        pred_data = pred_row.iloc[0].to_dict() if not pred_row.empty else {}
        rows.append(
            {
                "run_id": run_id,
                "sentence_id": sentence_id,
                "meeting_id": context_data.get("meeting_id", ""),
                "nro_reuniao": context_data.get("nro_reuniao", np.nan),
                "document_type": context_data.get("document_type", ""),
                "sentence_order": context_data.get("sentence_order", np.nan),
                "reviewer_count": int(row.get("reviewer_count", group["annotator_id"].nunique())),
                "conflict_types": "|".join(conflict_types),
                "topic_labels": "|".join(sorted(set(group["topic_label"].fillna("").astype(str)))),
                "stance_labels": "|".join(sorted(set(group["stance_label"].fillna("").astype(str)))),
                "informativeness_labels": "|".join(sorted(set(group["is_informative_label"].map(bool).astype(str)))),
                "annotator_ids": "|".join(sorted(value for value in set(group["annotator_id"].fillna("").astype(str)) if value)),
                "consensus_topic": row.get("topic_label", ""),
                "consensus_stance": row.get("stance_label", ""),
                "consensus_is_informative": bool(row.get("is_informative_label", False)),
                "predicted_topic": pred_data.get("predicted_topic", ""),
                "predicted_stance": pred_data.get("predicted_stance", ""),
                "predicted_is_informative": bool(pred_data.get("predicted_is_informative", False)),
                "text": context_data.get("text", ""),
            }
        )
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values(["conflict_types", "sentence_id"])


def build_v2_model_audit_error_classification(
    error_analysis: pd.DataFrame,
    reviewer_disagreements: pd.DataFrame,
    run_id: str,
) -> pd.DataFrame:
    if error_analysis.empty:
        return pd.DataFrame(
            columns=list(error_analysis.columns) + ["human_conflict_types", "error_interpretation", "taxonomy_boundary_flag", "taxonomy_boundary"]
        )
    disagreement_map = {}
    if not reviewer_disagreements.empty:
        disagreement_map = dict(
            zip(
                reviewer_disagreements["sentence_id"].astype(str),
                reviewer_disagreements["conflict_types"].astype(str),
                strict=False,
            )
        )
    rows: list[dict[str, object]] = []
    for _, row in error_analysis.iterrows():
        item = row.to_dict()
        conflict_types = disagreement_map.get(str(row.get("sentence_id", "")), "")
        issue_types = str(row.get("issue_types", ""))
        taxonomy_boundary = taxonomy_boundary_case(row)
        interpretation = classify_baseline_error(issue_types, conflict_types, taxonomy_boundary)
        item["run_id"] = run_id
        item["human_conflict_types"] = conflict_types
        item["taxonomy_boundary_flag"] = taxonomy_boundary
        item["taxonomy_boundary"] = taxonomy_boundary
        item["error_interpretation"] = interpretation
        rows.append(item)
    return pd.DataFrame(rows)


def human_conflict_types(group: pd.DataFrame) -> list[str]:
    conflicts: list[str] = []
    if group["topic_label"].fillna("").astype(str).nunique() > 1:
        conflicts.append("topic_conflict")
    if group["stance_label"].fillna("").astype(str).nunique() > 1:
        conflicts.append("stance_conflict")
    if group["is_informative_label"].map(bool).nunique() > 1:
        conflicts.append("informativeness_conflict")
    return conflicts


def classify_baseline_error(issue_types: str, human_conflict_types_text: str, taxonomy_boundary: str) -> str:
    if human_conflict_types_text:
        if conflict_overlaps_issue(issue_types, human_conflict_types_text):
            return "legitimate_taxonomy_ambiguity"
        return "baseline_error_with_reviewer_disagreement_context"
    if taxonomy_boundary:
        return "taxonomy_boundary_case"
    return "likely_baseline_error"


def conflict_overlaps_issue(issue_types: str, conflict_types: str) -> bool:
    pairs = [
        ("topic_mismatch", "topic_conflict"),
        ("stance_mismatch", "stance_conflict"),
        ("informativeness_mismatch", "informativeness_conflict"),
    ]
    return any(issue in issue_types and conflict in conflict_types for issue, conflict in pairs)


def taxonomy_boundary_case(row: pd.Series) -> str:
    explicit = str(row.get("taxonomy_boundary_flag", "") or "").strip()
    if explicit:
        return explicit
    topic_pair = {str(row.get("topic_label", "")), str(row.get("predicted_topic", ""))}
    boundary_pairs = [
        ({"policy_decision", "forward_guidance"}, "policy_decision_vs_forward_guidance"),
        ({"external_environment", "uncertainty"}, "external_environment_vs_uncertainty"),
        ({"inflation_current", "risk_balance"}, "inflation_current_vs_risk_balance"),
        ({"fiscal_risk", "external_environment"}, "fiscal_risk_vs_external_environment"),
        ({"activity_growth", "labor_market"}, "activity_growth_vs_labor_market"),
        ({"inflation_current", "inflation_expectations"}, "current_inflation_vs_expectations"),
    ]
    for pair, label in boundary_pairs:
        if topic_pair == pair:
            return label
    return ""


def audit_issue_types(row: pd.Series) -> list[str]:
    issues: list[str] = []
    if str(row.get("topic_label", "")) != str(row.get("predicted_topic", "")):
        issues.append("topic_mismatch")
    if str(row.get("stance_label", "")) != str(row.get("predicted_stance", "")):
        issues.append("stance_mismatch")
    if bool(row.get("is_informative_label", False)) != bool(row.get("predicted_is_informative", False)):
        issues.append("informativeness_mismatch")
    return issues


def audit_error_priority(row: pd.Series, issue_types: list[str]) -> float:
    confidence = numeric_or_zero(row.get("confidence", 0.0))
    tone = abs(numeric_or_zero(row.get("tone_level", 0.0)))
    information = numeric_or_zero(row.get("information_weight", 0.0))
    issue_weight = 1.0 + 0.35 * len(issue_types)
    if "stance_mismatch" in issue_types:
        issue_weight += 0.50
    if "informativeness_mismatch" in issue_types:
        issue_weight += 0.40
    return issue_weight + confidence + information + tone


def numeric_or_zero(value: object) -> float:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return 0.0 if pd.isna(number) else float(number)


def safe_divide(numerator: object, denominator: object) -> float:
    den = numeric_or_zero(denominator)
    if den == 0.0:
        return float("nan")
    return float(numeric_or_zero(numerator) / den)


def value_counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if frame.empty or column not in frame:
        return {}
    return {str(key): int(value) for key, value in frame[column].fillna("").astype(str).value_counts().items()}


def suggested_audit_action(row: pd.Series, issue_types: list[str]) -> str:
    if "informativeness_mismatch" in issue_types:
        if not bool(row.get("is_informative_label", False)) and bool(row.get("predicted_is_informative", False)):
            return "tighten_low_information_or_header_filter"
        return "recover_macro_sentences_marked_low_information"
    if "stance_mismatch" in issue_types:
        truth = str(row.get("stance_label", ""))
        pred = str(row.get("predicted_stance", ""))
        if truth in {"hawkish", "dovish"} and pred == "neutral":
            return "add_missing_directional_terms_or_context_rule"
        if truth == "neutral" and pred in {"hawkish", "dovish"}:
            return "reduce_directional_false_positive_terms"
        return "review_directional_lexicon_or_context_rule"
    if "topic_mismatch" in issue_types:
        return "review_topic_keywords_or_priority"
    return "inspect_label_prediction_pair"


def build_v2_model_audit_report_html(
    audit: pd.DataFrame,
    error_analysis: pd.DataFrame,
    error_summary: pd.DataFrame,
    run_id: str,
) -> str:
    metrics = dict(zip(audit.get("metric", pd.Series(dtype=str)), audit.get("value", pd.Series(dtype=float)), strict=False))
    top_errors = error_analysis.head(25)
    top_summary = error_summary.sort_values("count", ascending=False).head(20) if not error_summary.empty else error_summary
    return f"""<!doctype html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8">
  <title>COPOM Watch V2 Model Audit</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #1f2933; }}
    h1, h2 {{ color: #102a43; }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0; font-size: 13px; }}
    th, td {{ border: 1px solid #d9e2ec; padding: 8px; vertical-align: top; }}
    th {{ background: #f0f4f8; text-align: left; }}
    .badge {{ display: inline-block; padding: 4px 8px; border-radius: 4px; background: #fff3cd; color: #513c06; }}
    .metric {{ display: inline-block; margin: 4px 12px 4px 0; }}
  </style>
</head>
<body>
  <h1>COPOM Watch V2 Model Audit</h1>
  <p><span class="badge">Diagnostico, nao validacao causal</span></p>
  <p>Run ID: <code>{html.escape(run_id)}</code></p>
  <h2>Resumo</h2>
  <p>
    <span class="metric">Labels aceitos: <strong>{format_metric(metrics.get("accepted_labels"))}</strong></span>
    <span class="metric">Sentencas unicas: <strong>{format_metric(metrics.get("unique_accepted_sentences"))}</strong></span>
    <span class="metric">Acuracia stance: <strong>{format_metric(metrics.get("stance_accuracy"), pct=True)}</strong></span>
    <span class="metric">Acuracia topico: <strong>{format_metric(metrics.get("topic_accuracy"), pct=True)}</strong></span>
    <span class="metric">F1 macro stance: <strong>{format_metric(metrics.get("stance_f1_macro"), pct=True)}</strong></span>
  </p>
  <h2>Resumo De Erros</h2>
  {dataframe_to_html_table(top_summary)}
  <h2>Erros Priorizados</h2>
  {dataframe_to_html_table(top_errors)}
</body>
</html>
"""


def build_v2_validation_disagreement_report_html(
    audit: pd.DataFrame,
    reviewer_disagreements: pd.DataFrame,
    error_classification: pd.DataFrame,
    run_id: str,
) -> str:
    metrics = dict(zip(audit.get("metric", pd.Series(dtype=str)), audit.get("value", pd.Series(dtype=float)), strict=False))
    interpretation_summary = (
        error_classification.groupby("error_interpretation", dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        if not error_classification.empty
        else pd.DataFrame()
    )
    boundary_summary = (
        error_classification[error_classification["taxonomy_boundary"].astype(str).ne("")]
        .groupby("taxonomy_boundary", dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        if not error_classification.empty and "taxonomy_boundary" in error_classification
        else pd.DataFrame()
    )
    conflict_summary = (
        reviewer_disagreements.groupby("conflict_types", dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        if not reviewer_disagreements.empty
        else pd.DataFrame()
    )
    top_ambiguous = (
        error_classification[error_classification["error_interpretation"].astype(str).ne("likely_baseline_error")]
        .head(25)
        if not error_classification.empty
        else pd.DataFrame()
    )
    return f"""<!doctype html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8">
  <title>COPOM Watch V2 Validation Disagreement Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #1f2933; }}
    h1, h2 {{ color: #102a43; }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0; font-size: 13px; }}
    th, td {{ border: 1px solid #d9e2ec; padding: 8px; vertical-align: top; }}
    th {{ background: #f0f4f8; text-align: left; }}
    .badge {{ display: inline-block; padding: 4px 8px; border-radius: 4px; background: #e0f2fe; color: #0b4f71; }}
    .metric {{ display: inline-block; margin: 4px 12px 4px 0; }}
  </style>
</head>
<body>
  <h1>COPOM Watch V2 Validation Disagreement Report</h1>
  <p><span class="badge">Separacao entre erro provavel do baseline e ambiguidade da taxonomia</span></p>
  <p>Run ID: <code>{html.escape(run_id)}</code></p>
  <h2>Resumo</h2>
  <p>
    <span class="metric">Labels aceitos: <strong>{format_metric(metrics.get("accepted_labels"))}</strong></span>
    <span class="metric">Sentencas unicas: <strong>{format_metric(metrics.get("unique_accepted_sentences"))}</strong></span>
    <span class="metric">Concordancia humana stance: <strong>{format_metric(metrics.get("human_stance_agreement"), pct=True)}</strong></span>
    <span class="metric">Erros baseline vs consenso: <strong>{len(error_classification)}</strong></span>
    <span class="metric">Conflitos entre revisores: <strong>{len(reviewer_disagreements)}</strong></span>
  </p>
  <h2>Classificacao Dos Erros</h2>
  {dataframe_to_html_table(interpretation_summary)}
  <h2>Conflitos Entre Revisores</h2>
  {dataframe_to_html_table(conflict_summary)}
  <h2>Fronteiras De Taxonomia</h2>
  {dataframe_to_html_table(boundary_summary)}
  <h2>Casos Ambiguos Ou De Fronteira</h2>
  {dataframe_to_html_table(top_ambiguous)}
</body>
</html>
"""


def build_baseline_benchmark_tables(
    labels: pd.DataFrame,
    predictions: pd.DataFrame,
    sentence_scores: pd.DataFrame,
    run_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    accepted = accepted_v2_labels(labels)
    if accepted.empty or predictions.empty:
        empty_sample = pd.DataFrame(
            columns=[
                "run_id",
                "sample",
                "labels",
                "unique_sentences",
                "matched_labels",
                "topic_accuracy",
                "stance_accuracy",
                "informativeness_accuracy",
                "stance_f1_macro",
                "total_errors",
                "likely_baseline_error",
                "taxonomy_boundary_case",
                "legitimate_taxonomy_ambiguity",
                "error_rate",
            ]
        )
        return empty_sample, pd.DataFrame(), pd.DataFrame()
    slices = benchmark_label_slices(accepted)
    sample_rows: list[dict[str, object]] = []
    topic_rows: list[dict[str, object]] = []
    stance_rows: list[dict[str, object]] = []
    for sample_name, sample_labels in slices.items():
        if sample_labels.empty:
            continue
        consensus = consensus_v2_labels(sample_labels)
        merged = consensus.merge(predictions, on="sentence_id", how="inner")
        error_analysis = build_v2_model_audit_error_analysis(sample_labels, predictions, sentence_scores, run_id)
        disagreements = build_v2_reviewer_disagreements(sample_labels, predictions, sentence_scores, run_id)
        classified = build_v2_model_audit_error_classification(error_analysis, disagreements, run_id)
        interpretation_counts = value_counts(classified, "error_interpretation")
        sample_rows.append(
            {
                "run_id": run_id,
                "sample": sample_name,
                "labels": int(len(sample_labels)),
                "unique_sentences": int(len(consensus)),
                "matched_labels": int(len(merged)),
                "topic_accuracy": accuracy_metric(merged, "topic_label", "predicted_topic"),
                "stance_accuracy": accuracy_metric(merged, "stance_label", "predicted_stance"),
                "informativeness_accuracy": accuracy_metric(merged, "is_informative_label", "predicted_is_informative"),
                "stance_f1_macro": macro_f1_metric(merged["stance_label"], merged["predicted_stance"]) if not merged.empty else np.nan,
                "total_errors": int(len(error_analysis)),
                "likely_baseline_error": int(interpretation_counts.get("likely_baseline_error", 0)),
                "taxonomy_boundary_case": int(interpretation_counts.get("taxonomy_boundary_case", 0)),
                "legitimate_taxonomy_ambiguity": int(interpretation_counts.get("legitimate_taxonomy_ambiguity", 0)),
                "error_rate": safe_divide(len(error_analysis), len(merged)),
            }
        )
        topic_rows.extend(benchmark_topic_rows(sample_name, merged, run_id))
        stance_rows.extend(benchmark_stance_rows(sample_name, merged, run_id))
    return pd.DataFrame(sample_rows), pd.DataFrame(topic_rows), pd.DataFrame(stance_rows)


def benchmark_label_slices(accepted: pd.DataFrame) -> dict[str, pd.DataFrame]:
    annotators = accepted.get("annotator_id", pd.Series(dtype=str)).fillna("").astype(str).str.lower()
    sample_002_mask = annotators.str.contains("holdout_002|sample_002|002", regex=True)
    claude_mask = annotators.str.contains("claude_holdout_002|claude.*002", regex=True)
    gpt_mask = annotators.str.contains("gpt_holdout_002|gpt.*002", regex=True)
    sample_002_sentence_ids = set(accepted.loc[sample_002_mask, "sentence_id"].dropna().astype(str))
    slices = {
        "sample_001": accepted[~sample_002_mask].copy(),
        "sample_002_claude": accepted[claude_mask].copy(),
        "sample_002_gpt": accepted[gpt_mask].copy(),
        "sample_002_consensus": accepted[accepted["sentence_id"].astype(str).isin(sample_002_sentence_ids)].copy(),
        "total_consensus": accepted.copy(),
    }
    return slices


def benchmark_topic_rows(sample_name: str, merged: pd.DataFrame, run_id: str) -> list[dict[str, object]]:
    if merged.empty:
        return []
    rows: list[dict[str, object]] = []
    for topic, group in merged.groupby("topic_label", dropna=False):
        rows.append(
            {
                "run_id": run_id,
                "sample": sample_name,
                "topic": str(topic),
                "observations": int(len(group)),
                "topic_accuracy": accuracy_metric(group, "topic_label", "predicted_topic"),
                "stance_accuracy": accuracy_metric(group, "stance_label", "predicted_stance"),
                "informativeness_accuracy": accuracy_metric(group, "is_informative_label", "predicted_is_informative"),
                "neutral_share": safe_divide((group["stance_label"].astype(str) == "neutral").sum(), len(group)),
            }
        )
    return rows


def benchmark_stance_rows(sample_name: str, merged: pd.DataFrame, run_id: str) -> list[dict[str, object]]:
    if merged.empty:
        return []
    rows: list[dict[str, object]] = []
    true = merged["stance_label"].astype(str)
    pred = merged["predicted_stance"].astype(str)
    for stance in sorted(set(true) | set(pred)):
        tp = int(((true == stance) & (pred == stance)).sum())
        fp = int(((true != stance) & (pred == stance)).sum())
        fn = int(((true == stance) & (pred != stance)).sum())
        precision = safe_divide(tp, tp + fp)
        recall = safe_divide(tp, tp + fn)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append(
            {
                "run_id": run_id,
                "sample": sample_name,
                "stance": stance,
                "observations": int((true == stance).sum()),
                "predicted_observations": int((pred == stance).sum()),
                "true_positive": tp,
                "false_positive": fp,
                "false_negative": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return rows


def evaluate_benchmark_gates(
    by_sample: pd.DataFrame,
    by_stance: pd.DataFrame,
    previous_sample: pd.DataFrame,
    previous_stance: pd.DataFrame,
) -> dict[str, object]:
    warnings: list[str] = []
    errors: list[str] = []
    total = by_sample[by_sample["sample"] == "total_consensus"].iloc[0].to_dict() if "sample" in by_sample and (by_sample["sample"] == "total_consensus").any() else {}
    if total:
        if pd.notna(total.get("likely_baseline_error")) and float(total.get("likely_baseline_error", 0)) >= 180:
            warnings.append("likely_baseline_error remains above the V2.0.3 acceptance target of 180.")
        if pd.notna(total.get("stance_f1_macro")) and float(total.get("stance_f1_macro", 0)) < 0.70:
            warnings.append("stance_f1_macro remains below the V2.0.3 acceptance target of 70%.")
        if pd.notna(total.get("topic_accuracy")) and float(total.get("topic_accuracy", 0)) < 0.68:
            warnings.append("topic_accuracy remains below the V2.0.3 acceptance target of 68%.")
        if pd.notna(total.get("informativeness_accuracy")) and float(total.get("informativeness_accuracy", 0)) < 0.91:
            warnings.append("informativeness_accuracy fell below the V2.0.3 floor of 91%.")
    if not previous_sample.empty:
        current = by_sample.set_index("sample")
        previous = previous_sample.drop_duplicates("sample", keep="last").set_index("sample")
        if {"sample_001", "sample_002_consensus"}.issubset(current.index) and {"sample_001", "sample_002_consensus"}.issubset(previous.index):
            sample_001_improved = float(current.loc["sample_001", "error_rate"]) < float(previous.loc["sample_001", "error_rate"])
            holdout_worsened = float(current.loc["sample_002_consensus", "error_rate"]) > float(previous.loc["sample_002_consensus", "error_rate"])
            if sample_001_improved and holdout_worsened:
                errors.append("Benchmark gate failed: sample_001 improved while holdout sample_002_consensus worsened.")
        for sample_name in set(current.index).intersection(previous.index):
            if "informativeness_accuracy" not in current or "informativeness_accuracy" not in previous:
                continue
            current_value = float(current.loc[sample_name, "informativeness_accuracy"])
            previous_value = float(previous.loc[sample_name, "informativeness_accuracy"])
            if previous_value - current_value > 0.02:
                errors.append(f"Benchmark gate failed: informativeness dropped more than 2 p.p. for {sample_name}.")
    if not previous_stance.empty and not by_stance.empty:
        current_stance = by_stance.set_index(["sample", "stance"])
        previous_stance_index = previous_stance.drop_duplicates(["sample", "stance"], keep="last").set_index(["sample", "stance"])
        for key in set(current_stance.index).intersection(previous_stance_index.index):
            current_value = float(current_stance.loc[key, "f1"])
            previous_value = float(previous_stance_index.loc[key, "f1"])
            if previous_value - current_value > 0.03:
                errors.append(f"Benchmark gate failed: stance F1 dropped more than 3 p.p. for {key[0]} / {key[1]}.")
    status = "fail" if errors else "warning" if warnings else "pass"
    return {"status": status, "warnings": warnings, "errors": errors}


def build_baseline_benchmark_report_html(
    by_sample: pd.DataFrame,
    by_topic: pd.DataFrame,
    by_stance: pd.DataFrame,
    gates: dict[str, object],
    run_id: str,
) -> str:
    status = html.escape(str(gates.get("status", "warning")).upper())
    warning_items = "".join(f"<li>{html.escape(str(item))}</li>" for item in gates.get("warnings", []))
    error_items = "".join(f"<li>{html.escape(str(item))}</li>" for item in gates.get("errors", []))
    topic_focus = by_topic.sort_values(["sample", "observations"], ascending=[True, False]).head(40) if not by_topic.empty else by_topic
    stance_focus = by_stance.sort_values(["sample", "stance"]) if not by_stance.empty else by_stance
    return f"""<!doctype html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8">
  <title>COPOM Watch V2 Baseline Benchmark</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #1f2933; }}
    h1, h2 {{ color: #102a43; }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0; font-size: 13px; }}
    th, td {{ border: 1px solid #d9e2ec; padding: 8px; vertical-align: top; }}
    th {{ background: #f0f4f8; text-align: left; }}
    .badge {{ display: inline-block; padding: 4px 8px; border-radius: 4px; background: #e0f2fe; color: #0b4f71; }}
    .warning {{ background: #fff3cd; color: #513c06; }}
    .fail {{ background: #fde2e2; color: #7f1d1d; }}
  </style>
</head>
<body>
  <h1>COPOM Watch V2 Baseline Benchmark</h1>
  <p><span class="badge {'fail' if gates.get('status') == 'fail' else 'warning' if gates.get('status') == 'warning' else ''}">Status: {status}</span></p>
  <p>Run ID: <code>{html.escape(run_id)}</code></p>
  <h2>Resumo Por Amostra</h2>
  {dataframe_to_html_table(by_sample)}
  <h2>Gates</h2>
  <h3>Warnings</h3>
  <ul>{warning_items or '<li>Nenhum warning.</li>'}</ul>
  <h3>Errors</h3>
  <ul>{error_items or '<li>Nenhum error.</li>'}</ul>
  <h2>Topicos</h2>
  {dataframe_to_html_table(topic_focus)}
  <h2>Stance</h2>
  {dataframe_to_html_table(stance_focus)}
</body>
</html>
"""


def dataframe_to_html_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "<p>Nenhum registro disponivel.</p>"
    columns = list(frame.columns)
    header = "".join(f"<th>{html.escape(str(col))}</th>" for col in columns)
    body_rows = []
    for _, row in frame.iterrows():
        cells = "".join(f"<td>{html.escape(str(row.get(col, '')))}</td>" for col in columns)
        body_rows.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"


def format_metric(value: object, pct: bool = False) -> str:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(number):
        return "n/a"
    if pct:
        return f"{100 * float(number):.1f}%"
    return f"{float(number):.0f}"


def build_manual_label_sample(sentence_scores: pd.DataFrame, sample_size: int = 120) -> pd.DataFrame:
    columns = [
        "sentence_id",
        "meeting_id",
        "nro_reuniao",
        "document_type",
        "text",
        "predicted_topic",
        "predicted_stance",
        "predicted_is_informative",
        "label_source",
        "label_status",
        "topic_label",
        "stance_label",
        "is_informative_label",
        "annotator_id",
    ]
    if sentence_scores.empty:
        return pd.DataFrame(columns=columns)
    sample = (
        sentence_scores.sort_values(["primary_topic", "stance", "nro_reuniao", "sentence_id"])
        .groupby(["primary_topic", "stance"], group_keys=False)
        .head(max(1, sample_size // 12))
        .head(sample_size)
    )
    output = sample.rename(
        columns={
            "primary_topic": "predicted_topic",
            "stance": "predicted_stance",
            "is_informative": "predicted_is_informative",
        }
    ).copy()
    output["label_source"] = ""
    output["label_status"] = ""
    output["topic_label"] = ""
    output["stance_label"] = ""
    output["is_informative_label"] = ""
    output["annotator_id"] = ""
    return output[columns]


def write_v2_html_report(
    scores: pd.DataFrame,
    subindices: pd.DataFrame,
    evidence: pd.DataFrame,
    redline: pd.DataFrame,
    v2_settings: dict[str, Any],
    meeting: int | None = None,
) -> Path:
    output_dir = Path(v2_settings["reports"]["output_dir"])
    if not output_dir.is_absolute():
        output_dir = get_paths().database.parents[1] / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    scores = scores.sort_values("data_referencia")
    if meeting is None:
        selected = scores.iloc[-1]
    else:
        selected = scores[scores["nro_reuniao"] == meeting].iloc[0]
    meeting_id = selected["meeting_id"]
    meeting_label = int(selected["nro_reuniao"]) if pd.notna(selected["nro_reuniao"]) else meeting_id
    output_path = output_dir / f"copom_watch_v2_{meeting_label}.html"
    sub = subindices[subindices["meeting_id"] == meeting_id].copy()
    ev = evidence[evidence["meeting_id"] == meeting_id].copy()
    rl = redline[redline["meeting_id"] == meeting_id].copy()
    evidence_columns = [
        col
        for col in ["evidence_type", "citation", "document_type", "primary_topic", "tone_level", "text"]
        if col in ev.columns
    ]
    selected_versions = pd.DataFrame(
        [
            {
                "model_version": selected.get("model_version", ""),
                "prompt_version": selected.get("prompt_version", ""),
                "taxonomy_version": selected.get("taxonomy_version", ""),
                "lexicon_version": selected.get("lexicon_version", ""),
                "rule_engine_version": selected.get("rule_engine_version", rule_engine_version(v2_settings)),
                "calibration_version": selected.get("calibration_version", ""),
                "calibration_status": selected.get("calibration_status", ""),
            }
        ]
    )
    acceptance_path = output_dir / "acceptance_report.html"
    acceptance_html = (
        f"<p><a href='{html.escape(acceptance_path.name)}'>Acceptance report V2.0</a></p>"
        if acceptance_path.exists()
        else "<p class='muted'>Acceptance report V2.0 ainda nao gerado neste diretorio.</p>"
    )
    html_body = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>COPOM Watch V2 Report</title>",
        "<style>body{font-family:Arial,sans-serif;margin:32px;line-height:1.45}table{border-collapse:collapse;width:100%;margin:12px 0}td,th{border:1px solid #ddd;padding:6px;text-align:left}th{background:#f4f4f4}.muted{color:#666}.metric{display:inline-block;margin-right:24px}</style>",
        "</head><body>",
        f"<h1>COPOM Watch V2 - Reuniao {html.escape(str(meeting_label))}</h1>",
        "<p class='muted'>Relatorio automatico. Leitura descritiva, nao causal.</p>",
        "<h2>Headline</h2>",
        "<div>",
        f"<span class='metric'><b>Tone raw:</b> {fmt_number(selected.get('tone_raw'))}</span>",
        f"<span class='metric'><b>Index V2:</b> {fmt_number(selected.get('copom_tone_index_v2'))}</span>",
        f"<span class='metric'><b>Classificacao:</b> {html.escape(str(selected.get('classification_v2', 'unavailable')))}</span>",
        f"<span class='metric'><b>Surpresa textual naive:</b> {fmt_number(selected.get('communication_surprise_naive'))}</span>",
        "</div>",
        "<h2>Versoes e aceite</h2>",
        dataframe_to_html(selected_versions),
        acceptance_html,
        "<h2>Subindices</h2>",
        dataframe_to_html(sub[["subindex", "label", "tone_raw", "sentence_count"]] if not sub.empty else pd.DataFrame()),
        "<h2>Frases-chave</h2>",
        dataframe_to_html(ev[evidence_columns] if not ev.empty and evidence_columns else pd.DataFrame()),
        "<h2>Mudanca textual</h2>",
        dataframe_to_html(rl[["document_type", "change_type", "similarity", "tone_delta", "current_text", "previous_text"]].head(30) if not rl.empty else pd.DataFrame()),
        "</body></html>",
    ]
    output_path.write_text("\n".join(html_body), encoding="utf-8")
    return output_path


def build_event_calendar(documents: pd.DataFrame) -> pd.DataFrame:
    if documents.empty:
        return pd.DataFrame(
            columns=[
                "meeting_id",
                "nro_reuniao",
                "document_type",
                "release_date",
                "release_timestamp",
                "known_at_timestamp",
                "source_url",
                "market_close_convention",
            ]
        )
    frame = ensure_v2_document_source_urls(documents)
    frame["release_date"] = pd.to_datetime(frame["publication_date"], errors="coerce").dt.normalize()
    frame["release_timestamp"] = frame.apply(_known_at_timestamp, axis=1)
    frame["known_at_timestamp"] = frame["release_timestamp"]
    frame["market_close_convention"] = np.where(frame["document_type"] == "comunicado", "after_close", "before_or_during_session")
    return frame[
        [
            "meeting_id",
            "nro_reuniao",
            "document_type",
            "release_date",
            "release_timestamp",
            "known_at_timestamp",
            "source_url",
            "market_close_convention",
        ]
    ].sort_values(["nro_reuniao", "document_type"])


def build_v2_version_tables(
    topics: dict[str, Any],
    lexicon: dict[str, Any],
    v2_settings: dict[str, Any],
    run_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    taxonomy_payload = {
        "topics": topics,
        "subindices": v2_settings["subindices"],
        "reaction_function": v2_settings["reaction_function"],
        "rule_engine_version": rule_engine_version(v2_settings),
    }
    taxonomy_json = stable_json(taxonomy_payload)
    lexicon_json = stable_json(lexicon)
    taxonomy = pd.DataFrame(
        [
            {
                "taxonomy_version": v2_settings["project"]["taxonomy_version"],
                "content_hash": stable_text_hash(taxonomy_json),
                "payload_json": taxonomy_json,
                "run_id": run_id,
                "created_at": utc_now_naive(),
            }
        ]
    )
    lexicon_versions = pd.DataFrame(
        [
            {
                "lexicon_version": v2_settings["project"]["lexicon_version"],
                "content_hash": stable_text_hash(lexicon_json),
                "payload_json": lexicon_json,
                "run_id": run_id,
                "created_at": utc_now_naive(),
            }
        ]
    )
    return taxonomy, lexicon_versions


def append_run(database: Path, row: dict[str, Any]) -> pd.DataFrame:
    existing = read_optional_table(database, "v2_runs", pd.DataFrame())
    frame = pd.concat([existing, pd.DataFrame([row])], ignore_index=True) if not existing.empty else pd.DataFrame([row])
    return frame.drop_duplicates("run_id", keep="last")


def run_row(run_id: str, stage: str, status: str, params: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "stage": stage,
        "status": status,
        "params_json": json.dumps(params, ensure_ascii=False, sort_keys=True),
        "created_at": utc_now_naive(),
    }


def read_optional_table(database: Path, table: str, default: pd.DataFrame) -> pd.DataFrame:
    if not database.exists():
        return default.copy()
    with duckdb.connect(str(database), read_only=True) as con:
        exists = con.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
            [table],
        ).fetchone()[0]
        if not exists:
            return default.copy()
        return con.execute(f"SELECT * FROM {table}").df()


def empty_v2_sentence_scores() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "sentence_id",
            "document_id",
            "meeting_id",
            "primary_topic",
            "topics",
            "stance",
            "stance_score",
            "confidence",
            "is_informative",
            "information_weight",
            "tone_level",
            "taxonomy_boundary_flag",
            "rule_engine_version",
        ]
    )


def empty_v2_labels() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "sentence_id",
            "label_source",
            "label_status",
            "topic_label",
            "stance_label",
            "is_informative_label",
            "annotator_id",
            "accepted_at",
        ]
    )


def normalize_v2_labels(frame: pd.DataFrame, label_source: str = "human") -> pd.DataFrame:
    if frame.empty:
        return empty_v2_labels()
    renamed = frame.copy()
    aliases = {
        "human_topic": "topic_label",
        "human_stance": "stance_label",
        "human_is_informative": "is_informative_label",
        "reviewer_id": "annotator_id",
    }
    for source, target in aliases.items():
        if source in renamed and target not in renamed:
            renamed[target] = renamed[source]
        elif source in renamed and target in renamed:
            missing = renamed[target].isna() | (renamed[target].astype(str).str.strip() == "")
            renamed.loc[missing, target] = renamed.loc[missing, source]
    for column in empty_v2_labels().columns:
        if column not in renamed:
            renamed[column] = ""
    source_values = renamed["label_source"].where(renamed["label_source"].notna(), "").astype(str).str.strip()
    renamed["label_source"] = source_values.mask(source_values.eq(""), label_source).astype("object")
    status_values = renamed["label_status"].where(renamed["label_status"].notna(), "").astype(str).str.strip()
    explicit_status = status_values.mask(status_values.eq(""), np.nan).astype("object")
    accepted_flag = renamed["accepted"].map(parse_bool_label) if "accepted" in renamed else pd.Series([False] * len(renamed), index=renamed.index)
    any_label = (
        renamed["topic_label"].fillna("").astype(str).str.strip().ne("")
        | renamed["stance_label"].fillna("").astype(str).str.strip().ne("")
        | renamed["is_informative_label"].fillna("").astype(str).str.strip().ne("")
    )
    renamed["label_status"] = explicit_status.astype("object")
    renamed.loc[renamed["label_status"].isna() & accepted_flag, "label_status"] = "accepted"
    renamed.loc[renamed["label_status"].isna() & ~accepted_flag & any_label, "label_status"] = "human_reviewed"
    renamed["label_status"] = renamed["label_status"].fillna("pending_review")
    accepted_status = renamed["label_status"].isin(["accepted", "human_reviewed"])
    fallback_pairs = {
        "topic_label": ["baseline_topic", "predicted_topic"],
        "stance_label": ["baseline_stance", "predicted_stance"],
        "is_informative_label": ["baseline_is_informative", "predicted_is_informative"],
    }
    for label_col, fallback_cols in fallback_pairs.items():
        missing = renamed[label_col].isna() | (renamed[label_col].astype(str).str.strip() == "")
        for fallback_col in fallback_cols:
            if fallback_col not in renamed:
                continue
            fill_mask = missing & accepted_status
            renamed.loc[fill_mask, label_col] = renamed.loc[fill_mask, fallback_col]
            missing = renamed[label_col].isna() | (renamed[label_col].astype(str).str.strip() == "")
    renamed["is_informative_label"] = renamed["is_informative_label"].map(parse_bool_label)
    renamed["accepted_at"] = pd.to_datetime(renamed["accepted_at"], errors="coerce")
    missing_accepted = renamed["accepted_at"].isna() & accepted_status
    renamed.loc[missing_accepted, "accepted_at"] = utc_now_naive()
    return renamed[empty_v2_labels().columns].dropna(subset=["sentence_id"])


def parse_bool_label(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"true", "1", "yes", "sim", "y"}


def accepted_v2_labels(labels: pd.DataFrame) -> pd.DataFrame:
    if labels.empty or "label_status" not in labels:
        return empty_v2_labels()
    accepted = labels[labels["label_status"].isin(["accepted", "human_reviewed"])].copy()
    if "label_source" in accepted:
        source = accepted["label_source"].fillna("").astype(str).str.strip().str.lower()
        accepted = accepted[~source.isin({"llm_bootstrap", "bootstrap", "synthetic", "baseline", "model", "prediction"})]
    if accepted.empty:
        return empty_v2_labels()
    valid = (
        accepted["topic_label"].fillna("").astype(str).str.strip().ne("")
        & accepted["stance_label"].fillna("").astype(str).str.strip().ne("")
    )
    return accepted[valid].reset_index(drop=True)


def consensus_v2_labels(labels: pd.DataFrame) -> pd.DataFrame:
    if labels.empty:
        return pd.DataFrame(columns=list(empty_v2_labels().columns) + ["reviewer_count", "label_conflict"])
    accepted = accepted_v2_labels(labels)
    if accepted.empty:
        return pd.DataFrame(columns=list(empty_v2_labels().columns) + ["reviewer_count", "label_conflict"])
    rows: list[dict[str, object]] = []
    for sentence_id, group in accepted.groupby("sentence_id", sort=True):
        row = group.sort_values(["accepted_at", "annotator_id"], na_position="last").iloc[-1].to_dict()
        topic = majority_label(group["topic_label"])
        stance = majority_label(group["stance_label"])
        informative = bool(majority_label(group["is_informative_label"].map(bool)))
        row["sentence_id"] = sentence_id
        row["topic_label"] = topic
        row["stance_label"] = stance
        row["is_informative_label"] = informative
        row["label_source"] = "|".join(sorted(set(group["label_source"].fillna("").astype(str))))
        row["annotator_id"] = "|".join(sorted(value for value in set(group["annotator_id"].fillna("").astype(str)) if value))
        row["reviewer_count"] = int(group["annotator_id"].fillna("").astype(str).replace("", np.nan).nunique())
        row["label_conflict"] = bool(
            group["topic_label"].fillna("").astype(str).nunique() > 1
            or group["stance_label"].fillna("").astype(str).nunique() > 1
            or group["is_informative_label"].map(bool).nunique() > 1
        )
        rows.append(row)
    return pd.DataFrame(rows).reset_index(drop=True)


def majority_label(values: pd.Series) -> object:
    clean = values.dropna()
    if clean.empty:
        return ""
    counts = clean.value_counts()
    max_count = counts.max()
    winners = sorted(str(value) for value, count in counts.items() if count == max_count)
    if clean.dtype == bool:
        return winners[-1] == "True"
    return winners[0] if winners else ""


def accuracy_metric(frame: pd.DataFrame, truth: str, pred: str) -> float:
    if frame.empty or truth not in frame or pred not in frame:
        return float("nan")
    return float((frame[truth].astype(str) == frame[pred].astype(str)).mean())


def confusion_matrix_records(frame: pd.DataFrame, truth: str, pred: str) -> list[dict[str, object]]:
    if frame.empty or truth not in frame or pred not in frame:
        return []
    return (
        frame.groupby([truth, pred])
        .size()
        .reset_index(name="count")
        .rename(columns={truth: "actual", pred: "predicted"})
        .to_dict("records")
    )


def macro_f1_metric(y_true: pd.Series, y_pred: pd.Series) -> float:
    labels = sorted(set(y_true.dropna().astype(str)) | set(y_pred.dropna().astype(str)))
    if not labels:
        return float("nan")
    true = y_true.astype(str)
    pred = y_pred.astype(str)
    scores: list[float] = []
    for label in labels:
        tp = int(((true == label) & (pred == label)).sum())
        fp = int(((true != label) & (pred == label)).sum())
        fn = int(((true == label) & (pred != label)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return float(np.mean(scores))


def human_stance_agreement(labels: pd.DataFrame) -> float | None:
    if labels.empty or "annotator_id" not in labels:
        return None
    duplicated = labels[labels.duplicated("sentence_id", keep=False)].copy()
    if duplicated.empty:
        return None
    agreements: list[float] = []
    for _, group in duplicated.groupby("sentence_id"):
        if group["annotator_id"].nunique() < 2:
            continue
        agreements.append(float(group["stance_label"].astype(str).nunique() == 1))
    return float(np.mean(agreements)) if agreements else None


def normalized_rule_text(text: str) -> str:
    return strip_accents(normalize_whitespace(text).lower())


def contains_rule_term(normalized_text: str, terms: list[str]) -> bool:
    return any(term in normalized_text for term in terms)


def rule_engine_version(v2_settings: dict[str, Any]) -> str:
    return str(v2_settings.get("project", {}).get("rule_engine_version", DEFAULT_RULE_ENGINE_VERSION))


def topic_candidate_detection(text: str, topics: dict[str, Any]) -> list[str]:
    matched_topics = matched_topic_names(text, topics)
    return apply_topic_overrides(text, matched_topics, topics)


def topic_priority_resolution(text: str, matched_topics: list[str], topics: dict[str, Any]) -> str:
    current_topic = primary_topic_name(text, topics, matched_topics)
    return primary_topic_override(text, current_topic, matched_topics, topics)


def stance_direction_detection(
    text: str,
    primary_topic: str | None = None,
    matched_topics: list[str] | None = None,
) -> tuple[str, float, str] | None:
    return directional_stance_override(text, primary_topic=primary_topic, matched_topics=matched_topics)


def negation_and_reversal_guards(text: str, stance: str, primary_topic: str | None = None) -> tuple[str, float, str] | None:
    lowered = normalized_rule_text(text)
    if contains_rule_term(lowered, ["aumenta os riscos", "aumentar os riscos", "aumentam os riscos", "risco para o cenario benigno"]):
        return ("hawkish", 1.0, "rule:benign_scenario_risk_reversal")
    if contains_rule_term(
        lowered,
        [
            "nao permite uma reducao dos juros",
            "nao permite reducao dos juros",
            "sem espaco para reducao",
            "menor espaco para cortes",
            "menor espaco para reducao",
        ],
    ):
        return ("hawkish", 1.0, "rule:easing_space_restricted")
    if stance == "dovish" and primary_topic in {"risk_balance", "inflation_current", "inflation_expectations"}:
        if contains_rule_term(lowered, ["apesar de benigno", "cenario benigno"]) and contains_rule_term(
            lowered,
            ["risco", "riscos", "pressao", "pressoes", "altista", "aumento"],
        ):
            return ("hawkish", 1.0, "rule:benign_word_guard")
    return None


def ambiguity_flagging(text: str, primary_topic: str, matched_topics: list[str]) -> str:
    lowered = normalized_rule_text(text)
    topic_set = set(matched_topics)
    if (
        {"policy_decision", "forward_guidance"}.issubset(topic_set)
        and is_actual_policy_decision_text(lowered)
        and is_forward_guidance_text(lowered)
    ):
        return "policy_decision_vs_forward_guidance"
    if {"inflation_current", "inflation_expectations"}.issubset(topic_set):
        return "current_inflation_vs_expectations"
    if {"fiscal_risk", "external_environment"}.issubset(topic_set):
        return "fiscal_risk_vs_external_environment"
    if {"activity_growth", "labor_market"}.issubset(topic_set):
        return "activity_growth_vs_labor_market"
    if {"external_environment", "uncertainty"}.issubset(topic_set):
        return "external_environment_vs_uncertainty"
    if primary_topic == "uncertainty" and contains_rule_term(lowered, ["cenario externo", "ambiente externo", "mercados internacionais"]):
        return "external_environment_vs_uncertainty"
    return ""


def is_domestic_activity_text(normalized_text: str) -> bool:
    return contains_rule_term(
        normalized_text,
        [
            "indicadores setoriais",
            "atividade fabril",
            "producao industrial",
            "producao da industria",
            "producao fisica",
            "producao de insumos",
            "industria de transformacao",
            "industria brasileira",
            "industria nacional",
            "atividade industrial",
            "nivel de atividade",
            "retomada da atividade",
            "ritmo da retomada",
            "volume de vendas",
            "vendas internas",
            "vendas do comercio",
            "vendas de autoveiculos",
            "licenciamento de autoveiculos",
            "fenabrave",
            "fecomercio",
            "bens de capital",
            "bens duraveis",
            "demanda domestica",
            "comercio varejista",
            "confianca do consumidor",
            "confianca da industria",
            "confianca dos empresarios",
            "indice de confianca",
            "indice de condicoes economicas",
            "expectativas do consumidor",
            "icc",
            "cni",
            "utilizacao da capacidade",
            "capacidade instalada",
            "faturamento real",
        ],
    )


def is_labor_market_text(normalized_text: str) -> bool:
    return contains_rule_term(
        normalized_text,
        [
            "mercado de trabalho",
            "massa salarial",
            "rendimento medio",
            "salario",
            "salarios",
            "emprego",
            "desemprego",
            "pessoal ocupado",
            "populacao ocupada",
            "postos de trabalho",
            "taxa de ocupacao",
        ],
    )


def is_credit_conditions_text(normalized_text: str) -> bool:
    return contains_rule_term(
        normalized_text,
        [
            "credito",
            "inadimplencia",
            "condicoes de credito",
            "compras a credito",
            "concessoes",
            "consultas para compras",
            "sistema usecheque",
            "canais de credito",
        ],
    )


def is_inflation_expectations_text(normalized_text: str) -> bool:
    return contains_rule_term(
        normalized_text,
        [
            "expectativas de inflacao",
            "expectativas inflacionarias",
            "mediana das expectativas",
            "expectativas coletadas",
            "expectativas dos analistas",
            "expectativas de mercado",
            "focus",
            "gerin",
            "meta de inflacao",
            "projecao de inflacao",
            "projecoes de inflacao",
            "cenario de referencia",
            "cenario de mercado",
        ],
    ) or ("expectativas" in normalized_text and contains_rule_term(normalized_text, ["ipca", "inflacao", "meta"]))


def is_current_inflation_text(normalized_text: str) -> bool:
    return contains_rule_term(
        normalized_text,
        [
            "incc",
            "indice nacional da construcao civil",
            "ipa",
            "igp",
            "ipca",
            "ipca-15",
            "ipc-br",
            "nucleo",
            "medias aparadas",
            "indice de precos",
            "indices de precos",
            "precos livres",
            "precos administrados",
            "precos no atacado",
            "nucleos",
            "servicos",
            "alimentos",
            "bens industriais",
            "bens finais",
            "materias-primas",
            "cesta basica",
            "inflacao corrente",
            "inflacao acumulada",
        ],
    )


def is_external_environment_text(normalized_text: str) -> bool:
    return re.search(r"\bfed\b", normalized_text) is not None or contains_rule_term(
        normalized_text,
        [
            "estados unidos",
            "eua",
            "norte-americano",
            "norte-americana",
            "alemanha",
            "economia europeia",
            "treasuries",
            "federal reserve",
            "libor",
            "overnight index swap",
            "ois",
            "europa",
            "zona do euro",
            "china",
            "japao",
            "economia mundial",
            "economia global",
            "cenario externo",
            "ambiente externo",
            "mercados internacionais",
            "liquidez global",
            "aversao global ao risco",
            "us$",
        ],
    )


def is_strong_external_macro_context(normalized_text: str) -> bool:
    return contains_rule_term(
        normalized_text,
        [
            "inflacao norte-americana",
            "produtor norte-americana",
            "produtor norte-americano",
            "economias maduras",
            "economias emergentes",
            "economia mundial",
            "economia global",
            "cenario macroeconomico global",
            "crescimento da economia mundial",
            "economia chinesa",
            "desaceleracao na china",
            "condicoes economicas nos eua",
            "eua",
            "estados unidos",
            "alemanha",
            "economia europeia",
            "europa",
            "zona do euro",
            "china",
            "japao",
            "fed",
            "federal reserve",
            "mercados financeiros internacionais",
        ],
    )


def is_domestic_brazil_context(normalized_text: str) -> bool:
    return contains_rule_term(
        normalized_text,
        [
            "brasil",
            "brasileira",
            "brasileiro",
            "domestica",
            "domestico",
            "ibge",
            "cni",
            "fecomercio",
            "fenabrave",
            "abpo",
            "fiesp",
            "banco central",
            "copom",
            "focus",
            "gerin",
        ],
    )


def is_fiscal_risk_text(normalized_text: str) -> bool:
    return contains_rule_term(
        normalized_text,
        [
            "politica fiscal",
            "superavit primario",
            "resultado primario",
            "divida liquida",
            "divida publica",
            "financas publicas",
            "risco fiscal",
            "arcabouco fiscal",
            "sustentabilidade da divida",
            "deficit fiscal",
            "premio fiscal",
        ],
    )


def is_risk_balance_text(normalized_text: str) -> bool:
    return contains_rule_term(
        normalized_text,
        [
            "balanco de riscos",
            "riscos para inflacao",
            "riscos para a inflacao",
            "riscos para o cenario",
            "riscos para a consolidacao",
            "riscos para a concretizacao",
            "riscos de alta",
            "riscos de baixa",
            "cenario inflacionario benigno",
            "risco de repasse",
            "probabilidade de que desenvolvimentos inflacionarios",
        ],
    )


def is_uncertainty_text(normalized_text: str) -> bool:
    return contains_rule_term(
        normalized_text,
        [
            "incerteza",
            "volatilidade",
            "aversao ao risco",
            "duvidas quanto",
            "reticencia dos participantes",
            "grau de preocupacao",
            "sujeitas a riscos",
        ],
    )


def is_fx_commodities_text(normalized_text: str) -> bool:
    return contains_rule_term(
        normalized_text,
        ["petroleo", "brent", "commodities", "taxa de cambio", "cambio", "dolar", "repasse cambial"],
    )


def is_descriptive_macro_data_text(normalized_text: str) -> bool:
    return (
        is_domestic_activity_text(normalized_text)
        or is_labor_market_text(normalized_text)
        or is_current_inflation_text(normalized_text)
        or is_fx_commodities_text(normalized_text)
    )


def is_actual_policy_decision_text(normalized_text: str) -> bool:
    decision_verbs = [
        "decidiu",
        "decidiram",
        "votaram",
        "votou",
        "deliberou",
        "fixou",
        "estabeleceu",
        "a diretoria aumentou",
        "a diretoria reduziu",
        "aumentou a aliquota",
        "reduziu a aliquota",
        "elevou a aliquota",
        "levou o copom",
        "votos discordantes foram",
    ]
    policy_target_terms = [
        "taxa selic",
        "taxa basica de juros",
        "taxa de juros basica",
        "reducao da taxa",
        "meta para a taxa selic",
        "selic para",
        "selic em",
        "sem vies",
        "com vies",
        "compulsorio",
        "aliquota do compulsorio",
        "recolhimento compulsorio",
        "depositos a vista",
    ]
    movement_terms = [
        "manter",
        "mantida",
        "mantido",
        "reduzir",
        "reducao",
        "elevar",
        "aumentar",
        "aumento de",
        "corte",
        "p.p.",
        "pontos-base",
    ]
    return contains_rule_term(normalized_text, decision_verbs) and (
        contains_rule_term(normalized_text, policy_target_terms) or contains_rule_term(normalized_text, movement_terms)
    )


def is_forward_guidance_text(normalized_text: str) -> bool:
    strong_guidance_terms = [
        "proximos passos",
        "proxima reuniao",
        "ira acompanhar",
        "ira monitorar",
        "monitorar atentamente",
        "acompanhar atentamente",
        "definir os proximos passos",
        "estrategia de politica monetaria",
        "trajetoria prospectiva",
        "estimulo monetario adicional",
        "vies de baixa",
        "decisao contribuira",
        "aumentar a magnitude do ajuste a ser implementado",
        "processo de ajuste",
        "acao preventiva",
    ]
    weak_guidance_terms = ["continuidade", "trajetoria", "perspectiva", "ciclo", "magnitude", "processo decisorio"]
    if contains_rule_term(normalized_text, strong_guidance_terms):
        return True
    return contains_rule_term(normalized_text, weak_guidance_terms) and not is_descriptive_macro_data_text(normalized_text)


def is_institutional_or_header_sentence(text: str) -> bool:
    lowered = normalized_rule_text(text)
    if not lowered:
        return True
    institutional_markers = [
        "sumario",
        "presentes:",
        "membros do copom",
        "diretores do banco central",
        "chefes de departamento",
        "demais participantes",
        "local:",
        "horario de inicio",
        "horario de termino",
        "data:",
        "notas de rodape",
        "votaram por essa decisao",
        "votaram por esta decisao",
    ]
    marker_count = sum(1 for marker in institutional_markers if marker in lowered)
    if marker_count >= 2:
        return True
    if lowered.startswith(
        (
            "sumario",
            "presentes:",
            "membros do copom",
            "chefes de departamento",
            "demais participantes",
            "local:",
            "horario de inicio",
            "notas de rodape",
            "votaram por essa decisao",
            "votaram por esta decisao",
        )
    ):
        return True
    section_terms = [
        "atividade economica",
        "mercado de trabalho",
        "credito",
        "inadimplencia",
        "expectativas e sondagens",
        "inflacao",
        "ambiente externo",
        "cenario externo",
        "mercado financeiro",
        "politica monetaria",
        "decisao de politica monetaria",
        "votos",
    ]
    if len(lowered) <= 90 and contains_rule_term(lowered, section_terms) and re.search(r"\b\d+\.?$", lowered):
        return True
    model_projection_markers = [
        "modelo var",
        "modelo de determinacao endogena",
        "componentes sazonais",
        "projecao para o spread",
        "indicador coincidente",
    ]
    if sum(1 for marker in model_projection_markers if marker in lowered) >= 2:
        return True
    if lowered.startswith(("esse modelo considera", "para os demais itens, as projecoes permaneceram")):
        return True
    if "constitui importante indicador coincidente" in lowered:
        return True
    return False


def apply_topic_overrides(text: str, matched_topics: list[str], topics: dict[str, Any]) -> list[str]:
    lowered = normalized_rule_text(text)
    adjusted = list(dict.fromkeys(matched_topics))
    if "policy_decision" in topics and is_actual_policy_decision_text(lowered) and "policy_decision" not in adjusted:
        adjusted.append("policy_decision")
    if "forward_guidance" in topics and is_forward_guidance_text(lowered) and "forward_guidance" not in adjusted:
        adjusted.append("forward_guidance")
    if "activity_growth" in topics and is_domestic_activity_text(lowered) and "activity_growth" not in adjusted:
        adjusted.append("activity_growth")
    if "labor_market" in topics and is_labor_market_text(lowered) and "labor_market" not in adjusted:
        adjusted.append("labor_market")
    if "credit_conditions" in topics and is_credit_conditions_text(lowered) and "credit_conditions" not in adjusted:
        adjusted.append("credit_conditions")
    if "inflation_expectations" in topics and is_inflation_expectations_text(lowered) and "inflation_expectations" not in adjusted:
        adjusted.append("inflation_expectations")
    if "inflation_current" in topics and is_current_inflation_text(lowered) and "inflation_current" not in adjusted:
        adjusted.append("inflation_current")
    if "external_environment" in topics and is_external_environment_text(lowered) and "external_environment" not in adjusted:
        adjusted.append("external_environment")
    if "fiscal_risk" in topics and is_fiscal_risk_text(lowered) and "fiscal_risk" not in adjusted:
        adjusted.append("fiscal_risk")
    if "risk_balance" in topics and is_risk_balance_text(lowered) and "risk_balance" not in adjusted:
        adjusted.append("risk_balance")
    if "uncertainty" in topics and is_uncertainty_text(lowered) and "uncertainty" not in adjusted:
        adjusted.append("uncertainty")
    if "fx_commodities" in topics and is_fx_commodities_text(lowered) and "fx_commodities" not in adjusted:
        adjusted.append("fx_commodities")
    for topic, terms in {
        "inflation_expectations": [
            "expectativas de inflacao",
            "mediana das expectativas",
            "expectativas coletadas",
            "gerin",
            "meta de inflacao",
        ],
        "inflation_current": [
            "incc",
            "ipa",
            "igp",
            "ipca",
            "indice de precos",
            "indices de precos",
            "precos",
            "pressao inflacionaria",
            "pressoes inflacionarias",
            "pressao de precos",
            "pressoes de precos",
            "cesta basica",
            "precos livres",
            "precos no atacado",
            "bens comercializaveis",
            "nucleos",
            "servicos",
            "administrados",
            "bens industriais",
            "alimentos",
        ],
        "external_environment": [
            "estados unidos",
            "eua",
            "fed",
            "federal reserve",
            "europa",
            "zona do euro",
            "alemanha",
            "japao",
            "china",
            "setor externo",
            "comercio exterior",
            "balanca comercial",
            "superavit comercial",
            "superavits comerciais",
            "exportacoes",
            "importacoes",
            "bancos centrais",
            "economias emergentes",
            "economia mundial",
            "economia global",
            "cenario externo",
            "ambiente externo",
        ],
        "activity_growth": [
            "atividade economica",
            "atividade fabril",
            "indicadores setoriais",
            "crescimento",
            "volume de vendas",
            "comercio varejista",
            "faturamento",
            "producao industrial",
            "capacidade instalada",
            "uci",
            "autoveiculos",
            "confianca",
            "cni",
        ],
        "labor_market": [
            "emprego",
            "mercado de trabalho",
            "vagas",
            "postos de trabalho",
            "mte",
            "desemprego",
        ],
        "policy_decision": [
            "taxa selic",
            "taxa basica de juros",
            "taxa de juros basica",
            "vies de baixa",
            "votaram por uma reducao",
            "estímulo monetário adicional",
            "estimulo monetario adicional",
            "taxa de juros neutra",
        ],
        "forward_guidance": [
            "processo de ajuste",
            "acao preventiva",
            "sinalizacao",
            "proximos trimestres",
            "perspectiva",
            "ciclo expansionista",
            "processo decisorio",
        ],
        "fx_commodities": [
            "petroleo",
            "brent",
            "commodities",
            "cambio",
            "taxa de cambio",
            "dolar",
            "real",
            "repasse cambial",
        ],
    }.items():
        if topic in topics and contains_rule_term(lowered, terms) and topic not in adjusted:
            adjusted.append(topic)
    if not adjusted:
        return ["institutional"]
    if "institutional" in adjusted and len(adjusted) > 1:
        adjusted = [topic for topic in adjusted if topic != "institutional"]
    return adjusted


def primary_topic_override(text: str, current_topic: str, matched_topics: list[str], topics: dict[str, Any]) -> str:
    lowered = normalized_rule_text(text)
    if "policy_decision" in topics and "policy_decision" in matched_topics and is_actual_policy_decision_text(lowered):
        return "policy_decision"
    if "inflation_expectations" in topics and "inflation_expectations" in matched_topics and is_inflation_expectations_text(lowered):
        return "inflation_expectations"
    if "risk_balance" in topics and "risk_balance" in matched_topics and is_risk_balance_text(lowered):
        return "risk_balance"
    if (
        "external_environment" in topics
        and "external_environment" in matched_topics
        and is_external_environment_text(lowered)
        and is_strong_external_macro_context(lowered)
        and not is_domestic_brazil_context(lowered)
    ):
        return "external_environment"
    if "labor_market" in topics and "labor_market" in matched_topics and is_labor_market_text(lowered):
        return "labor_market"
    if "activity_growth" in topics and "activity_growth" in matched_topics and is_domestic_activity_text(lowered):
        return "activity_growth"
    if (
        "forward_guidance" in topics
        and "forward_guidance" in matched_topics
        and contains_rule_term(
            lowered,
            [
                "objetivo promover a convergencia",
                "promover a convergencia da inflacao",
                "estrategia visa assegurar",
                "assegurar que a convergencia",
            ],
        )
    ):
        return "forward_guidance"
    if (
        "external_environment" in topics
        and "external_environment" in matched_topics
        and is_external_environment_text(lowered)
        and (is_strong_external_macro_context(lowered) or not is_domestic_brazil_context(lowered))
    ):
        return "external_environment"
    if "inflation_current" in topics and "inflation_current" in matched_topics and is_current_inflation_text(lowered):
        return "inflation_current"
    if "fx_commodities" in topics and "fx_commodities" in matched_topics and is_fx_commodities_text(lowered):
        return "fx_commodities"
    if "external_environment" in topics and "external_environment" in matched_topics and is_external_environment_text(lowered):
        return "external_environment"
    if "credit_conditions" in topics and "credit_conditions" in matched_topics and is_credit_conditions_text(lowered):
        return "credit_conditions"
    if "fiscal_risk" in topics and "fiscal_risk" in matched_topics and is_fiscal_risk_text(lowered):
        return "fiscal_risk"
    if "uncertainty" in topics and "uncertainty" in matched_topics and is_uncertainty_text(lowered):
        return "uncertainty"
    if (
        "forward_guidance" in topics
        and "forward_guidance" in matched_topics
        and is_forward_guidance_text(lowered)
        and not is_descriptive_macro_data_text(lowered)
    ):
        return "forward_guidance"
    priority_rules = [
        (
            "inflation_expectations",
            [
                "expectativas de inflacao",
                "mediana das expectativas",
                "expectativas coletadas",
                "expectativas dos analistas",
                "expectativas de mercado",
                "gerin",
                "focus",
                "meta de inflacao",
                "projecao de inflacao",
                "projecoes de inflacao",
                "cenario de referencia",
                "cenario de mercado",
            ],
        ),
        (
            "labor_market",
            [
                "emprego",
                "mercado de trabalho",
                "vagas",
                "postos de trabalho",
                "mte",
                "desemprego",
                "massa salarial",
                "rendimento medio",
                "salarios",
                "pessoal ocupado",
            ],
        ),
        (
            "activity_growth",
            [
                "atividade economica",
                "atividade fabril",
                "indicadores setoriais",
                "crescimento",
                "volume de vendas",
                "comercio varejista",
                "faturamento",
                "producao industrial",
                "producao da industria",
                "industria",
                "vendas internas",
                "vendas de autoveiculos",
                "licenciamento de autoveiculos",
                "bens de capital",
                "bens duraveis",
                "demanda domestica",
                "capacidade instalada",
                "uci",
                "autoveiculos",
                "confianca",
                "cni",
                "fenabrave",
                "fecomercio",
                "icc",
            ],
        ),
        (
            "external_environment",
            [
                "estados unidos",
                "eua",
                "fed",
                "federal reserve",
                "europa",
                "zona do euro",
                "alemanha",
                "japao",
                "china",
                "setor externo",
                "comercio exterior",
                "balanca comercial",
                "superavit comercial",
                "superavits comerciais",
                "bancos centrais",
                "economias emergentes",
                "economia mundial",
                "economia global",
                "cenario externo",
                "ambiente externo",
            ],
        ),
        (
            "forward_guidance",
            [
                "processo de ajuste",
                "acao preventiva",
                "sinalizacao",
                "proximos trimestres",
                "perspectiva",
                "processo decisorio",
            ],
        ),
        (
            "fx_commodities",
            [
                "petroleo",
                "brent",
                "commodities",
                "taxa de cambio",
                "cambio",
                "dolar",
                "repasse cambial",
            ],
        ),
        (
            "inflation_current",
            [
                "incc",
                "ipa",
                "igp",
                "ipca",
                "indice de precos",
                "indices de precos",
                "precos",
                "pressao inflacionaria",
                "pressoes inflacionarias",
                "pressao de precos",
                "pressoes de precos",
                "cesta basica",
                "precos livres",
                "precos no atacado",
                "bens comercializaveis",
                "nucleos",
                "servicos",
                "administrados",
                "bens industriais",
                "alimentos",
            ],
        ),
    ]
    for topic, terms in priority_rules:
        if topic == "forward_guidance" and is_descriptive_macro_data_text(lowered):
            continue
        if topic in topics and topic in matched_topics and contains_rule_term(lowered, terms):
            return topic
    return current_topic


def directional_stance_override(
    text: str,
    primary_topic: str | None = None,
    matched_topics: list[str] | None = None,
) -> tuple[str, float, str] | None:
    lowered = normalized_rule_text(text)
    upside_terms = [
        "alta",
        "altista",
        "elevacao",
        "aumento",
        "aceleracao",
        "pression",
        "pressionada",
        "pressionado",
        "persistencia",
        "resiliencia",
        "menos benigno",
    ]
    downside_terms = [
        "queda",
        "reducao",
        "reduzir",
        "reduziram",
        "recuo",
        "recuaram",
        "declinio",
        "desaceleracao",
        "contracao",
        "arrefecimento",
        "apreciacao",
        "valorizacao",
        "diminuindo",
        "acomodacao",
        "minimos historicos",
        "inferior",
        "patamar inferior",
    ]
    inflation_terms = [
        "inflacao",
        "inflacionaria",
        "inflacionarias",
        "precos",
        "ipca",
        "igp",
        "ipa",
        "incc",
        "nucleo",
        "bens comercializaveis",
        "nucleos",
        "servicos",
    ]
    activity_terms = [
        "atividade",
        "crescimento",
        "vendas",
        "comercio varejista",
        "producao",
        "faturamento",
        "confianca",
        "industria",
        "demanda",
    ]
    labor_terms = ["emprego", "vagas", "postos de trabalho", "mercado de trabalho", "massa salarial", "rendimento medio", "salarios"]
    expectation_context = is_inflation_expectations_text(lowered)
    risk_context = is_risk_balance_text(lowered) or contains_rule_term(
        lowered,
        ["balanco de riscos", "riscos de alta", "risco altista", "risco para o cenario benigno", "riscos nesse cenario"],
    )
    if contains_rule_term(lowered, ["nao indica tendencia de queda", "nao indicam tendencia de queda"]):
        return ("neutral", 0.0, "rule:explicit_activity_trend_neutralizer")
    if contains_rule_term(lowered, ["interpretada com cautela", "interpretado com cautela"]) and contains_rule_term(
        lowered,
        ["continuam em expansao", "continua em expansao", "aumento extraordinario"],
    ):
        return ("neutral", 0.0, "rule:mixed_activity_data_neutralizer")
    if contains_rule_term(lowered, ["tendencias contracionistas prevalecem sobre as pressoes inflacionarias"]):
        return ("neutral", 0.0, "rule:external_mixed_inflation_growth_neutralizer")
    if is_strong_external_macro_context(lowered) and contains_rule_term(lowered, ["permanece favoravel", "cenario externo favoravel"]):
        return ("neutral", 0.0, "rule:external_favorable_context_neutralizer")
    if contains_rule_term(lowered, ["inobservancia de taxas mais favoraveis", "reticencia dos participantes", "grau de preocupacao"]):
        return ("hawkish", 1.0, "rule:market_uncertainty_hawkish")
    if contains_rule_term(lowered, ["deficit fiscal", "deficit comercial"]) and contains_rule_term(
        lowered,
        ["duvidas quanto a sustentabilidade", "trajetorias crescentes", "sustentabilidade de suas trajetorias"],
    ):
        return ("hawkish", 1.0, "rule:external_fiscal_sustainability_risk")
    if contains_rule_term(lowered, ["piora na classificacao de risco", "piora da classificacao de risco"]):
        return ("hawkish", 1.0, "rule:fiscal_credit_rating_deterioration")
    if contains_rule_term(lowered, ["recorde historico de baixa", "recorde historico de baixa"]) and contains_rule_term(
        lowered,
        ["vix", "volatilidade", "aversao ao risco"],
    ):
        return ("dovish", -1.0, "rule:market_volatility_low")
    if contains_rule_term(lowered, ["houve melhora em relacao ao cenario", "diminuicao da aversao ao risco", "diminuição da aversão ao risco"]):
        return ("dovish", -1.0, "rule:market_uncertainty_easing")
    if is_strong_external_macro_context(lowered) and contains_rule_term(lowered, ["contracao da economia global estao enfraquecendo"]):
        return ("hawkish", 1.0, "rule:external_growth_recovery")
    if is_strong_external_macro_context(lowered) and contains_rule_term(lowered, ["riscos de baixa", "contracao", "desaceleracao"]):
        return ("dovish", -1.0, "rule:external_growth_downside")
    if contains_rule_term(lowered, ["remocao de estimulos", "retirada de estimulos", "retirada dos estimulos"]):
        return ("hawkish", 1.0, "rule:external_policy_stimulus_removal")
    if contains_rule_term(lowered, ["aumenta os riscos", "aumentar os riscos", "aumentam os riscos", "riscos para o cenario benigno"]):
        return ("hawkish", 1.0, "rule:risk_balance_upside")
    if is_actual_policy_decision_text(lowered) and contains_rule_term(lowered, ["reduzir", "reduziu", "reducao", "corte"]):
        return ("dovish", -1.0, "rule:policy_easing_signal")
    if is_actual_policy_decision_text(lowered) and contains_rule_term(lowered, ["elevar", "elevou", "aumentar", "aumentou", "aumento"]):
        return ("hawkish", 1.0, "rule:policy_tightening_signal")
    if contains_rule_term(
        lowered,
        [
            "flexibilizacao",
            "estimulo monetario adicional",
            "vies de baixa",
            "reducao da taxa de juros",
            "reducoes de juros",
            "ciclo de corte",
            "aprofundar o ciclo de corte",
        ],
    ):
        if expectation_context and contains_rule_term(
            lowered,
            ["acima da meta", "acima das metas", "acima do objetivo", "acima do valor central", "piora das expectativas"],
        ):
            return ("hawkish", 1.0, "rule:expectations_above_target_guard")
        return ("dovish", -1.0, "rule:policy_easing_signal")
    if contains_rule_term(lowered, ["postura firme", "atue de forma prudente", "atuacao prudente"]):
        return ("hawkish", 1.0, "rule:policy_caution_signal")
    if contains_rule_term(lowered, ["taxa real de juros"]) and contains_rule_term(lowered, ["elevou-se", "aumentou", "alta"]):
        return ("hawkish", 1.0, "rule:real_rate_tightening")
    if expectation_context and contains_rule_term(
        lowered,
        ["diminuiu a probabilidade de rompimento", "menor probabilidade de rompimento", "probabilidade de rompimento diminuiu"],
    ):
        return ("neutral", 0.0, "rule:expectations_mixed_above_target_probability_down")
    if expectation_context and contains_rule_term(
        lowered,
        ["bem ancoradas", "bem ancorada", "alinhadas com a meta", "alinhada com a meta"],
    ):
        return ("dovish", -1.0, "rule:expectations_anchored_after_minor_increase")
    if expectation_context and contains_rule_term(
        lowered,
        [
            "elevacao",
            "tendencia de elevacao",
            "passou de",
            "acima da meta",
            "acima das metas",
            "acima do objetivo",
            "acima do valor central",
            "desancor",
            "deterioracao",
            "afastamento da meta",
            "piora das expectativas",
        ],
    ):
        return ("hawkish", 1.0, "rule:expectations_upside")
    if expectation_context and contains_rule_term(
        lowered,
        ["reduziram", "reducao", "mais proximas da meta", "ficando mais proximas da meta", "ancoradas", "melhora das expectativas"],
    ):
        return ("dovish", -1.0, "rule:expectations_downside")
    if risk_context and contains_rule_term(lowered, ["baixa", "reducao", "diminuiu", "decrescentes", "arrefecimento", "arrefecer", "mitigacao"]):
        return ("dovish", -1.0, "rule:risk_balance_downside")
    if risk_context and contains_rule_term(lowered, ["menos efetiva", "continuam sendo relevantes os riscos", "riscos continuam relevantes"]):
        return ("hawkish", 1.0, "rule:risk_balance_upside")
    if risk_context and contains_rule_term(
        lowered,
        ["alta", "altista", "deterioracao", "aumento", "maior probabilidade", "expansao da demanda", "intensificacao", "elevando"],
    ):
        return ("hawkish", 1.0, "rule:risk_balance_upside")
    if contains_rule_term(lowered, ["reduzidos de ociosidade", "reduzida ociosidade", "ociosidade reduzida", "esgotamento da margem de ociosidade"]):
        return ("hawkish", 1.0, "rule:low_slack_activity_hawkish")
    if contains_rule_term(lowered, labor_terms) and contains_rule_term(
        lowered,
        [
            "taxa de desemprego recuou",
            "taxa de desocupacao foi",
            "contingente de ocupados",
            "pessoas ocupadas subiu",
            "foram criados",
            "postos de trabalho",
            "menor nivel da serie",
            "minimo da serie",
            "queda de 1,3 p.p. na taxa de desemprego",
            "rendimento medio real habitual avancou",
        ],
    ):
        return ("hawkish", 1.0, "rule:labor_market_tightening")
    if contains_rule_term(lowered, activity_terms) and contains_rule_term(
        lowered,
        [
            "evoluiu positivamente",
            "desempenho sera superior",
            "atingiu",
            "manteve-se acima",
            "crescimento de",
            "apresentou crescimento",
            "tendencia de crescimento",
            "permaneceram com tendencia de crescimento",
            "confirmam que",
            "expansao de 7,8",
            "mostrando sinais inequivocos da aceleracao",
            "cresceu nos tres meses",
        ],
    ):
        return ("hawkish", 1.0, "rule:activity_strength")
    if contains_rule_term(lowered, activity_terms) and contains_rule_term(lowered, ["apos queda", "apos dois meses consecutivos em queda"]) and contains_rule_term(
        lowered,
        ["expansao", "cresceu", "altas de"],
    ):
        return ("hawkish", 1.0, "rule:activity_rebound_after_drop")
    if contains_rule_term(lowered, activity_terms + labor_terms) and contains_rule_term(
        lowered,
        [
            "declinaram",
            "declinou",
            "diminuiu",
            "recuo",
            "recuou",
            "queda",
            "queda da producao",
            "queda de producao",
            "quedas mensais",
            "desemprego cresceu",
            "desemprego elevado",
            "recessao",
            "deterioracao",
            "crescimento nulo",
            "contracao",
            "desaceleracao",
            "ociosidade",
            "perdeu tracao",
            "ritmo mais lento",
            "taxas reduzidas",
        ],
    ):
        return ("dovish", -1.0, "rule:activity_or_labor_weakness")
    if contains_rule_term(lowered, inflation_terms) and contains_rule_term(
        lowered,
        ["voltou a subir", "pressao altista adicional", "a despeito de arrefecimento", "ante 3,39", "ante 3,39%"],
    ):
        return ("hawkish", 1.0, "rule:inflation_reacceleration_guard")
    if contains_rule_term(lowered, inflation_terms) and contains_rule_term(
        lowered,
        [
            "mitigar",
            "contidas",
            "conter pressoes",
            "queda de preco",
            "quedas de preco",
            "reducao das pressoes",
            "reducao da inflacao",
            "processo de reducao",
            "reducao das variacoes",
            "inexistencia de pressoes",
            "pressões negativas",
            "pressoes negativas",
            "esgotamento das pressoes",
            "segue sendo limitada",
            "nao resultar em pressoes",
            "arrefecimento",
            "desaceleracao",
            "ociosidade",
            "deflacao",
            "menor inflacao",
        ],
    ):
        return ("dovish", -1.0, "rule:inflation_pressure_easing")
    if contains_rule_term(lowered, inflation_terms) and contains_rule_term(
        lowered,
        ["voltou a subir", "pressao altista adicional", "a despeito de arrefecimento", "ante 3,39", "ante 3,39%"],
    ):
        return ("hawkish", 1.0, "rule:inflation_reacceleration_guard")
    if contains_rule_term(lowered, inflation_terms) and contains_rule_term(
        lowered,
        [
            "pressoes inflacionarias",
            "pressao inflacionaria",
            "aceleracao",
            "mostrou aceleracao",
            "mostraram aceleracao",
            "evidenciou aceleracao",
            "elevacao",
            "aumento",
            "persistencia",
            "resiliencia",
            "intensificando",
            "permanecer",
            "registrou",
            "totalizando",
            "acumulou alta",
            "passaram de variacoes negativas",
            "passaram de variacoes negativas para elevacoes",
            "taxas maiores",
            "afetaram os precos",
            "impacto importante",
        ],
    ):
        return ("hawkish", 1.0, "rule:inflation_pressure_upside")
    if contains_rule_term(lowered, ["impacto defasado", "repasse de precos", "patamares bastante elevados"]):
        return ("hawkish", 1.0, "rule:inflation_or_fx_pass_through_risk")
    if primary_topic in {"inflation_current", "inflation_expectations"} and (
        contains_rule_term(lowered, ["abaixo da meta", "dados recentes da inflacao"])
        or ("perspectivas favoraveis" in lowered and contains_rule_term(lowered, ["inflacao", "ipca", "precos"]))
    ):
        return ("dovish", -1.0, "rule:inflation_benign_outlook")
    if contains_rule_term(lowered, ["patamar recorde", "taxas anualizadas superiores", "tendencia ascendente"]):
        return ("hawkish", 1.0, "rule:macro_strength")
    if primary_topic == "forward_guidance" and contains_rule_term(
        lowered,
        ["convergencia da inflacao para a trajetoria de metas", "convergencia da inflacao para a meta"],
    ):
        return ("neutral", 0.0, "rule:guidance_convergence_statement_neutral")
    if primary_topic == "credit_conditions" and contains_rule_term(
        lowered,
        ["substancial melhora nas condicoes de credito", "condicoes favoraveis do credito", "expansoes", "expansao de"],
    ):
        return ("hawkish", 1.0, "rule:credit_expansion")
    if contains_rule_term(lowered, activity_terms + labor_terms) and contains_rule_term(
        lowered,
        [
            "forte crescimento",
            "forte expansao",
            "trajetoria ascendente",
            "aceleracao",
            "acelerou",
            "cresceu",
            "cresceram",
            "crescimento real",
            "crescimento do volume",
            "expansao",
            "elevou-se",
            "elevacao",
            "registrou elevacao",
            "acrescimos",
            "alta de",
            "aumentou",
            "aumento das vendas",
            "recorde",
            "patamar recorde",
            "niveis elevados",
            "mercado de trabalho apertado",
            "demanda aquecida",
        ],
    ):
        return ("hawkish", 1.0, "rule:activity_or_labor_strength")
    if primary_topic == "credit_conditions" and contains_rule_term(lowered, ["obstrucao do credito", "contracao do credito", "restricao de credito"]):
        return ("dovish", -1.0, "rule:credit_contraction")
    if contains_rule_term(lowered, ["afastado o espectro", "afastado o risco"]) and contains_rule_term(
        lowered,
        ["elevacao rapida dos juros", "alta dos juros", "aperto monetario"],
    ):
        return ("dovish", -1.0, "rule:tightening_risk_receded")
    if contains_rule_term(lowered, ["cortes das respectivas taxas"]):
        return ("dovish", -1.0, "rule:external_policy_easing")
    if contains_rule_term(lowered, ["processo de ajuste", "acao preventiva", "aperto monetario"]):
        return ("hawkish", 1.0, "rule:policy_tightening_signal")
    if contains_rule_term(lowered, ["petroleo", "brent", "commodities"]) and contains_rule_term(
        lowered,
        ["pressao altista adicional", "pressoes altistas adicionais", "sofreram pressao altista"],
    ):
        return ("hawkish", 1.0, "rule:commodities_upside")
    if contains_rule_term(lowered, ["petroleo", "brent", "commodities"]) and contains_rule_term(lowered, downside_terms):
        return ("dovish", -1.0, "rule:commodities_downside")
    if contains_rule_term(lowered, ["petroleo", "brent", "commodities"]) and contains_rule_term(lowered, upside_terms):
        return ("hawkish", 1.0, "rule:commodities_upside")
    if contains_rule_term(lowered, ["petroleo", "brent", "commodities"]) and contains_rule_term(
        lowered,
        ["manteve acima", "se manteve acima", "patamar elevado", "patamares elevados", "mudanca permanente de patamar"],
    ):
        return ("hawkish", 1.0, "rule:commodities_upside")
    if contains_rule_term(lowered, ["libor", "taxa overnight", "ois"]) and contains_rule_term(lowered, ["recuando", "recuo", "queda"]):
        return ("dovish", -1.0, "rule:external_rates_downside")
    if contains_rule_term(lowered, ["premio de risco", "risco-brasil", "risco brasil"]) and contains_rule_term(
        lowered,
        ["mas esse movimento nao se sustentou", "movimento nao se sustentou"],
    ):
        return ("neutral", 0.0, "rule:risk_premium_temporary_move_neutral")
    if contains_rule_term(lowered, ["premio de risco", "risco-brasil", "risco brasil"]) and (
        contains_rule_term(lowered, downside_terms) or contains_rule_term(lowered, ["quebrou a marca", "novo minimo historico"])
    ):
        return ("dovish", -1.0, "rule:risk_premium_down")
    if contains_rule_term(lowered, ["reducao das preocupacoes", "menor preocupacao"]) and contains_rule_term(
        lowered,
        ["fiscal", "deterioracao", "risco", "incerteza"],
    ):
        return ("dovish", -1.0, "rule:risk_concern_easing")
    if contains_rule_term(lowered, ["cambio", "taxa de cambio", "dolar", "real"]) and contains_rule_term(
        lowered,
        ["apreciacao", "valorizacao do real", "moeda domestica apreciada", "real apreciado"],
    ):
        return ("dovish", -1.0, "rule:currency_appreciation")
    return None


def matched_topic_names(text: str, topics: dict[str, Any]) -> list[str]:
    matches = [
        topic
        for topic, details in topics.items()
        if topic != "institutional" and contains_any(text, details.get("keywords", [])) > 0
    ]
    return matches or ["institutional"]


def primary_topic_name(text: str, topics: dict[str, Any], matched_topics: list[str]) -> str:
    if matched_topics == ["institutional"]:
        return "institutional"
    counts = {topic: contains_any(text, topics.get(topic, {}).get("keywords", [])) for topic in matched_topics}
    return max(counts, key=counts.get) if counts else "institutional"


def score_terms(text: str, terms: list[dict[str, Any]]) -> tuple[float, list[str]]:
    lowered = strip_accents(text.lower())
    score = 0.0
    matched: list[str] = []
    for item in terms:
        term = item["term"]
        if strip_accents(term.lower()) in lowered:
            score += float(item.get("weight", 1.0))
            matched.append(term)
    return score, matched


def confidence_score(evidence_terms: list[str], stance_score: float, matched_topics: list[str]) -> float:
    if not evidence_terms and matched_topics == ["institutional"]:
        return 0.15
    if not evidence_terms:
        return 0.35
    return float(min(0.98, 0.40 + 0.10 * len(evidence_terms) + 0.25 * abs(stance_score)))


def infer_information_weight(
    text: str,
    primary_topic: str,
    matched_topics: list[str],
    stance: str,
    low_information_threshold: float,
) -> float:
    if primary_topic == "institutional":
        return 0.0
    weight = 0.60 if matched_topics and matched_topics != ["institutional"] else 0.0
    if stance != "neutral":
        weight += 0.25
    if len(text) >= 120:
        weight += 0.10
    return float(max(0.0, min(1.0, weight)))


def v2_rationale(stance: str, topic: str, evidence_terms: list[str], is_informative: bool) -> str:
    if not is_informative:
        return f"Low-information or institutional sentence for V2 scoring; primary topic={topic}."
    if evidence_terms:
        return f"Baseline terms support {stance} stance in {topic}: {', '.join(evidence_terms[:4])}."
    return f"Informative macro sentence in {topic}; stance remains {stance} under baseline lexicon."


def select_doc_score(document_scores: pd.DataFrame, document_type: str) -> float:
    subset = document_scores[document_scores["document_type"] == document_type]
    if subset.empty:
        return np.nan
    return float(subset.iloc[0]["document_tone_raw"])


def build_previous_redline_index(previous: pd.DataFrame) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    exact_lookup: dict[str, list[int]] = {}
    token_lookup: dict[str, set[int]] = {}
    if previous.empty:
        return {"records": records, "exact_lookup": exact_lookup, "token_lookup": token_lookup}
    for idx, record in enumerate(previous.to_dict("records")):
        normalized = str(record.get("_redline_normalized") or redline_normalize(str(record.get("text", ""))))
        tokens = record.get("_redline_tokens")
        if not isinstance(tokens, set):
            tokens = redline_tokens(normalized)
        record["_redline_normalized"] = normalized
        record["_redline_tokens"] = tokens
        records.append(record)
        if normalized:
            exact_lookup.setdefault(normalized, []).append(idx)
        for token in tokens:
            token_lookup.setdefault(str(token), set()).add(idx)
    return {"records": records, "exact_lookup": exact_lookup, "token_lookup": token_lookup}


def best_previous_match_from_index(
    text: str,
    previous_index: dict[str, Any],
    used_previous: set[str],
    max_candidates: int = 12,
) -> tuple[dict[str, Any] | None, float]:
    records: list[dict[str, Any]] = previous_index.get("records", [])
    if not records:
        return None, 0.0
    normalized = redline_normalize(text)
    current_tokens = redline_tokens(normalized)
    for idx in previous_index.get("exact_lookup", {}).get(normalized, []):
        record = records[idx]
        if record.get("sentence_id") not in used_previous:
            return record, 1.0
    candidate_indices: set[int] = set()
    token_lookup: dict[str, set[int]] = previous_index.get("token_lookup", {})
    for token in current_tokens:
        candidate_indices.update(token_lookup.get(token, set()))
    candidates: list[tuple[float, int, dict[str, Any]]] = []
    for idx in candidate_indices:
        record = records[idx]
        if record.get("sentence_id") in used_previous:
            continue
        overlap_score = token_overlap_score(current_tokens, record.get("_redline_tokens", set()))
        if overlap_score > 0:
            candidates.append((overlap_score, idx, record))
    if candidates:
        candidate_rows = [item[2] for item in sorted(candidates, key=lambda item: (-item[0], item[1]))[:max_candidates]]
    else:
        candidate_rows = [record for record in records if record.get("sentence_id") not in used_previous][:max_candidates]
    best_row: dict[str, Any] | None = None
    best_similarity = 0.0
    for row in candidate_rows:
        previous_normalized = str(row.get("_redline_normalized", redline_normalize(str(row.get("text", "")))))
        similarity = SequenceMatcher(None, normalized, previous_normalized).ratio()
        if similarity > best_similarity:
            best_similarity = similarity
            best_row = row
    return best_row, best_similarity


def best_previous_match(text: str, previous: pd.DataFrame, used_previous: set[str], max_candidates: int = 12) -> tuple[pd.Series | None, float]:
    previous_index = build_previous_redline_index(previous)
    row, similarity = best_previous_match_from_index(text, previous_index, used_previous, max_candidates=max_candidates)
    if row is None:
        return None, 0.0
    return pd.Series(row), similarity


def redline_normalize(text: str) -> str:
    return strip_accents(normalize_whitespace(text).lower())


def redline_tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]{4,}", strip_accents(str(text).lower())))


def token_overlap_score(left: set[str], right: Any) -> float:
    if not isinstance(right, set):
        right = set(right) if isinstance(right, (list, tuple)) else set()
    if not left or not right:
        return 0.0
    intersection = len(left & right)
    if intersection == 0:
        return 0.0
    return intersection / len(left | right)


def redline_is_current(redline: pd.DataFrame, documents: pd.DataFrame, sentence_scores: pd.DataFrame, min_coverage: float = 0.98) -> bool:
    if redline.empty or documents.empty:
        return False
    expected = expected_redline_pairs(documents, sentence_scores)
    if expected == 0:
        return False
    actual = redline_document_pair_count(redline)
    if actual / expected < min_coverage:
        return False
    if "current_sentence_id" not in redline or "previous_sentence_id" not in redline:
        return False
    current_ids = set(sentence_scores.get("sentence_id", pd.Series(dtype=str)).dropna().astype(str))
    redline_ids = set(redline["current_sentence_id"].dropna().astype(str)) | set(redline["previous_sentence_id"].dropna().astype(str))
    redline_ids.discard("")
    return redline_ids.issubset(current_ids)


def redline_document_pair_count(redline: pd.DataFrame) -> int:
    if redline.empty or not {"meeting_id", "document_type"}.issubset(redline.columns):
        return 0
    return int(redline.groupby(["meeting_id", "document_type"]).ngroups)


def expected_redline_pairs(documents: pd.DataFrame, sentence_scores: pd.DataFrame) -> int:
    if documents.empty or "document_type" not in documents or sentence_scores.empty or "document_id" not in sentence_scores:
        return 0
    docs_with_sentences = set(sentence_scores["document_id"].dropna().astype(str))
    expected = 0
    for _, group in documents.sort_values(["document_type", "nro_reuniao"]).groupby("document_type"):
        previous_doc_id: str | None = None
        previous_has_sentences = False
        for _, doc in group.iterrows():
            current_doc_id = str(doc["document_id"])
            current_has_sentences = current_doc_id in docs_with_sentences
            if previous_doc_id is not None and (previous_has_sentences or current_has_sentences):
                expected += 1
            previous_doc_id = current_doc_id
            previous_has_sentences = current_has_sentences
    return expected


def existing_v2_backfill_is_sufficient(database: Path, quantity: int | None = None) -> bool:
    meetings = read_optional_table(database, "v2_meetings", pd.DataFrame())
    documents = read_optional_table(database, "v2_documents", pd.DataFrame())
    if meetings.empty or documents.empty:
        return False
    meeting_count = int(meetings["meeting_id"].nunique()) if "meeting_id" in meetings else len(meetings)
    if quantity is None:
        return meeting_count >= 50
    target = min(int(quantity), 200)
    return meeting_count >= target


def _meetings_from_v2_documents(documents: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for meeting_id, group in documents.groupby("meeting_id"):
        first = group.sort_values("publication_date").iloc[0]
        rows.append(
            {
                "meeting_id": meeting_id,
                "nro_reuniao": int(first["nro_reuniao"]),
                "data_referencia": pd.to_datetime(first["publication_date"], errors="coerce"),
            }
        )
    return pd.DataFrame(rows)


def _known_at_timestamp(row: pd.Series) -> pd.Timestamp:
    publication_date = pd.to_datetime(row.get("publication_date"), errors="coerce")
    if pd.isna(publication_date):
        return pd.NaT
    if row.get("document_type") == "comunicado":
        return publication_date.normalize() + pd.Timedelta(hours=18, minutes=30)
    return publication_date.normalize() + pd.Timedelta(hours=8)


def make_run_id(prefix: str) -> str:
    timestamp = pd.Timestamp.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{timestamp}_{hashlib.sha256(prefix.encode('utf-8')).hexdigest()[:6]}"


def stable_text_hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def stable_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)


def utc_now_naive() -> pd.Timestamp:
    return pd.Timestamp.now(tz=timezone.utc).tz_localize(None).floor("s")


def fmt_number(value: Any) -> str:
    if pd.isna(value):
        return "n.d."
    return f"{float(value):.2f}"


def dataframe_to_html(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "<p class='muted'>Sem dados disponiveis.</p>"
    return frame.to_html(index=False, escape=True)


def validate_v2_schema(database: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not database.exists():
        return pd.DataFrame([{"table": table, "exists": False} for table in V2_TABLES])
    with duckdb.connect(str(database), read_only=True) as con:
        existing = {
            row[0]
            for row in con.execute("SELECT table_name FROM information_schema.tables").fetchall()
        }
    for table in V2_TABLES:
        rows.append({"table": table, "exists": table in existing})
    return pd.DataFrame(rows)


def coalesce_llm_with_baseline_for_test(baseline: pd.DataFrame, llm_updates: pd.DataFrame) -> pd.DataFrame:
    if llm_updates.empty:
        return baseline.copy()
    columns = [
        "sentence_id",
        "topic",
        "stance",
        "stance_score",
        "confidence",
        "rationale",
        "evidence_terms",
        "model_version",
        "prompt_version",
    ]
    updated = baseline.merge(llm_updates[columns], on="sentence_id", how="left", suffixes=("", "_llm"))
    for column in columns:
        if column == "sentence_id":
            continue
        updated[column] = updated[f"{column}_llm"].combine_first(updated[column])
        updated = updated.drop(columns=[f"{column}_llm"])
    return updated

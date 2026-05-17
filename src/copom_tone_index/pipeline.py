from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from copom_tone_index.bcb import (
    attach_selic_to_meetings,
    date_window_for_sources,
    fetch_copom_documents,
    fetch_selic,
)
from copom_tone_index.config import ensure_directories, get_paths, load_lexicon, load_settings, load_topics
from copom_tone_index.focus import (
    build_focus_revisions_from_observations,
    empty_focus_observations,
    fetch_focus_observations_for_meetings,
    read_optional_table,
    write_focus_coverage_outputs,
)
from copom_tone_index.http_client import CachedHttpClient
from copom_tone_index.nlp import classify_sentences_baseline, maybe_classify_with_llm
from copom_tone_index.reporting import generate_meeting_notes, write_figures
from copom_tone_index.scoring import aggregate_scores, evidence_sentences, topic_distribution_json
from copom_tone_index.storage import export_tables, write_tables
from copom_tone_index.text import attach_clean_text, build_sentence_frame
from copom_tone_index.validation import has_errors, validate_pipeline_outputs, write_validation_report

LOGGER = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    database: Path
    tables: dict[str, pd.DataFrame]
    validation_errors: int
    validation_warnings: int


def run_pipeline(
    months: int | None = None,
    use_llm: str = "auto",
    copom_quantity: int | None = None,
    refresh_focus: bool = False,
) -> PipelineResult:
    settings = load_settings()
    topics = load_topics()
    lexicon = load_lexicon()
    paths = get_paths(settings)
    ensure_directories(paths)

    months = months or int(settings["pipeline"]["months"])
    prompt_version = settings["project"]["prompt_version"]
    model_version = settings["project"]["default_model_version"]
    client = CachedHttpClient(
        cache_dir=paths.raw,
        timeout_seconds=int(settings["pipeline"]["request_timeout_seconds"]),
        retries=int(settings["pipeline"]["retries"]),
        retry_backoff_seconds=float(settings["pipeline"]["retry_backoff_seconds"]),
    )

    meetings, documents = fetch_copom_documents(
        client,
        quantity=int(copom_quantity or settings["pipeline"]["copom_quantity"]),
    )
    meetings = _attach_document_titles(meetings, documents)
    start_date, end_date = date_window_for_sources(meetings)
    selic = fetch_selic(client, start_date, end_date)
    meetings = attach_selic_to_meetings(meetings, selic)
    meetings = _mark_operational_window(meetings, months)

    focus_observations = read_optional_table(paths.database, "focus_observations", empty_focus_observations())
    if refresh_focus or focus_observations.empty:
        focus_client = CachedHttpClient(
            cache_dir=paths.raw / "focus",
            timeout_seconds=int(settings["pipeline"]["request_timeout_seconds"]),
            retries=int(settings["pipeline"]["retries"]),
            retry_backoff_seconds=float(settings["pipeline"]["retry_backoff_seconds"]),
        )
        focus_observations = fetch_focus_observations_for_meetings(
            focus_client,
            meetings[meetings["in_operational_window"]].copy(),
            settings["pipeline"]["focus_variables"],
        )
    focus_revisions = build_focus_revisions_from_observations(
        meetings,
        focus_observations,
        settings["pipeline"]["focus_variables"],
    )

    documents = attach_clean_text(documents)
    documents["model_version"] = model_version
    documents["prompt_version"] = prompt_version
    sentences = build_sentence_frame(documents)
    sentences = classify_sentences_baseline(
        sentences,
        topics=topics,
        lexicon=lexicon,
        prompt_version=prompt_version,
        model_version=model_version,
        neutral_band=float(settings["scoring"]["neutral_band"]),
    )
    sentences = maybe_classify_with_llm(
        sentences,
        topics=topics,
        prompt_version=prompt_version,
        cache_dir=paths.raw / "llm_cache",
        use_llm=use_llm,
    )
    _refresh_document_versions(documents, sentences)

    document_scores, topic_scores, scores = aggregate_scores(
        meetings,
        documents,
        sentences,
        topics,
        communication_weight=float(settings["scoring"]["communication_weight"]),
        minutes_weight=float(settings["scoring"]["minutes_weight"]),
        min_observations_for_surprise=int(settings["scoring"]["min_observations_for_surprise"]),
    )
    topic_json = topic_distribution_json(topic_scores)
    scores = scores.merge(topic_json, on="meeting_id", how="left")
    scores = scores.merge(
        meetings[["meeting_id", "selic_pre", "selic_pos", "delta_selic", "in_operational_window"]],
        on="meeting_id",
        how="left",
    )
    evidence = evidence_sentences(sentences)

    tables = {
        "copom_meetings": meetings,
        "copom_documents": documents,
        "copom_sentences": sentences,
        "copom_document_scores": document_scores,
        "copom_topic_scores": topic_scores,
        "copom_scores": scores,
        "focus_observations": focus_observations,
        "focus_revisions": focus_revisions,
        "evidence_sentences": evidence,
    }
    write_tables(paths.database, tables)
    export_tables(paths.database, paths.processed)
    write_focus_coverage_outputs(focus_revisions)
    write_figures(scores, topic_scores, paths.figures)
    generate_meeting_notes(meetings, scores, focus_revisions, evidence, paths.reports)

    issues = validate_pipeline_outputs(tables)
    write_validation_report(issues, paths.reports.parent / "validation_report.md")
    for issue in issues:
        log = LOGGER.error if issue.severity == "error" else LOGGER.warning
        log("%s: %s", issue.table, issue.message)
    return PipelineResult(
        database=paths.database,
        tables=tables,
        validation_errors=sum(1 for issue in issues if issue.severity == "error"),
        validation_warnings=sum(1 for issue in issues if issue.severity == "warning"),
    )


def _attach_document_titles(meetings: pd.DataFrame, documents: pd.DataFrame) -> pd.DataFrame:
    meetings = meetings.copy()
    titles = documents.pivot_table(index="meeting_id", columns="document_type", values="title", aggfunc="first")
    titles = titles.rename(columns={"comunicado": "titulo_comunicado", "ata": "titulo_ata"}).reset_index()
    return meetings.merge(titles, on="meeting_id", how="left")


def _mark_operational_window(meetings: pd.DataFrame, months: int) -> pd.DataFrame:
    meetings = meetings.copy()
    cutoff = meetings["data_referencia"].max() - pd.DateOffset(months=months)
    meetings["in_operational_window"] = meetings["data_referencia"] >= cutoff
    return meetings


def _refresh_document_versions(documents: pd.DataFrame, sentences: pd.DataFrame) -> None:
    if sentences.empty:
        return
    versions = sentences.groupby("document_id").agg(model_version=("model_version", "first"), prompt_version=("prompt_version", "first"))
    for document_id, row in versions.iterrows():
        mask = documents["document_id"] == document_id
        documents.loc[mask, "model_version"] = row["model_version"]
        documents.loc[mask, "prompt_version"] = row["prompt_version"]


def validate_existing_outputs() -> PipelineResult:
    from copom_tone_index.config import get_paths
    from copom_tone_index.storage import read_table

    paths = get_paths()
    table_names = [
        "copom_meetings",
        "copom_documents",
        "copom_sentences",
        "copom_scores",
        "focus_observations",
        "focus_revisions",
    ]
    tables: dict[str, pd.DataFrame] = {}
    for table in table_names:
        if table == "focus_observations":
            tables[table] = read_optional_table(paths.database, table, empty_focus_observations())
        else:
            tables[table] = read_table(paths.database, table)
    issues = validate_pipeline_outputs(tables)
    write_validation_report(issues, paths.reports.parent / "validation_report.md")
    return PipelineResult(
        database=paths.database,
        tables=tables,
        validation_errors=sum(1 for issue in issues if issue.severity == "error"),
        validation_warnings=sum(1 for issue in issues if issue.severity == "warning"),
    )

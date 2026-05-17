from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class ValidationIssue:
    severity: str
    table: str
    message: str


def validate_pipeline_outputs(tables: dict[str, pd.DataFrame]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    issues.extend(_validate_required(tables, "copom_meetings", ["meeting_id", "nro_reuniao", "data_referencia"]))
    issues.extend(_validate_required(tables, "copom_documents", ["document_id", "meeting_id", "document_type", "raw_text", "clean_text"]))
    issues.extend(_validate_required(tables, "copom_sentences", ["sentence_id", "document_id", "topic", "stance_score", "confidence"]))
    issues.extend(_validate_required(tables, "copom_scores", ["meeting_id", "tone_raw", "copom_tone_index", "classification"]))

    meetings = tables.get("copom_meetings", pd.DataFrame())
    if not meetings.empty:
        duplicated = meetings["meeting_id"].duplicated().sum()
        if duplicated:
            issues.append(ValidationIssue("error", "copom_meetings", f"{duplicated} duplicated meeting_id values."))
        if meetings["data_referencia"].isna().any():
            issues.append(ValidationIssue("error", "copom_meetings", "Missing data_referencia values."))

    sentences = tables.get("copom_sentences", pd.DataFrame())
    if not sentences.empty:
        if not sentences["stance_score"].between(-1, 1).all():
            issues.append(ValidationIssue("error", "copom_sentences", "stance_score outside [-1, 1]."))
        if not sentences["confidence"].between(0, 1).all():
            issues.append(ValidationIssue("error", "copom_sentences", "confidence outside [0, 1]."))

    scores = tables.get("copom_scores", pd.DataFrame())
    if not scores.empty:
        if scores["copom_tone_index"].isna().all():
            issues.append(ValidationIssue("warning", "copom_scores", "All COPOM Tone Index values are missing."))
        if scores["meeting_id"].duplicated().any():
            issues.append(ValidationIssue("error", "copom_scores", "Duplicated meeting scores."))

    focus = tables.get("focus_revisions", pd.DataFrame())
    if not focus.empty and focus["focus_pre_value"].isna().all():
        issues.append(
            ValidationIssue(
                "warning",
                "focus_revisions",
                "Focus revisions were created but no pre-event value was available; check OData availability.",
            )
        )
    return issues


def _validate_required(tables: dict[str, pd.DataFrame], table: str, columns: list[str]) -> list[ValidationIssue]:
    if table not in tables:
        return [ValidationIssue("error", table, "Table was not generated.")]
    frame = tables[table]
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        return [ValidationIssue("error", table, f"Missing columns: {missing}.")]
    if frame.empty:
        return [ValidationIssue("warning", table, "Table is empty.")]
    return []


def write_validation_report(issues: list[ValidationIssue], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not issues:
        path.write_text("# Validation Report\n\nNo validation issues found.\n", encoding="utf-8")
        return
    lines = ["# Validation Report", ""]
    for issue in issues:
        lines.append(f"- **{issue.severity.upper()}** `{issue.table}`: {issue.message}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def has_errors(issues: list[ValidationIssue]) -> bool:
    return any(issue.severity == "error" for issue in issues)

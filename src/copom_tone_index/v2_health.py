from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from copom_tone_index.config import get_paths, load_topics, load_v2_settings
from copom_tone_index.v2 import accepted_v2_labels, consensus_v2_labels, read_optional_table, utc_now_naive


@dataclass(frozen=True)
class V2HealthResult:
    status: str
    json_path: Path
    html_path: Path
    warnings: int
    errors: int


@dataclass(frozen=True)
class V2LabelSampleResult:
    output_path: Path
    codebook_path: Path
    rows: int


def v2_health_check_command() -> V2HealthResult:
    paths = get_paths()
    report = build_v2_acceptance_report(paths.database)
    output_dir = paths.reports.parent / "v2"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "acceptance_report.json"
    html_path = output_dir / "acceptance_report.html"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    html_path.write_text(render_acceptance_report_html(report, output_dir), encoding="utf-8")
    return V2HealthResult(
        status=str(report["status"]),
        json_path=json_path,
        html_path=html_path,
        warnings=len(report["warnings"]),
        errors=len(report["errors"]),
    )


def export_label_sample_command(n: int = 300, out: str | Path = "data/labels/review_sample_001.csv") -> V2LabelSampleResult:
    paths = get_paths()
    sentence_scores = read_optional_table(paths.database, "v2_sentence_scores", pd.DataFrame())
    subindices = read_optional_table(paths.database, "v2_subindices", pd.DataFrame())
    labels = read_optional_table(paths.database, "v2_labels", pd.DataFrame())
    reviewed_sentence_ids = set()
    if not labels.empty and "sentence_id" in labels:
        reviewed = accepted_v2_labels(labels)
        reviewed_sentence_ids = set(reviewed["sentence_id"].dropna().astype(str)) if not reviewed.empty else set()
    sample = build_v2_label_review_sample(sentence_scores, subindices, n=n, exclude_sentence_ids=reviewed_sentence_ids)
    output_path = Path(out)
    if not output_path.is_absolute():
        output_path = paths.database.parents[1] / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(output_path, index=False)
    codebook_path = label_codebook_path(output_path)
    codebook_path.write_text(build_v2_label_codebook(load_topics(), load_v2_settings(), output_path), encoding="utf-8")
    return V2LabelSampleResult(output_path=output_path, codebook_path=codebook_path, rows=len(sample))


def build_v2_acceptance_report(database: Path) -> dict[str, Any]:
    settings = load_v2_settings()
    tables = load_health_tables(database)
    warnings: list[str] = []
    errors: list[str] = []

    coverage = coverage_section(tables)
    documents = documents_section(tables)
    sentences = sentences_section(tables)
    scores = scores_section(tables)
    calibration = calibration_section(tables, settings)
    formula_check = formula_section(tables)
    subindices = subindices_section(tables, settings)
    redline = redline_section(tables)
    evidence = evidence_section(tables)
    labels = labels_section(tables)
    focus_v21 = focus_v21_section(tables)
    market = market_section(tables)
    v21_event_panel = v21_event_panel_section(tables)
    semantic = semantic_section(tables)
    idempotency = idempotency_section(tables)

    collect_findings(
        warnings,
        errors,
        coverage,
        documents,
        sentences,
        scores,
        calibration,
        formula_check,
        subindices,
        redline,
        evidence,
        labels,
        focus_v21,
        market,
        v21_event_panel,
        semantic,
        idempotency,
    )

    status = "fail" if errors else "warning" if warnings else "pass"
    implementation_status = implementation_status_section(
        status,
        coverage,
        documents,
        sentences,
        scores,
        calibration,
        formula_check,
        subindices,
        redline,
        evidence,
        labels,
        focus_v21,
        market,
        v21_event_panel,
        semantic,
        idempotency,
    )
    return {
        "status": status,
        "generated_at": utc_now_naive().isoformat(),
        "database_path": str(database),
        "implementation_status": implementation_status,
        "coverage": coverage,
        "documents": documents,
        "sentences": sentences,
        "scores": scores,
        "calibration": calibration,
        "formula_check": formula_check,
        "subindices": subindices,
        "redline": redline,
        "evidence": evidence,
        "labels": labels,
        "focus_v21": focus_v21,
        "market": market,
        "v21_event_panel": v21_event_panel,
        "semantic": semantic,
        "idempotency": idempotency,
        "warnings": warnings,
        "errors": errors,
    }


def load_health_tables(database: Path) -> dict[str, pd.DataFrame]:
    table_names = [
        "v2_meetings",
        "v2_documents",
        "v2_sentences",
        "v2_sentence_scores",
        "v2_document_scores",
        "v2_meeting_scores",
        "v2_subindices",
        "v2_calibration",
        "v2_redline",
        "v2_evidence",
        "v2_labels",
        "v2_model_predictions",
        "v2_model_audit",
        "focus_vintages",
        "focus_event_features",
        "decision_expectations",
        "decision_expectation_source_audit",
        "v21_event_panel",
        "market_observations",
        "market_event_windows",
        "public_market_source_audit",
        "public_market_coverage",
        "semantic_chunks",
    ]
    tables: dict[str, pd.DataFrame] = {}
    for table in table_names:
        tables[table] = read_optional_table(database, table, pd.DataFrame())
    return tables


def coverage_section(tables: dict[str, pd.DataFrame]) -> dict[str, Any]:
    meetings = tables["v2_meetings"]
    documents = tables["v2_documents"]
    if meetings.empty and not documents.empty:
        meetings = documents[["meeting_id", "nro_reuniao"]].drop_duplicates()
    meeting_ids = set(meetings.get("meeting_id", pd.Series(dtype=str)).dropna().astype(str))
    atas = documents[documents.get("document_type", "") == "ata"] if not documents.empty else pd.DataFrame()
    comunicados = documents[documents.get("document_type", "") == "comunicado"] if not documents.empty else pd.DataFrame()
    meetings_with_ata = set(atas.get("meeting_id", pd.Series(dtype=str)).dropna().astype(str))
    meetings_with_comunicado = set(comunicados.get("meeting_id", pd.Series(dtype=str)).dropna().astype(str))
    both = meetings_with_ata & meetings_with_comunicado
    min_meeting_number = numeric_min(meetings, "nro_reuniao")
    max_meeting_number = numeric_max(meetings, "nro_reuniao")
    return {
        "total_meetings": int(len(meeting_ids)),
        "min_meeting_id": meeting_id_for_number(meetings, min_meeting_number) or safe_min(meeting_ids),
        "max_meeting_id": meeting_id_for_number(meetings, max_meeting_number) or safe_max(meeting_ids),
        "min_meeting_number": min_meeting_number,
        "max_meeting_number": max_meeting_number,
        "total_documents": int(len(documents)),
        "minutes_documents": int(len(atas)),
        "statement_documents": int(len(comunicados)),
        "meetings_with_minutes": int(len(meetings_with_ata)),
        "meetings_with_statement": int(len(meetings_with_comunicado)),
        "meetings_with_both": int(len(both)),
        "meetings_without_minutes": sorted(meeting_ids - meetings_with_ata),
        "meetings_without_statement": sorted(meeting_ids - meetings_with_comunicado),
        "duplicate_documents_by_meeting_type": duplicate_count(documents, ["meeting_id", "document_type"]),
    }


def documents_section(tables: dict[str, pd.DataFrame]) -> dict[str, Any]:
    documents = tables["v2_documents"]
    if documents.empty:
        return {
            "raw_text_null": 0,
            "raw_text_too_short": 0,
            "missing_source_hash": 0,
            "missing_run_id": 0,
            "missing_collected_at": 0,
            "missing_source_url": 0,
            "status": "missing_documents",
        }
    raw = documents["raw_text"] if "raw_text" in documents else pd.Series([np.nan] * len(documents))
    url_col = "source_url" if "source_url" in documents else "url" if "url" in documents else None
    missing_source_url = (
        int((documents[url_col].isna() | (documents[url_col].fillna("").astype(str).str.strip() == "")).sum())
        if url_col
        else 0
    )
    return {
        "raw_text_null": int(raw.isna().sum()),
        "raw_text_too_short": int(raw.fillna("").astype(str).str.len().lt(100).sum()),
        "missing_source_hash": missing_count(documents, "source_hash"),
        "missing_run_id": missing_count(documents, "run_id"),
        "missing_collected_at": missing_count(documents, "collected_at"),
        "missing_source_url": missing_source_url,
        "source_url_column": url_col or "",
        "status": "ok",
    }


def sentences_section(tables: dict[str, pd.DataFrame]) -> dict[str, Any]:
    documents = tables["v2_documents"]
    sentences = tables["v2_sentences"]
    scores = tables["v2_sentence_scores"]
    informative = scores[scores["is_informative"].fillna(False).astype(bool)] if not scores.empty and "is_informative" in scores else pd.DataFrame()
    doc_ids = set(documents.get("document_id", pd.Series(dtype=str)).dropna().astype(str))
    invalid_doc = 0
    if not sentences.empty and "document_id" in sentences:
        invalid_doc = int((~sentences["document_id"].astype(str).isin(doc_ids)).sum())
    by_document = {}
    if not sentences.empty and "document_id" in sentences:
        by_document = to_int_dict(sentences.groupby("document_id").size())
    docs_without_informative: list[str] = []
    if not documents.empty and not informative.empty:
        informative_docs = set(informative["document_id"].dropna().astype(str))
        docs_without_informative = sorted(doc_ids - informative_docs)
    elif not documents.empty:
        docs_without_informative = sorted(doc_ids)
    return {
        "total_sentences": int(len(sentences)),
        "sentences_by_document": by_document,
        "informative_sentences": int(len(informative)),
        "informative_sentence_pct": safe_ratio(len(informative), len(sentences)),
        "documents_without_informative_sentence": docs_without_informative,
        "empty_or_short_text_sentences": int(sentences.get("text", pd.Series(dtype=str)).fillna("").astype(str).str.len().lt(20).sum()) if not sentences.empty else 0,
        "duplicate_sentence_order_by_document": duplicate_count(sentences, ["document_id", "sentence_order"]),
        "sentences_without_valid_document_id": invalid_doc,
    }


def scores_section(tables: dict[str, pd.DataFrame]) -> dict[str, Any]:
    documents = tables["v2_documents"]
    meetings = tables["v2_meetings"]
    sentence_scores = tables["v2_sentence_scores"]
    document_scores = tables["v2_document_scores"]
    meeting_scores = tables["v2_meeting_scores"]
    informative = (
        sentence_scores[sentence_scores["is_informative"].fillna(False).astype(bool)]
        if not sentence_scores.empty and "is_informative" in sentence_scores
        else pd.DataFrame()
    )
    docs_without_score = sorted(
        set(documents.get("document_id", pd.Series(dtype=str)).dropna().astype(str))
        - set(document_scores.get("document_id", pd.Series(dtype=str)).dropna().astype(str))
    )
    meetings_without_score = sorted(
        set(meetings.get("meeting_id", pd.Series(dtype=str)).dropna().astype(str))
        - set(meeting_scores.get("meeting_id", pd.Series(dtype=str)).dropna().astype(str))
    )
    tone_distribution = distribution(sentence_scores.get("tone_level", pd.Series(dtype=float))) if not sentence_scores.empty else {}
    sort_col = "tone_raw" if "tone_raw" in meeting_scores else "copom_tone_index_v2"
    top_hawkish = top_meetings(meeting_scores, sort_col, ascending=False)
    top_dovish = top_meetings(meeting_scores, sort_col, ascending=True)
    if not meeting_scores.empty and "tone_raw" in meeting_scores:
        null_tone = meeting_scores[meeting_scores["tone_raw"].isna()]
    else:
        null_tone = meeting_scores.copy() if not meeting_scores.empty else pd.DataFrame()
    if not null_tone.empty and "score_status" in null_tone:
        expected_null = null_tone[null_tone["score_status"].isin(["low_information", "no_informative_sentences"])]
        unexpected_null = null_tone[~null_tone["score_status"].isin(["low_information", "no_informative_sentences"])]
    else:
        expected_null = pd.DataFrame()
        unexpected_null = null_tone
    return {
        "sentence_scores": int(len(sentence_scores)),
        "sentence_scores_by_calibration_version": value_counts(sentence_scores, "calibration_version"),
        "sentence_scores_by_model_version": value_counts(sentence_scores, "model_version"),
        "informative_without_tone_level": missing_count(informative, "tone_level"),
        "informative_without_stance_score": missing_count(informative, "stance_score"),
        "informative_without_topic_weight": missing_count(informative, "topic_weight"),
        "informative_without_confidence": missing_count(informative, "confidence"),
        "informative_without_information_weight": missing_count(informative, "information_weight"),
        "documents_without_aggregate_score": docs_without_score,
        "meetings_without_meeting_score": meetings_without_score,
        "meetings_with_null_final_tone": int(len(null_tone)),
        "meetings_with_expected_low_information_null_tone": int(len(expected_null)),
        "meetings_with_unexpected_null_final_tone": int(len(unexpected_null)),
        "tone_level_distribution": tone_distribution,
        "top_10_hawkish": top_hawkish,
        "top_10_dovish": top_dovish,
    }


def calibration_section(tables: dict[str, pd.DataFrame], settings: dict[str, Any]) -> dict[str, Any]:
    calibration = tables["v2_calibration"]
    meeting_scores = tables["v2_meeting_scores"]
    default_version = str(settings["scoring"]["default_calibration"])
    versions = calibration.get("calibration_version", pd.Series(dtype=str)).dropna().astype(str).tolist()
    details = calibration.to_dict("records") if not calibration.empty else []
    score_counts = value_counts(meeting_scores, "calibration_version")
    return {
        "calibration_versions": versions,
        "default_calibration_version": default_version,
        "default_detected_in_scores": default_version in score_counts,
        "has_calibration_v1_2006_2019": "calibration_v1_2006_2019" in versions,
        "details": details,
        "score_counts_by_calibration_version": score_counts,
        "scores_without_calibration_version": missing_count(meeting_scores, "calibration_version"),
    }


def formula_section(tables: dict[str, pd.DataFrame]) -> dict[str, Any]:
    scores = tables["v2_sentence_scores"]
    required = ["stance_score", "topic_weight", "confidence", "information_weight", "tone_level"]
    missing = [col for col in required if col not in scores]
    if scores.empty or missing:
        return {"status": "WARNING", "detail": f"Could not verify formula automatically. Missing columns: {missing}"}
    data = scores.dropna(subset=required).copy()
    if data.empty:
        return {"status": "WARNING", "detail": "No complete sentence score rows for formula check."}
    expected = data["stance_score"] * data["topic_weight"] * data["confidence"] * data["information_weight"]
    max_abs_diff = float((data["tone_level"] - expected).abs().max())
    novelty_evidence = False
    if "novelty_score" in data and data["novelty_score"].notna().any():
        novelty_expected = expected * data["novelty_score"]
        novelty_diff = float((data["tone_level"] - novelty_expected).abs().max())
        novelty_evidence = novelty_diff < max_abs_diff and novelty_diff < 1e-5
    if novelty_evidence:
        return {"status": "ERROR", "max_abs_diff": max_abs_diff, "detail": "tone_level appears closer to formula including novelty_score."}
    if max_abs_diff <= 1e-5:
        return {
            "status": "PASS",
            "max_abs_diff": max_abs_diff,
            "formula": "tone_level = stance_score * topic_weight * confidence * information_weight",
            "novelty_in_tone_level": False,
        }
    return {"status": "ERROR", "max_abs_diff": max_abs_diff, "detail": "tone_level does not match the documented V2 formula."}


def subindices_section(tables: dict[str, pd.DataFrame], settings: dict[str, Any]) -> dict[str, Any]:
    subindices = tables["v2_subindices"]
    sentence_scores = tables["v2_sentence_scores"]
    expected = {key: cfg["label"] for key, cfg in settings["subindices"].items()}
    expected["text_implied_reaction_function"] = settings["reaction_function"]["label"]
    rows: dict[str, Any] = {}
    for key, label in expected.items():
        group = subindices[(subindices.get("subindex", "") == key) | (subindices.get("label", "") == label)] if not subindices.empty else pd.DataFrame()
        document_count = subindex_document_count(key, sentence_scores, settings)
        values = pd.to_numeric(group.get("tone_raw", pd.Series(dtype=float)), errors="coerce")
        rows[label] = {
            "subindex": key,
            "meetings_with_value": int(values.notna().sum()),
            "documents_with_value": int(document_count),
            "null_values": int(values.isna().sum()),
            "min": safe_stat(values, "min"),
            "mean": safe_stat(values, "mean"),
            "median": safe_stat(values, "median"),
            "max": safe_stat(values, "max"),
        }
    return rows


def redline_section(tables: dict[str, pd.DataFrame]) -> dict[str, Any]:
    redline = tables["v2_redline"]
    documents = tables["v2_documents"]
    sentence_scores = tables["v2_sentence_scores"]
    expected_pairs = expected_redline_pairs(documents, sentence_scores)
    actual_pairs = int(redline.groupby(["meeting_id", "document_type"]).ngroups) if not redline.empty else 0
    return {
        "total_redlines": int(len(redline)),
        "redlines_by_document_type": value_counts(redline, "document_type"),
        "document_pairs_compared": actual_pairs,
        "expected_document_pairs": expected_pairs,
        "document_pair_coverage": safe_ratio(actual_pairs, expected_pairs),
        "maintained": int((redline.get("change_type", pd.Series(dtype=str)) == "maintained").sum()) if not redline.empty else 0,
        "removed": int((redline.get("change_type", pd.Series(dtype=str)) == "removed").sum()) if not redline.empty else 0,
        "added": int((redline.get("change_type", pd.Series(dtype=str)) == "added").sum()) if not redline.empty else 0,
        "tone_changed": int((redline.get("change_type", pd.Series(dtype=str)) == "tone_changed").sum()) if not redline.empty else 0,
    }


def evidence_section(tables: dict[str, pd.DataFrame]) -> dict[str, Any]:
    evidence = tables["v2_evidence"]
    meetings = tables["v2_meetings"]
    meetings_without = sorted(
        set(meetings.get("meeting_id", pd.Series(dtype=str)).dropna().astype(str))
        - set(evidence.get("meeting_id", pd.Series(dtype=str)).dropna().astype(str))
    )
    return {
        "hawkish_evidence": int((evidence.get("evidence_type", pd.Series(dtype=str)) == "hawkish").sum()) if not evidence.empty else 0,
        "dovish_evidence": int((evidence.get("evidence_type", pd.Series(dtype=str)) == "dovish").sum()) if not evidence.empty else 0,
        "evidence_by_topic": value_counts(evidence, "primary_topic"),
        "meetings_without_evidence": meetings_without,
    }


def labels_section(tables: dict[str, pd.DataFrame]) -> dict[str, Any]:
    labels = tables["v2_labels"]
    predictions = tables["v2_model_predictions"]
    audit = tables["v2_model_audit"]
    human = labels[labels.get("label_source", pd.Series(dtype=str)).astype(str).str.contains("human", case=False, na=False)] if not labels.empty else pd.DataFrame()
    llm = labels[labels.get("label_source", pd.Series(dtype=str)).astype(str).eq("llm_bootstrap")] if not labels.empty else pd.DataFrame()
    accepted = accepted_v2_labels(labels) if not labels.empty else pd.DataFrame()
    consensus = consensus_v2_labels(accepted) if not accepted.empty else pd.DataFrame()
    metrics = label_metrics(consensus, predictions)
    accepted_count = len(accepted)
    unique_accepted = len(consensus)
    if unique_accepted == 0:
        status = "needs_labels"
    elif unique_accepted < 30:
        status = "insufficient_labels"
    elif metrics.get("stance_accuracy") is not None and metrics.get("stance_accuracy", 0) >= 0.6:
        status = "passed"
    else:
        status = "ready" if metrics else "failed"
    if metrics:
        stance_agreement = human_agreement(accepted)
        metrics["human_stance_agreement"] = stance_agreement
        metrics["human_agreement"] = stance_agreement
    return {
        "human_labels": int(len(human)),
        "llm_bootstrap_labels": int(len(llm)),
        "accepted_labels": int(accepted_count),
        "unique_accepted_sentences": int(unique_accepted),
        "audit_status": status,
        "available_audit_rows": audit.to_dict("records") if not audit.empty else [],
        "metrics": metrics,
    }


def focus_v21_section(tables: dict[str, pd.DataFrame]) -> dict[str, Any]:
    vintages = tables["focus_vintages"]
    features = tables["focus_event_features"]
    if vintages.empty and features.empty:
        status = "not_started"
    elif vintages.empty or features.empty:
        status = "partial"
    else:
        delta_coverage = float(features[["delta_post_1", "delta_post_2"]].notna().any(axis=1).mean()) if not features.empty else 0.0
        status = "ready" if delta_coverage >= 0.40 else "partial"
    indicator_coverage = value_counts(vintages, "indicator")
    event_coverage = value_counts(features, "event_type")
    delta_coverage = float(features[["delta_post_1", "delta_post_2"]].notna().any(axis=1).mean()) if not features.empty else 0.0
    return {
        "focus_vintages_exist": not vintages.empty,
        "focus_event_features_exist": not features.empty,
        "focus_vintages": int(len(vintages)),
        "focus_event_features": int(len(features)),
        "indicator_coverage": indicator_coverage,
        "event_coverage": event_coverage,
        "delta_coverage": delta_coverage,
        "status": status,
    }


def market_section(tables: dict[str, pd.DataFrame]) -> dict[str, Any]:
    observations = tables["market_observations"]
    windows = tables["market_event_windows"]
    expectations = tables["decision_expectations"]
    public_audit = tables["public_market_source_audit"]
    decision_audit = tables["decision_expectation_source_audit"]
    public_coverage = tables["public_market_coverage"]
    if observations.empty and windows.empty:
        status = "no_market_data"
    elif observations.empty or windows.empty:
        status = "partial_market_data"
    else:
        status = "ready"
    if not expectations.empty and "is_proxy" in expectations:
        is_proxy = expectations["is_proxy"].fillna(False).astype(str).str.lower().isin({"true", "1", "yes", "sim"})
    else:
        is_proxy = pd.Series(False, index=expectations.index)
    return {
        "market_observations_exist": not observations.empty,
        "market_event_windows_exist": not windows.empty,
        "decision_expectations_exist": not expectations.empty,
        "public_market_source_audit_exist": not public_audit.empty,
        "decision_expectation_source_audit_exist": not decision_audit.empty,
        "public_market_coverage_exist": not public_coverage.empty,
        "market_observations": int(len(observations)),
        "market_event_windows": int(len(windows)),
        "decision_expectations": int(len(expectations)),
        "official_decision_expectations": int((~is_proxy).sum()) if not expectations.empty else 0,
        "proxy_decision_expectations": int(is_proxy.sum()) if not expectations.empty else 0,
        "ok_market_windows": int((windows.get("status", pd.Series(dtype=str)) == "ok").sum()) if not windows.empty else 0,
        "market_assets": value_counts(observations, "asset"),
        "public_source_status": value_counts(public_audit, "status"),
        "decision_source_status": value_counts(decision_audit, "status"),
        "public_coverage_rows": int(len(public_coverage)),
        "status": status,
    }


def v21_event_panel_section(tables: dict[str, pd.DataFrame]) -> dict[str, Any]:
    panel = tables["v21_event_panel"]
    if panel.empty:
        return {
            "exists": False,
            "rows": 0,
            "decision_surprise_available": 0,
            "focus_ready_rows": 0,
            "market_ready_rows": 0,
            "status": "not_started",
        }
    decision_official = int((panel.get("decision_surprise_status", pd.Series(dtype=str)) == "official").sum())
    decision_proxy = int((panel.get("decision_surprise_status", pd.Series(dtype=str)) == "proxy").sum())
    decision_available = decision_official + decision_proxy
    focus_ready = int((panel.get("focus_status", pd.Series(dtype=str)) == "ok").sum())
    market_ready = int((panel.get("market_status", pd.Series(dtype=str)) == "ready").sum())
    if decision_available or focus_ready or market_ready:
        status = "ready" if focus_ready and (market_ready or decision_available) else "partial"
    else:
        status = "partial"
    return {
        "exists": True,
        "rows": int(len(panel)),
        "decision_surprise_available": decision_available,
        "decision_surprise_official": decision_official,
        "decision_surprise_proxy": decision_proxy,
        "focus_ready_rows": focus_ready,
        "market_ready_rows": market_ready,
        "status": status,
    }


def semantic_section(tables: dict[str, pd.DataFrame]) -> dict[str, Any]:
    chunks = tables["semantic_chunks"]
    sentence_scores = tables["v2_sentence_scores"]
    if chunks.empty:
        status = "not_built"
    elif not sentence_scores.empty and "sentence_id" in sentence_scores and "sentence_id" in chunks:
        expected = set(sentence_scores["sentence_id"].dropna().astype(str))
        indexed = set(chunks["sentence_id"].dropna().astype(str))
        status = "built" if expected.issubset(indexed) else "unavailable"
    else:
        status = "built"
    return {"semantic_chunks_exist": not chunks.empty, "chunks": int(len(chunks)), "index_generated": status == "built", "status": status}


def idempotency_section(tables: dict[str, pd.DataFrame]) -> dict[str, Any]:
    redline_cols = ["meeting_id", "document_type", "current_sentence_id", "previous_sentence_id", "change_type"]
    return {
        "v2_documents": duplicate_count(tables["v2_documents"], ["meeting_id", "document_type", "source_hash"]),
        "v2_sentences": duplicate_count(tables["v2_sentences"], ["document_id", "sentence_order"]),
        "v2_sentence_scores": duplicate_count(tables["v2_sentence_scores"], ["sentence_id", "calibration_version", "model_version"]),
        "v2_document_scores": duplicate_count(tables["v2_document_scores"], ["document_id", "calibration_version", "model_version"]),
        "v2_subindices": duplicate_count(tables["v2_subindices"], ["meeting_id", "subindex"]),
        "v2_redline": duplicate_count(tables["v2_redline"], redline_cols),
        "focus_vintages": duplicate_count(
            tables.get("focus_vintages", pd.DataFrame()),
            ["focus_release_date", "indicator", "reference_year", "horizon", "statistic", "source", "query_signature"],
        ),
        "focus_event_features": duplicate_count(tables.get("focus_event_features", pd.DataFrame()), ["meeting_id", "event_type", "indicator", "horizon", "statistic"]),
        "decision_expectations": duplicate_count(tables.get("decision_expectations", pd.DataFrame()), ["meeting_id", "as_of_timestamp", "expected_selic_change_bps"]),
        "v21_event_panel": duplicate_count(tables.get("v21_event_panel", pd.DataFrame()), ["meeting_id"]),
    }


def implementation_status_section(
    overall_status: str,
    coverage: dict[str, Any],
    documents: dict[str, Any],
    sentences: dict[str, Any],
    scores: dict[str, Any],
    calibration: dict[str, Any],
    formula: dict[str, Any],
    subindices: dict[str, Any],
    redline: dict[str, Any],
    evidence: dict[str, Any],
    labels: dict[str, Any],
    focus_v21: dict[str, Any],
    market: dict[str, Any],
    v21_event_panel: dict[str, Any],
    semantic: dict[str, Any],
    idempotency: dict[str, Any],
) -> dict[str, Any]:
    subindex_values = [detail.get("meetings_with_value", 0) for detail in subindices.values()]
    idempotency_ok = all(int(count) == 0 for count in idempotency.values())
    micro = [
        status_item(
            "Backfill historico",
            "DONE" if coverage["total_meetings"] >= 200 and coverage["total_documents"] > 0 else "IN_PROGRESS",
            100 if coverage["total_meetings"] >= 200 and coverage["total_documents"] > 0 else 40,
            f"{coverage['total_meetings']} reunioes e {coverage['total_documents']} documentos.",
        ),
        status_item(
            "Metadados de origem",
            "DONE"
            if all(
                documents.get(key, 0) == 0
                for key in ["missing_source_hash", "missing_run_id", "missing_collected_at", "missing_source_url"]
            )
            else "WARN",
            100
            if all(
                documents.get(key, 0) == 0
                for key in ["missing_source_hash", "missing_run_id", "missing_collected_at", "missing_source_url"]
            )
            else 75,
            f"{documents.get('missing_source_url', 0)} documentos sem source_url.",
        ),
        status_item(
            "Sentencas e informatividade",
            "DONE" if sentences["total_sentences"] > 0 and sentences["informative_sentences"] > 0 else "FAIL",
            100 if sentences["total_sentences"] > 0 and sentences["informative_sentences"] > 0 else 0,
            f"{sentences['informative_sentences']} sentencas informativas de {sentences['total_sentences']}.",
        ),
        status_item(
            "Scoring deterministico",
            "DONE" if scores["sentence_scores"] > 0 and scores.get("informative_without_tone_level", 0) == 0 else "FAIL",
            100 if scores["sentence_scores"] > 0 and scores.get("informative_without_tone_level", 0) == 0 else 30,
            f"{scores['sentence_scores']} sentence_scores.",
        ),
        status_item(
            "Formula tone_level",
            "DONE" if formula["status"] == "PASS" else "FAIL" if formula["status"] == "ERROR" else "WARN",
            100 if formula["status"] == "PASS" else 0 if formula["status"] == "ERROR" else 70,
            str(formula.get("detail", "")),
        ),
        status_item(
            "Calibracao fixa",
            "DONE" if calibration["has_calibration_v1_2006_2019"] and calibration["scores_without_calibration_version"] == 0 else "WARN",
            100 if calibration["has_calibration_v1_2006_2019"] and calibration["scores_without_calibration_version"] == 0 else 60,
            f"Default: {calibration.get('default_calibration_version', '')}.",
        ),
        status_item(
            "Subindices",
            "DONE" if subindex_values and min(subindex_values) > 0 else "WARN",
            100 if subindex_values and min(subindex_values) > 0 else 60,
            f"{len(subindex_values)} subindices avaliados.",
        ),
        status_item(
            "Redline textual",
            "DONE" if redline["total_redlines"] > 0 and redline.get("document_pair_coverage", 0) >= 0.9 else "WARN",
            int(round(100 * float(redline.get("document_pair_coverage", 0)))) if redline.get("expected_document_pairs", 0) else 0,
            f"{redline['document_pairs_compared']}/{redline['expected_document_pairs']} pares comparados.",
        ),
        status_item(
            "Evidencias textuais",
            "DONE" if evidence["hawkish_evidence"] + evidence["dovish_evidence"] > 0 else "WARN",
            100 if evidence["hawkish_evidence"] + evidence["dovish_evidence"] > 0 else 40,
            f"{evidence['hawkish_evidence']} hawkish e {evidence['dovish_evidence']} dovish.",
        ),
        status_item(
            "Idempotencia",
            "DONE" if idempotency_ok else "FAIL",
            100 if idempotency_ok else 0,
            "Sem duplicatas logicas." if idempotency_ok else "Duplicatas logicas detectadas.",
        ),
        status_item(
            "Labels humanos",
            label_visual_status(labels),
            label_visual_progress(labels),
            label_visual_detail(labels),
        ),
        status_item(
            "Mercado opcional",
            "OPTIONAL_MISSING" if market["status"] == "no_market_data" else "DONE" if market["status"] == "ready" else "WARN",
            0 if market["status"] == "no_market_data" else 100 if market["status"] == "ready" else 50,
            f"Status: {market['status']}.",
        ),
        status_item(
            "Focus V2.1 vintage-safe",
            "DONE" if focus_v21["status"] == "ready" else "WARN" if focus_v21["status"] == "partial" else "PENDING",
            100 if focus_v21["status"] == "ready" else 55 if focus_v21["status"] == "partial" else 0,
            f"{focus_v21['focus_vintages']} vintages e {focus_v21['focus_event_features']} features.",
        ),
        status_item(
            "Painel V2.1 de eventos",
            "DONE" if v21_event_panel["status"] == "ready" else "WARN" if v21_event_panel["status"] == "partial" else "PENDING",
            100 if v21_event_panel["status"] == "ready" else 55 if v21_event_panel["status"] == "partial" else 0,
            f"{v21_event_panel['rows']} linhas; decisao disponivel={v21_event_panel['decision_surprise_available']}.",
        ),
        status_item(
            "Semantica local",
            "DONE" if semantic["status"] == "built" else "OPTIONAL_MISSING",
            100 if semantic["status"] == "built" else 0,
            f"Status: {semantic['status']}; chunks: {semantic['chunks']}.",
        ),
    ]
    macro = [
        status_item(
            "V2.0 Indice defensavel",
            phase_status([micro[0], micro[1], micro[2], micro[3], micro[4], micro[5], micro[6], micro[9]]),
            phase_progress([micro[0], micro[1], micro[2], micro[3], micro[4], micro[5], micro[6], micro[9]]),
            "Core textual versionado, calibrado e auditavel.",
        ),
        status_item(
            "V2.0.1 Redline e relatorio",
            phase_status([micro[7], micro[8]]),
            phase_progress([micro[7], micro[8]]),
            "Mudanca textual e evidencias por reuniao.",
        ),
        status_item(
            "V2.0.2 Validacao e modelo",
            label_visual_status(labels),
            label_visual_progress(labels),
            label_visual_detail(labels),
        ),
        status_item(
            "V2.1 Mercado e Focus expandido",
            phase_status([micro[11], micro[12], micro[13]]),
            phase_progress([micro[11], micro[12], micro[13]]),
            "Focus vintage-safe, mercado opcional e painel de eventos.",
        ),
        status_item(
            "V2.2 Produto profissional gratuito",
            "IN_PROGRESS" if semantic["status"] == "built" else "PENDING",
            45 if semantic["status"] == "built" else 25,
            "Dashboard core e busca local existem; produto final ainda nao esta fechado.",
        ),
    ]
    return {
        "overall": overall_status,
        "macro": macro,
        "micro": micro,
    }


def status_item(name: str, status: str, progress: int, detail: str) -> dict[str, Any]:
    return {"name": name, "status": status, "progress": int(max(0, min(100, progress))), "detail": detail}


def phase_status(items: list[dict[str, Any]]) -> str:
    statuses = {str(item["status"]) for item in items}
    if "FAIL" in statuses:
        return "FAIL"
    if statuses <= {"DONE"}:
        return "DONE"
    if statuses & {"WARN"}:
        return "READY_WITH_WARNINGS"
    return "IN_PROGRESS"


def phase_progress(items: list[dict[str, Any]]) -> int:
    if not items:
        return 0
    return int(round(sum(int(item["progress"]) for item in items) / len(items)))


def label_visual_status(labels: dict[str, Any]) -> str:
    audit_status = str(labels.get("audit_status", "needs_labels"))
    if audit_status in {"needs_labels", "insufficient_labels"}:
        return "PENDING"
    if audit_status == "passed":
        return "DONE"
    if audit_status == "ready":
        return "READY_WITH_WARNINGS"
    return "FAIL"


def label_visual_progress(labels: dict[str, Any]) -> int:
    audit_status = str(labels.get("audit_status", "needs_labels"))
    if audit_status == "needs_labels":
        return 25
    if audit_status == "insufficient_labels":
        return 50
    if audit_status == "ready":
        return 75
    if audit_status == "passed":
        return 100
    return 0


def label_visual_detail(labels: dict[str, Any]) -> str:
    metrics = labels.get("metrics", {}) or {}
    stance = metrics.get("stance_accuracy")
    stance_text = f"; stance_accuracy={float(stance):.3f}" if stance is not None else ""
    return (
        f"Status: {labels.get('audit_status')}; "
        f"accepted={labels.get('accepted_labels', 0)}; "
        f"unique={labels.get('unique_accepted_sentences', 0)}"
        f"{stance_text}."
    )


def collect_findings(warnings: list[str], errors: list[str], *sections: dict[str, Any]) -> None:
    (
        coverage,
        documents,
        sentences,
        scores,
        calibration,
        formula,
        subindices,
        redline,
        evidence,
        labels,
        focus_v21,
        market,
        v21_event_panel,
        semantic,
        idempotency,
    ) = sections
    if coverage["total_meetings"] == 0 or coverage["total_documents"] == 0:
        errors.append("V2 documents or meetings are missing.")
    if coverage["duplicate_documents_by_meeting_type"] > 0:
        errors.append("Duplicate documents by meeting_id + document_type compromise document coverage.")
    if coverage["meetings_without_minutes"]:
        warnings.append(f"{len(coverage['meetings_without_minutes'])} meetings do not have minutes.")
    if coverage["meetings_without_statement"]:
        warnings.append(f"{len(coverage['meetings_without_statement'])} meetings do not have statements.")
    if documents["raw_text_null"] > 0 and documents["raw_text_null"] / max(1, coverage["total_documents"]) > 0.05:
        errors.append("More than 5% of documents have null raw_text.")
    elif documents["raw_text_null"] > 0 or documents["raw_text_too_short"] > 0:
        warnings.append("Some documents have null or very short raw_text.")
    for key in ["missing_source_hash", "missing_run_id", "missing_collected_at"]:
        if documents[key] > 0:
            errors.append(f"{documents[key]} documents are missing {key.replace('missing_', '')}.")
    if documents["missing_source_url"] > 0:
        warnings.append("Some documents do not have source URL metadata.")
    if sentences["total_sentences"] == 0:
        errors.append("No V2 sentences were generated.")
    if sentences["documents_without_informative_sentence"]:
        warnings.append(f"{len(sentences['documents_without_informative_sentence'])} documents have no informative sentence.")
    if sentences["duplicate_sentence_order_by_document"] > 0:
        errors.append("Duplicate sentences by document_id + sentence_order detected.")
    if sentences["sentences_without_valid_document_id"] > 0:
        errors.append("Some V2 sentences reference invalid document_id.")
    if scores["sentence_scores"] == 0:
        errors.append("No V2 sentence scores are available.")
    for key in [
        "informative_without_tone_level",
        "informative_without_stance_score",
        "informative_without_topic_weight",
        "informative_without_confidence",
        "informative_without_information_weight",
    ]:
        if scores[key] > 0:
            errors.append(f"{scores[key]} informative sentences are missing {key.replace('informative_without_', '')}.")
    if scores["documents_without_aggregate_score"]:
        warnings.append("Some V2 documents do not have aggregate document score, usually because no informative sentence was available.")
    if scores["meetings_without_meeting_score"]:
        errors.append("Some V2 meetings do not have meeting score.")
    if scores.get("meetings_with_expected_low_information_null_tone", 0) > 0:
        warnings.append(
            f"{scores['meetings_with_expected_low_information_null_tone']} V2 meetings have no final tone because they are explicitly low-information."
        )
    if scores.get("meetings_with_unexpected_null_final_tone", scores["meetings_with_null_final_tone"]) > 0:
        errors.append("Some V2 meeting scores have null final tone.")
    if not calibration["has_calibration_v1_2006_2019"]:
        errors.append("Default calibration_v1_2006_2019 is missing.")
    if calibration["scores_without_calibration_version"] > 0:
        warnings.append("Some scores are missing calibration_version.")
    if formula["status"] == "ERROR":
        errors.append(str(formula.get("detail", "V2 tone formula check failed.")))
    elif formula["status"] == "WARNING":
        warnings.append(str(formula.get("detail", "V2 tone formula could not be verified.")))
    for label, detail in subindices.items():
        if detail["meetings_with_value"] == 0:
            warnings.append(f"Subindex {label} has no meeting values.")
    if redline["total_redlines"] == 0:
        warnings.append("V2 redline is not available.")
    elif redline.get("document_pair_coverage", 1.0) < 0.9:
        warnings.append(
            f"V2 redline covers {redline['document_pairs_compared']}/{redline['expected_document_pairs']} expected document pairs."
        )
    if evidence["hawkish_evidence"] == 0 and evidence["dovish_evidence"] == 0:
        warnings.append("No V2 evidence sentences are available.")
    if labels["audit_status"] in {"needs_labels", "insufficient_labels"}:
        warnings.append("Human labels are missing or insufficient for formal model validation.")
    elif labels["audit_status"] == "ready":
        warnings.append("Human validation is available, but baseline stance accuracy is below the pass threshold.")
    elif labels["audit_status"] == "failed":
        errors.append("Human validation failed or could not compute usable audit metrics.")
    if market["status"] == "no_market_data":
        warnings.append("Optional market data is not available.")
    if focus_v21["status"] == "not_started":
        warnings.append("Focus V2.1 vintage-safe layer is not built.")
    elif focus_v21["status"] == "partial":
        warnings.append("Focus V2.1 coverage is partial or below ready threshold.")
    if v21_event_panel["status"] == "not_started":
        warnings.append("V2.1 event panel is not built.")
    if semantic["status"] == "not_built":
        warnings.append("Optional semantic index is not built.")
    if semantic["status"] == "unavailable":
        warnings.append("Optional semantic index is stale or incomplete for current V2 scores.")
    for table, count in idempotency.items():
        if count > 0:
            errors.append(f"Idempotency duplicate check failed for {table}: {count} duplicates.")


def build_v2_label_review_sample(
    sentence_scores: pd.DataFrame,
    subindices: pd.DataFrame,
    n: int = 300,
    exclude_sentence_ids: set[str] | None = None,
) -> pd.DataFrame:
    columns = [
        "meeting_id",
        "document_type",
        "document_id",
        "sentence_id",
        "sentence_index",
        "sentence_text",
        "baseline_topic",
        "baseline_stance",
        "baseline_confidence",
        "baseline_is_informative",
        "tone_level",
        "subindex_name",
        "human_topic",
        "human_stance",
        "human_is_informative",
        "human_notes",
        "reviewer_id",
        "accepted",
    ]
    if sentence_scores.empty:
        return pd.DataFrame(columns=columns)
    frame = sentence_scores.copy()
    if exclude_sentence_ids:
        frame = frame[~frame["sentence_id"].astype(str).isin(exclude_sentence_ids)].copy()
    if frame.empty:
        return pd.DataFrame(columns=columns)
    frame["tone_abs_bucket"] = pd.qcut(frame["tone_level"].abs().rank(method="first"), q=min(4, len(frame)), labels=False, duplicates="drop")
    frame["time_bucket"] = pd.qcut(frame["nro_reuniao"].rank(method="first"), q=min(4, len(frame)), labels=False, duplicates="drop")
    frame["subindex_name"] = frame["primary_topic"].map(topic_to_subindex_map(load_v2_settings())).fillna("")
    strata = ["time_bucket", "document_type", "primary_topic", "stance", "is_informative", "tone_abs_bucket"]
    per_group = max(1, int(np.ceil(n / max(1, frame.groupby(strata, dropna=False).ngroups))))
    sample = (
        frame.sort_values(strata + ["sentence_id"])
        .groupby(strata, dropna=False, group_keys=False)
        .head(per_group)
        .head(n)
        .copy()
    )
    if len(sample) < min(n, len(frame)):
        remaining = frame[~frame["sentence_id"].isin(sample["sentence_id"])].copy()
        remaining = remaining.sort_values(["nro_reuniao", "document_type", "primary_topic", "stance", "sentence_id"])
        sample = pd.concat([sample, remaining.head(n - len(sample))], ignore_index=True)
    output = pd.DataFrame(
        {
            "meeting_id": sample["meeting_id"],
            "document_type": sample["document_type"],
            "document_id": sample["document_id"],
            "sentence_id": sample["sentence_id"],
            "sentence_index": sample.get("sentence_order", ""),
            "sentence_text": sample["text"],
            "baseline_topic": sample["primary_topic"],
            "baseline_stance": sample["stance"],
            "baseline_confidence": sample["confidence"],
            "baseline_is_informative": sample["is_informative"],
            "tone_level": sample["tone_level"],
            "subindex_name": sample["subindex_name"],
            "human_topic": "",
            "human_stance": "",
            "human_is_informative": "",
            "human_notes": "",
            "reviewer_id": "",
            "accepted": "",
        }
    )
    return output[columns]


def label_codebook_path(sample_path: Path) -> Path:
    return sample_path.with_name(f"{sample_path.stem}_codebook.md")


def build_v2_label_codebook(topics: dict[str, Any], v2_settings: dict[str, Any], sample_path: Path) -> str:
    project = v2_settings.get("project", {})
    topic_rows = []
    for topic, details in sorted(topics.items()):
        keywords = ", ".join(str(item) for item in details.get("keywords", [])[:8])
        topic_rows.append(f"| `{topic}` | {details.get('weight', '')} | {keywords} |")
    subindex_rows = []
    for subindex, details in sorted(v2_settings.get("subindices", {}).items()):
        topic_list = ", ".join(f"`{topic}`" for topic in details.get("topics", []))
        subindex_rows.append(f"| `{subindex}` | {details.get('label', '')} | {topic_list} |")
    reaction_weights = v2_settings.get("reaction_function", {}).get("weights", {})
    reaction_rows = [f"| `{name}` | {weight} |" for name, weight in sorted(reaction_weights.items())]
    generated_at = utc_now_naive().isoformat()
    return "\n".join(
        [
            "# COPOM Watch V2 Human Label Review Codebook",
            "",
            f"Generated at: `{generated_at}`",
            f"Review sample: `{sample_path}`",
            "",
            "## Versions",
            "",
            f"- model_version: `{project.get('default_model_version', '')}`",
            f"- prompt_version: `{project.get('prompt_version', '')}`",
            f"- taxonomy_version: `{project.get('taxonomy_version', '')}`",
            f"- lexicon_version: `{project.get('lexicon_version', '')}`",
            f"- calibration_version: `{v2_settings.get('scoring', {}).get('default_calibration', '')}`",
            "",
            "## Review Rules",
            "",
            "- Fill only human fields: `human_topic`, `human_stance`, `human_is_informative`, `human_notes`, `reviewer_id`, `accepted`.",
            "- Do not edit identifiers, baseline fields, `tone_level`, or sentence text.",
            "- Use `accepted=true` only after a human reviewer has checked the row.",
            "- Leave `accepted` empty for rows that still need review.",
            "- `llm_bootstrap` labels are not formal ground truth and must not be used as accepted human labels.",
            "- When a sentence spans multiple topics, choose the main economic topic for `human_topic` and explain secondary topics in `human_notes`.",
            "- If the sentence is institutional, procedural, table-only, or too generic for economic signal, set `human_is_informative=false`.",
            "",
            "## Allowed Values",
            "",
            "- `human_stance`: `hawkish`, `dovish`, `neutral`.",
            "- `human_is_informative`: `true`, `false`.",
            "- `accepted`: `true`, `false`, or blank.",
            "- `reviewer_id`: stable human reviewer identifier, for example `reviewer_a`.",
            "",
            "## Topic Taxonomy",
            "",
            "| topic | weight | keyword examples |",
            "|---|---:|---|",
            *topic_rows,
            "",
            "## Subindices",
            "",
            "| subindex | label | topics |",
            "|---|---|---|",
            *subindex_rows,
            "",
            "## Text-Implied Reaction Function Weights",
            "",
            "| subindex | weight |",
            "|---|---:|",
            *reaction_rows,
            "",
            "## Import",
            "",
            "After review, import the completed CSV with:",
            "",
            "```powershell",
            f"copom-watch v2 import-labels --path {sample_path} --label-source human",
            "copom-watch v2 audit",
            "copom-watch v2 health-check",
            "```",
            "",
            "Formal validation remains unavailable until there are enough accepted human-reviewed rows.",
            "",
        ]
    )


def render_acceptance_report_html(report: dict[str, Any], output_dir: Path) -> str:
    status = str(report["status"]).upper()
    status_color = {"PASS": "#0f766e", "WARNING": "#b45309", "FAIL": "#b91c1c"}.get(status, "#374151")
    existing_reports = sorted(path for path in output_dir.glob("*.html") if path.name != "acceptance_report.html")
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>COPOM Watch V2 Acceptance Report</title>",
        "<style>body{font-family:Arial,sans-serif;margin:32px;line-height:1.45;color:#111827}table{border-collapse:collapse;width:100%;margin:12px 0}td,th{border:1px solid #ddd;padding:6px;vertical-align:top}th{background:#f3f4f6}.badge{display:inline-block;padding:6px 10px;border-radius:4px;color:white;font-weight:bold}.muted{color:#6b7280}code{background:#f3f4f6;padding:2px 4px}.status{font-weight:bold}.status-DONE{color:#0f766e}.status-READY_WITH_WARNINGS,.status-WARN{color:#b45309}.status-PENDING,.status-IN_PROGRESS,.status-OPTIONAL_MISSING{color:#374151}.status-FAIL{color:#b91c1c}.bar{background:#e5e7eb;border-radius:4px;height:10px;min-width:120px}.fill{background:#0f766e;border-radius:4px;height:10px}.fill-warn{background:#b45309}.fill-fail{background:#b91c1c}.progress-cell{display:flex;gap:8px;align-items:center}</style>",
        "</head><body>",
        "<h1>COPOM Watch V2 Acceptance Report</h1>",
        f"<p><span class='badge' style='background:{status_color}'>{status}</span></p>",
        "<h2>Resumo executivo</h2>",
        f"<p>Gerado em <code>{report['generated_at']}</code> para <code>{report['database_path']}</code>.</p>",
        f"<p>Warnings: <b>{len(report['warnings'])}</b>. Errors: <b>{len(report['errors'])}</b>.</p>",
        "<h2>Status de implementacao V2</h2>",
        "<h3>Macro roadmap</h3>",
        status_records_table(report["implementation_status"]["macro"]),
        "<h3>Micro componentes</h3>",
        status_records_table(report["implementation_status"]["micro"]),
        "<h3>Warnings</h3>",
        html_list(report["warnings"]),
        "<h3>Errors</h3>",
        html_list(report["errors"]),
        "<h2>Cobertura</h2>",
        dict_table(report["coverage"]),
        "<h2>Documentos</h2>",
        dict_table(report["documents"]),
        "<h2>Sentencas</h2>",
        dict_table(shorten_dict(report["sentences"], ["sentences_by_document"])),
        "<h2>Scores</h2>",
        dict_table(shorten_dict(report["scores"], ["top_10_hawkish", "top_10_dovish"])),
        "<h3>Top 10 hawkish</h3>",
        records_table(report["scores"]["top_10_hawkish"]),
        "<h3>Top 10 dovish</h3>",
        records_table(report["scores"]["top_10_dovish"]),
        "<h2>Calibracao</h2>",
        dict_table(shorten_dict(report["calibration"], ["details"])),
        "<h2>Formula do score</h2>",
        dict_table(report["formula_check"]),
        "<h2>Subindices</h2>",
        records_table([{"label": key, **value} for key, value in report["subindices"].items()]),
        "<h2>Redline</h2>",
        dict_table(report["redline"]),
        "<h2>Evidencias</h2>",
        dict_table(report["evidence"]),
        "<h2>Labels e auditoria</h2>",
        dict_table(report["labels"]),
        "<h2>Focus V2.1</h2>",
        dict_table(report["focus_v21"]),
        "<h2>Mercado</h2>",
        dict_table(report["market"]),
        "<h2>Painel V2.1 de eventos</h2>",
        dict_table(report["v21_event_panel"]),
        "<h2>Semantica</h2>",
        dict_table(report["semantic"]),
        "<h2>Idempotencia</h2>",
        dict_table(report["idempotency"]),
        "<h2>Relatorios V2 detectados</h2>",
        html_list([str(path) for path in existing_reports]),
        "</body></html>",
    ]
    return "\n".join(parts)


def duplicate_count(frame: pd.DataFrame, columns: list[str]) -> int:
    if frame.empty or any(col not in frame for col in columns):
        return 0
    return int(frame.duplicated(columns, keep=False).sum())


def missing_count(frame: pd.DataFrame, column: str) -> int:
    if frame.empty or column not in frame:
        return 0
    series = frame[column]
    return int(series.isna().sum() + (series.fillna("").astype(str).str.strip() == "").sum())


def value_counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if frame.empty or column not in frame:
        return {}
    return {str(key): int(value) for key, value in frame[column].fillna("missing").value_counts().to_dict().items()}


def distribution(series: pd.Series) -> dict[str, float | None]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {key: None for key in ["min", "p5", "p25", "mean", "median", "p75", "p95", "max"]}
    return {
        "min": float(values.min()),
        "p5": float(values.quantile(0.05)),
        "p25": float(values.quantile(0.25)),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "p75": float(values.quantile(0.75)),
        "p95": float(values.quantile(0.95)),
        "max": float(values.max()),
    }


def top_meetings(frame: pd.DataFrame, column: str, ascending: bool) -> list[dict[str, Any]]:
    if frame.empty or column not in frame:
        return []
    cols = [col for col in ["meeting_id", "nro_reuniao", "data_referencia", column, "classification_v2"] if col in frame]
    return frame.dropna(subset=[column]).sort_values(column, ascending=ascending)[cols].head(10).to_dict("records")


def expected_redline_pairs(documents: pd.DataFrame, sentence_scores: pd.DataFrame) -> int:
    if documents.empty or "document_type" not in documents or sentence_scores.empty or "document_id" not in sentence_scores:
        return 0
    docs_with_sentences = set(sentence_scores["document_id"].dropna().astype(str))
    expected = 0
    for _, group in documents.sort_values(["document_type", "nro_reuniao"]).groupby("document_type"):
        previous_has_sentences = False
        seen_previous = False
        for _, doc in group.iterrows():
            current_has_sentences = str(doc["document_id"]) in docs_with_sentences
            if seen_previous and (previous_has_sentences or current_has_sentences):
                expected += 1
            seen_previous = True
            previous_has_sentences = current_has_sentences
    return expected


def label_metrics(labels: pd.DataFrame, predictions: pd.DataFrame) -> dict[str, Any]:
    if labels.empty or predictions.empty:
        return {}
    merged = labels.merge(predictions, on="sentence_id", how="inner")
    if merged.empty:
        return {}
    metrics: dict[str, Any] = {
        "stance_accuracy": accuracy(merged, "stance_label", "predicted_stance"),
        "topic_accuracy": accuracy(merged, "topic_label", "predicted_topic"),
        "informativeness_accuracy": accuracy(merged, "is_informative_label", "predicted_is_informative"),
        "confusion_matrix": confusion_matrix_records(merged, "stance_label", "predicted_stance"),
        "f1_macro": macro_f1(merged["stance_label"], merged["predicted_stance"]),
    }
    return metrics


def accuracy(frame: pd.DataFrame, truth: str, pred: str) -> float | None:
    if truth not in frame or pred not in frame or frame.empty:
        return None
    return float((frame[truth].astype(str) == frame[pred].astype(str)).mean())


def confusion_matrix_records(frame: pd.DataFrame, truth: str, pred: str) -> list[dict[str, Any]]:
    if truth not in frame or pred not in frame:
        return []
    return (
        frame.groupby([truth, pred])
        .size()
        .reset_index(name="count")
        .rename(columns={truth: "actual", pred: "predicted"})
        .to_dict("records")
    )


def macro_f1(y_true: pd.Series, y_pred: pd.Series) -> float | None:
    labels = sorted(set(y_true.dropna().astype(str)) | set(y_pred.dropna().astype(str)))
    if not labels:
        return None
    scores = []
    true = y_true.astype(str)
    pred = y_pred.astype(str)
    for label in labels:
        tp = int(((true == label) & (pred == label)).sum())
        fp = int(((true != label) & (pred == label)).sum())
        fn = int(((true == label) & (pred != label)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return float(np.mean(scores))


def human_agreement(labels: pd.DataFrame) -> float | None:
    if labels.empty or "annotator_id" not in labels:
        return None
    duplicated = labels[labels.duplicated("sentence_id", keep=False)].copy()
    if duplicated.empty:
        return None
    agreements = []
    for _, group in duplicated.groupby("sentence_id"):
        if group["annotator_id"].nunique() < 2:
            continue
        agreements.append(float(group["stance_label"].nunique() == 1))
    return float(np.mean(agreements)) if agreements else None


def subindex_document_count(subindex: str, sentence_scores: pd.DataFrame, settings: dict[str, Any]) -> int:
    if sentence_scores.empty:
        return 0
    if subindex == "text_implied_reaction_function":
        return int(sentence_scores[sentence_scores["is_informative"].fillna(False).astype(bool)]["document_id"].nunique())
    topics = set(settings["subindices"].get(subindex, {}).get("topics", []))
    if not topics:
        return 0
    mask = sentence_scores["topics"].fillna("").map(lambda value: bool(topics.intersection(str(value).split("|"))))
    return int(sentence_scores[mask]["document_id"].nunique())


def topic_to_subindex_map(settings: dict[str, Any]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for subindex, config in settings["subindices"].items():
        for topic in config.get("topics", []):
            mapping[topic] = subindex
    return mapping


def safe_ratio(num: int, den: int) -> float:
    return float(num / den) if den else 0.0


def safe_min(values: set[str]) -> str | None:
    return min(values) if values else None


def safe_max(values: set[str]) -> str | None:
    return max(values) if values else None


def meeting_id_for_number(frame: pd.DataFrame, meeting_number: float | None) -> str | None:
    if frame.empty or meeting_number is None or "nro_reuniao" not in frame or "meeting_id" not in frame:
        return None
    numbers = pd.to_numeric(frame["nro_reuniao"], errors="coerce")
    matches = frame[numbers == meeting_number]
    if matches.empty:
        return None
    value = matches["meeting_id"].dropna().astype(str)
    return value.iloc[0] if not value.empty else None


def numeric_min(frame: pd.DataFrame, column: str) -> float | None:
    if frame.empty or column not in frame:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.min()) if not values.empty else None


def numeric_max(frame: pd.DataFrame, column: str) -> float | None:
    if frame.empty or column not in frame:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.max()) if not values.empty else None


def safe_stat(series: pd.Series, stat: str) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return None
    if stat == "min":
        return float(values.min())
    if stat == "mean":
        return float(values.mean())
    if stat == "median":
        return float(values.median())
    if stat == "max":
        return float(values.max())
    return None


def to_int_dict(series: pd.Series) -> dict[str, int]:
    return {str(key): int(value) for key, value in series.to_dict().items()}


def shorten_dict(data: dict[str, Any], omitted_keys: list[str]) -> dict[str, Any]:
    output = dict(data)
    for key in omitted_keys:
        if key in output:
            value = output[key]
            if isinstance(value, list):
                output[key] = f"{len(value)} records omitted from summary"
            elif isinstance(value, dict):
                output[key] = f"{len(value)} keys omitted from summary"
    return output


def html_list(items: list[Any]) -> str:
    if not items:
        return "<p class='muted'>Nenhum item.</p>"
    return "<ul>" + "".join(f"<li>{escape_html(str(item))}</li>" for item in items) + "</ul>"


def dict_table(data: dict[str, Any]) -> str:
    rows = []
    for key, value in data.items():
        rows.append(f"<tr><th>{escape_html(str(key))}</th><td>{escape_html(json.dumps(value, ensure_ascii=False, default=_json_default) if isinstance(value, (dict, list)) else str(value))}</td></tr>")
    return "<table>" + "".join(rows) + "</table>"


def records_table(records: list[dict[str, Any]]) -> str:
    if not records:
        return "<p class='muted'>Sem dados.</p>"
    frame = pd.DataFrame(records)
    return frame.to_html(index=False, escape=True)


def status_records_table(records: list[dict[str, Any]]) -> str:
    if not records:
        return "<p class='muted'>Sem dados.</p>"
    rows = ["<tr><th>Item</th><th>Status</th><th>Progresso</th><th>Detalhe</th></tr>"]
    for record in records:
        name = escape_html(str(record.get("name", "")))
        status = escape_html(str(record.get("status", "")))
        progress = int(record.get("progress", 0))
        detail = escape_html(str(record.get("detail", "")))
        fill_class = "fill-fail" if status == "FAIL" else "fill-warn" if status in {"WARN", "READY_WITH_WARNINGS"} else ""
        rows.append(
            "<tr>"
            f"<td>{name}</td>"
            f"<td><span class='status status-{status}'>{status}</span></td>"
            "<td>"
            "<div class='progress-cell'>"
            f"<div class='bar'><div class='fill {fill_class}' style='width:{progress}%'></div></div>"
            f"<span>{progress}%</span>"
            "</div>"
            "</td>"
            f"<td>{detail}</td>"
            "</tr>"
        )
    return "<table>" + "".join(rows) + "</table>"


def escape_html(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _json_default(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if pd.isna(value):
        return None
    return str(value)

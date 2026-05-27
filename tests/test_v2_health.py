from pathlib import Path

import duckdb
import pandas as pd

from copom_tone_index.config import ProjectPaths, load_v2_settings
from copom_tone_index.nlp import classify_sentences_baseline
from copom_tone_index.pipeline import validate_existing_outputs
from copom_tone_index.storage import write_tables
from copom_tone_index.v2 import (
    accepted_v2_labels,
    aggregate_v2_scores,
    apply_v2_calibration,
    build_baseline_benchmark_tables,
    build_v2_calibration,
    build_v2_evidence,
    build_v2_model_audit,
    build_v2_model_audit_details,
    build_v2_model_audit_error_analysis,
    build_v2_model_audit_error_classification,
    build_v2_model_audit_error_summary,
    build_v2_model_audit_report_html,
    build_remaining_error_review,
    build_v2_supervised_model_artifacts,
    build_v2_reviewer_disagreements,
    build_v2_validation_disagreement_report_html,
    build_v2_model_predictions,
    build_v2_redline,
    build_v2_sentences,
    build_v2_subindices,
    empty_v2_labels,
    evaluate_benchmark_gates,
    normalize_v2_labels,
    prepare_v2_backfill,
    score_v2_sentences,
    v2_audit_command,
    v2_benchmark_baseline_command,
    v2_freeze_release_command,
    v2_review_remaining_errors_command,
    v2_train_supervised_command,
)
from copom_tone_index.v2_health import (
    build_v2_label_codebook,
    build_v2_label_review_sample,
    build_v2_acceptance_report,
    export_label_sample_command,
    formula_section,
    idempotency_section,
    label_visual_status,
    labels_section,
    redline_section,
    v2_health_check_command,
)
from copom_tone_index.v21 import v21_freeze_release_command, v21_release_official_proxy_contamination
from copom_tone_index.v22 import package_public_data_command, v22_freeze_release_command, v22_health_command
from copom_tone_index.semantic import build_semantic_chunks


TOPICS = {
    "inflation_expectations": {"weight": 1.3, "keywords": ["expectativas", "desancoradas"]},
    "inflation_current": {"weight": 1.1, "keywords": ["inflacao", "servicos"]},
    "activity_growth": {"weight": 0.95, "keywords": ["atividade", "demanda"]},
    "fiscal_risk": {"weight": 1.15, "keywords": ["fiscal"]},
    "risk_balance": {"weight": 1.2, "keywords": ["riscos"]},
    "forward_guidance": {"weight": 1.4, "keywords": ["proximos passos"]},
    "institutional": {"weight": 0.2, "keywords": ["votaram"]},
}

LEXICON = {
    "hawkish": [{"term": "desancoradas", "weight": 2.0}, {"term": "cautela adicional", "weight": 1.0}],
    "dovish": [{"term": "desaceleracao", "weight": 1.0}, {"term": "ociosidade", "weight": 1.0}],
}


def _fixture_documents() -> pd.DataFrame:
    rows = []
    for nro in range(1, 5):
        rows.append(
            {
                "document_id": f"copom_{nro}_comunicado",
                "meeting_id": f"copom_{nro}",
                "nro_reuniao": nro,
                "document_type": "comunicado",
                "publication_date": pd.Timestamp(f"2010-0{nro}-01"),
                "title": f"Comunicado {nro}",
                "url": f"https://example.com/comunicado/{nro}",
                "raw_text": "As expectativas desancoradas exigem cautela adicional. A inflacao de servicos segue pressionada.",
                "source": "fixture",
            }
        )
        rows.append(
            {
                "document_id": f"copom_{nro}_ata",
                "meeting_id": f"copom_{nro}",
                "nro_reuniao": nro,
                "document_type": "ata",
                "publication_date": pd.Timestamp(f"2010-0{nro}-05"),
                "title": f"Ata {nro}",
                "url": f"https://example.com/ata/{nro}",
                "raw_text": "A atividade mostra desaceleracao e ociosidade. Os riscos fiscais seguem monitorados.",
                "source": "fixture",
            }
        )
    return pd.DataFrame(rows)


def _fixture_meetings() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"meeting_id": f"copom_{nro}", "nro_reuniao": nro, "data_referencia": pd.Timestamp(f"2010-0{nro}-01")}
            for nro in range(1, 5)
        ]
    )


def _build_v2_tables() -> dict[str, pd.DataFrame]:
    v2_settings = load_v2_settings()
    meetings, documents = prepare_v2_backfill(
        _fixture_meetings(),
        _fixture_documents(),
        "run_fixture",
        {"backfill": {"min_minutes_meeting": 1, "min_statement_meeting": 1}},
    )
    sentences = build_v2_sentences(documents, "run_fixture")
    sentence_scores = score_v2_sentences(sentences, TOPICS, LEXICON, v2_settings, "run_fixture")
    document_scores, meeting_scores_raw = aggregate_v2_scores(meetings, documents, sentence_scores, v2_settings, "run_fixture")
    calibration = build_v2_calibration(meeting_scores_raw, v2_settings, "run_fixture")
    meeting_scores = apply_v2_calibration(meeting_scores_raw, calibration, v2_settings)
    subindices = build_v2_subindices(sentence_scores, meeting_scores, v2_settings, "run_fixture")
    redline = build_v2_redline(documents, sentence_scores, v2_settings, "run_fixture")
    evidence = build_v2_evidence(sentence_scores, meeting_scores)
    predictions = build_v2_model_predictions(sentence_scores)
    audit = build_v2_model_audit(empty_v2_labels(), predictions, "run_fixture")
    return {
        "v2_meetings": meetings,
        "v2_documents": documents,
        "v2_sentences": sentences,
        "v2_sentence_scores": sentence_scores,
        "v2_document_scores": document_scores,
        "v2_meeting_scores": meeting_scores,
        "v2_subindices": subindices,
        "v2_calibration": calibration,
        "v2_redline": redline,
        "v2_evidence": evidence,
        "v2_labels": empty_v2_labels(),
        "v2_model_predictions": predictions,
        "v2_model_audit": audit,
    }


def _write_fixture_db(path: Path) -> None:
    write_tables(path, _build_v2_tables())


def test_v2_health_check_outputs(tmp_path, monkeypatch) -> None:
    database = tmp_path / "copom_tone.duckdb"
    _write_fixture_db(database)
    paths = ProjectPaths(
        database=database,
        raw=tmp_path / "raw",
        processed=tmp_path / "processed",
        figures=tmp_path / "figures",
        reports=tmp_path / "reports" / "meeting_notes",
    )
    monkeypatch.setattr("copom_tone_index.v2_health.get_paths", lambda: paths)

    result = v2_health_check_command()

    assert result.json_path.exists()
    assert result.html_path.exists()
    payload = pd.read_json(result.json_path, typ="series")
    assert payload["status"] in {"pass", "warning"}
    assert "warnings" in payload
    assert "errors" in payload
    assert "Resumo executivo" in result.html_path.read_text(encoding="utf-8")
    assert "implementation_status" in payload
    assert "Status de implementacao V2" in result.html_path.read_text(encoding="utf-8")


def test_v2_health_check_no_market_is_not_error(tmp_path) -> None:
    database = tmp_path / "copom_tone.duckdb"
    _write_fixture_db(database)

    report = build_v2_acceptance_report(database)

    assert report["market"]["status"] == "no_market_data"
    assert report["focus_v21"]["status"] == "not_started"
    assert report["v21_event_panel"]["status"] == "not_started"
    assert report["status"] != "fail"
    assert not any("market" in error.lower() for error in report["errors"])


def test_v2_health_orders_meeting_ids_by_meeting_number(tmp_path) -> None:
    database = tmp_path / "copom_tone.duckdb"
    _write_fixture_db(database)

    report = build_v2_acceptance_report(database)

    assert report["coverage"]["min_meeting_id"] == "copom_1"
    assert report["coverage"]["max_meeting_id"] == "copom_4"


def test_v2_health_includes_macro_micro_implementation_status(tmp_path) -> None:
    database = tmp_path / "copom_tone.duckdb"
    _write_fixture_db(database)

    report = build_v2_acceptance_report(database)
    implementation = report["implementation_status"]

    assert "macro" in implementation
    assert "micro" in implementation
    assert any(item["name"] == "V2.0 Indice defensavel" for item in implementation["macro"])
    assert any(item["name"] == "Scoring deterministico" for item in implementation["micro"])
    assert all(0 <= item["progress"] <= 100 for item in implementation["macro"] + implementation["micro"])


def test_v2_idempotency_duplicate_detection() -> None:
    tables = _build_v2_tables()
    duplicate = pd.concat([tables["v2_documents"], tables["v2_documents"].iloc[[0]]], ignore_index=True)
    tables["v2_documents"] = duplicate

    idempotency = idempotency_section(tables)

    assert idempotency["v2_documents"] > 0


def test_v2_health_reports_partial_redline_coverage() -> None:
    tables = _build_v2_tables()
    partial = tables["v2_redline"].iloc[:1].copy()
    tables["v2_redline"] = partial

    section = redline_section(tables)

    assert section["expected_document_pairs"] > section["document_pairs_compared"]
    assert section["document_pair_coverage"] < 0.9


def test_v2_score_formula_no_novelty_in_tone_level() -> None:
    tables = _build_v2_tables()

    check = formula_section(tables)

    assert check["status"] == "PASS"
    assert check["novelty_in_tone_level"] is False


def test_v2_informative_sentence_has_tone_level() -> None:
    tables = _build_v2_tables()
    informative = tables["v2_sentence_scores"][tables["v2_sentence_scores"]["is_informative"]]

    assert not informative.empty
    assert informative["tone_level"].notna().all()


def test_v2_health_allows_explicit_low_information_null_meeting_tone(tmp_path) -> None:
    tables = _build_v2_tables()
    scores = tables["v2_meeting_scores"].copy()
    scores.loc[scores.index[0], "tone_raw"] = pd.NA
    scores.loc[scores.index[0], "score_status"] = "low_information"
    tables["v2_meeting_scores"] = scores
    database = tmp_path / "copom_tone.duckdb"
    write_tables(database, tables)

    report = build_v2_acceptance_report(database)

    assert report["scores"]["meetings_with_expected_low_information_null_tone"] == 1
    assert report["scores"]["meetings_with_unexpected_null_final_tone"] == 0
    assert "Some V2 meeting scores have null final tone." not in report["errors"]


def test_v2_fixed_calibration_stability() -> None:
    v2_settings = load_v2_settings()
    tables = _build_v2_tables()
    base_scores = tables["v2_meeting_scores"].head(3).copy()
    calibration = build_v2_calibration(base_scores, v2_settings, "base")
    calibrated_base = apply_v2_calibration(base_scores, calibration, v2_settings)
    extended = pd.concat([base_scores, tables["v2_meeting_scores"].tail(1)], ignore_index=True)
    calibrated_extended = apply_v2_calibration(extended, calibration, v2_settings)

    left = calibrated_base[["meeting_id", "copom_tone_index_v2"]].reset_index(drop=True)
    right = calibrated_extended.head(3)[["meeting_id", "copom_tone_index_v2"]].reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right)


def test_v2_export_label_sample(tmp_path, monkeypatch) -> None:
    database = tmp_path / "copom_tone.duckdb"
    _write_fixture_db(database)
    paths = ProjectPaths(
        database=database,
        raw=tmp_path / "raw",
        processed=tmp_path / "processed",
        figures=tmp_path / "figures",
        reports=tmp_path / "reports" / "meeting_notes",
    )
    monkeypatch.setattr("copom_tone_index.v2_health.get_paths", lambda: paths)
    output = tmp_path / "labels" / "review_sample.csv"

    result = export_label_sample_command(n=12, out=output)
    sample = pd.read_csv(result.output_path)
    codebook = result.codebook_path.read_text(encoding="utf-8")

    assert result.rows <= 12
    assert result.codebook_path.exists()
    assert {"human_topic", "human_stance", "human_is_informative", "human_notes", "reviewer_id", "accepted"}.issubset(sample.columns)
    assert sample["human_topic"].isna().all()
    assert sample["document_type"].nunique() > 1
    assert sample["baseline_stance"].nunique() > 1
    assert sample["baseline_is_informative"].nunique() >= 1
    assert "COPOM Watch V2 Human Label Review Codebook" in codebook
    assert "`llm_bootstrap` labels are not formal ground truth" in codebook


def test_v2_export_label_sample_excludes_reviewed_sentences() -> None:
    tables = _build_v2_tables()
    scores = tables["v2_sentence_scores"]
    reviewed_id = str(scores["sentence_id"].iloc[0])

    sample = build_v2_label_review_sample(scores, tables["v2_subindices"], n=10, exclude_sentence_ids={reviewed_id})

    assert reviewed_id not in set(sample["sentence_id"].astype(str))
    assert len(sample) <= 10


def test_v2_label_codebook_lists_topics_and_import_command(tmp_path) -> None:
    sample_path = tmp_path / "review_sample.csv"
    codebook = build_v2_label_codebook(TOPICS, load_v2_settings(), sample_path)

    assert "`inflation_expectations`" in codebook
    assert "`human_stance`: `hawkish`, `dovish`, `neutral`" in codebook
    assert "copom-watch v2 import-labels" in codebook


def test_v2_import_blank_review_sample_is_not_accepted() -> None:
    sample = pd.DataFrame(
        [
            {
                "sentence_id": "s1",
                "baseline_topic": "inflation_expectations",
                "baseline_stance": "hawkish",
                "baseline_is_informative": True,
                "human_topic": "",
                "human_stance": "",
                "human_is_informative": "",
                "reviewer_id": "",
                "accepted": "",
            }
        ]
    )

    labels = normalize_v2_labels(sample, label_source="human")

    assert labels["label_status"].iloc[0] == "pending_review"
    assert labels["topic_label"].iloc[0] == ""
    assert labels["stance_label"].iloc[0] == ""


def test_v2_import_accepted_human_layout_and_audit_metrics() -> None:
    predictions = pd.DataFrame(
        [
            {"sentence_id": "s1", "predicted_topic": "inflation_expectations", "predicted_stance": "hawkish", "predicted_is_informative": True},
            {"sentence_id": "s2", "predicted_topic": "activity_growth", "predicted_stance": "dovish", "predicted_is_informative": True},
            {"sentence_id": "s3", "predicted_topic": "fiscal_risk", "predicted_stance": "hawkish", "predicted_is_informative": True},
        ]
    )
    imported = pd.DataFrame(
        [
            {
                "sentence_id": "s1",
                "baseline_topic": "inflation_expectations",
                "baseline_stance": "hawkish",
                "baseline_is_informative": True,
                "human_topic": "",
                "human_stance": "",
                "human_is_informative": "",
                "reviewer_id": "a",
                "accepted": True,
            },
            {
                "sentence_id": "s2",
                "human_topic": "activity_growth",
                "human_stance": "neutral",
                "human_is_informative": True,
                "reviewer_id": "a",
                "accepted": True,
            },
            {
                "sentence_id": "s3",
                "human_topic": "fiscal_risk",
                "human_stance": "hawkish",
                "human_is_informative": True,
                "reviewer_id": "a",
                "accepted": True,
            },
            {
                "sentence_id": "s3",
                "human_topic": "fiscal_risk",
                "human_stance": "hawkish",
                "human_is_informative": True,
                "reviewer_id": "b",
                "accepted": True,
            },
        ]
    )
    labels = normalize_v2_labels(imported, label_source="human")

    audit = build_v2_model_audit(labels, predictions, "run_audit")
    details = build_v2_model_audit_details(labels, predictions, "run_audit")

    metrics = dict(zip(audit["metric"], audit["value"], strict=False))
    assert labels["label_status"].eq("accepted").all()
    assert labels.loc[labels["sentence_id"] == "s1", "stance_label"].iloc[0] == "hawkish"
    assert metrics["accepted_labels"] == 4.0
    assert metrics["unique_accepted_sentences"] == 3.0
    assert metrics["matched_labels"] == 3.0
    assert round(metrics["stance_accuracy"], 6) == round(2 / 3, 6)
    assert metrics["stance_f1_macro"] < 1.0
    assert (audit["metric"] == "stance_confusion").any()
    assert metrics["human_stance_agreement"] == 1.0
    assert len(details) == 3
    assert "reviewer_count" in details
    assert details.loc[details["sentence_id"] == "s3", "reviewer_count"].iloc[0] == 2
    assert details["stance_correct"].tolist().count(False) == 1


def test_v2_benchmark_baseline_outputs(tmp_path, monkeypatch) -> None:
    database = tmp_path / "copom_tone.duckdb"
    tables = _build_v2_tables()
    scores = tables["v2_sentence_scores"].head(4).copy()
    labels = pd.DataFrame(
        [
            {
                "sentence_id": scores["sentence_id"].iloc[0],
                "human_topic": scores["primary_topic"].iloc[0],
                "human_stance": scores["stance"].iloc[0],
                "human_is_informative": bool(scores["is_informative"].iloc[0]),
                "reviewer_id": "claude",
                "accepted": True,
            },
            {
                "sentence_id": scores["sentence_id"].iloc[1],
                "human_topic": "activity_growth",
                "human_stance": "dovish",
                "human_is_informative": True,
                "reviewer_id": "claude_holdout_002",
                "accepted": True,
            },
            {
                "sentence_id": scores["sentence_id"].iloc[1],
                "human_topic": "activity_growth",
                "human_stance": "dovish",
                "human_is_informative": True,
                "reviewer_id": "gpt_holdout_002",
                "accepted": True,
            },
        ]
    )
    tables["v2_labels"] = normalize_v2_labels(labels, label_source="human")
    write_tables(database, tables)
    paths = ProjectPaths(
        database=database,
        raw=tmp_path / "raw",
        processed=tmp_path / "processed",
        figures=tmp_path / "figures",
        reports=tmp_path / "reports" / "meeting_notes",
    )
    monkeypatch.setattr("copom_tone_index.v2.get_paths", lambda _settings=None: paths)

    result = v2_benchmark_baseline_command()

    assert result.report_path.exists()
    assert (result.output_dir / "baseline_benchmark_by_sample.csv").exists()
    assert (result.output_dir / "baseline_benchmark_by_topic.csv").exists()
    assert (result.output_dir / "baseline_benchmark_by_stance.csv").exists()
    by_sample = pd.read_csv(result.output_dir / "baseline_benchmark_by_sample.csv")
    assert {"sample_001", "sample_002_claude", "sample_002_gpt", "sample_002_consensus", "total_consensus"}.issubset(
        set(by_sample["sample"])
    )
    assert "Baseline Benchmark" in result.report_path.read_text(encoding="utf-8")


def test_v2_benchmark_tables_include_taxonomy_boundary_cases() -> None:
    predictions = pd.DataFrame(
        [
            {
                "sentence_id": "s1",
                "predicted_topic": "policy_decision",
                "predicted_stance": "hawkish",
                "predicted_is_informative": True,
                "taxonomy_boundary_flag": "policy_decision_vs_forward_guidance",
            }
        ]
    )
    labels = normalize_v2_labels(
        pd.DataFrame(
            [
                {
                    "sentence_id": "s1",
                    "human_topic": "forward_guidance",
                    "human_stance": "hawkish",
                    "human_is_informative": True,
                    "reviewer_id": "claude",
                    "accepted": True,
                }
            ]
        ),
        label_source="human",
    )
    sentence_scores = pd.DataFrame(
        [
            {
                "sentence_id": "s1",
                "meeting_id": "copom_1",
                "nro_reuniao": 1,
                "document_type": "ata",
                "sentence_order": 1,
                "text": "O Copom decidiu manter a Selic e avaliara os proximos passos.",
                "tone_level": 0.5,
                "confidence": 0.8,
                "information_weight": 0.9,
                "evidence_terms": "[]",
                "taxonomy_boundary_flag": "policy_decision_vs_forward_guidance",
            }
        ]
    )

    by_sample, _, _ = build_baseline_benchmark_tables(labels, predictions, sentence_scores, "run_benchmark")
    total = by_sample[by_sample["sample"] == "total_consensus"].iloc[0]

    assert total["taxonomy_boundary_case"] == 1
    assert total["likely_baseline_error"] == 0


def test_v2_benchmark_gates_block_sample001_overfit() -> None:
    current_sample = pd.DataFrame(
        [
            {"sample": "sample_001", "error_rate": 0.20, "informativeness_accuracy": 0.94},
            {"sample": "sample_002_consensus", "error_rate": 0.60, "informativeness_accuracy": 0.93},
            {
                "sample": "total_consensus",
                "likely_baseline_error": 120,
                "stance_f1_macro": 0.72,
                "topic_accuracy": 0.70,
                "informativeness_accuracy": 0.94,
                "error_rate": 0.40,
            },
        ]
    )
    previous_sample = pd.DataFrame(
        [
            {"sample": "sample_001", "error_rate": 0.30, "informativeness_accuracy": 0.95},
            {"sample": "sample_002_consensus", "error_rate": 0.50, "informativeness_accuracy": 0.94},
        ]
    )

    gates = evaluate_benchmark_gates(current_sample, pd.DataFrame(), previous_sample, pd.DataFrame())

    assert gates["status"] == "fail"
    assert any("sample_001 improved while holdout" in message for message in gates["errors"])


def test_v2_benchmark_gates_block_large_stance_f1_drop() -> None:
    sample = pd.DataFrame(
        [
            {
                "sample": "total_consensus",
                "likely_baseline_error": 120,
                "stance_f1_macro": 0.72,
                "topic_accuracy": 0.70,
                "informativeness_accuracy": 0.94,
                "error_rate": 0.40,
            }
        ]
    )
    current_stance = pd.DataFrame([{"sample": "sample_002_consensus", "stance": "dovish", "f1": 0.60}])
    previous_stance = pd.DataFrame([{"sample": "sample_002_consensus", "stance": "dovish", "f1": 0.65}])

    gates = evaluate_benchmark_gates(sample, current_stance, pd.DataFrame(), previous_stance)

    assert gates["status"] == "fail"
    assert any("stance F1 dropped more than 3 p.p." in message for message in gates["errors"])


def test_v2_review_remaining_errors_outputs(tmp_path, monkeypatch) -> None:
    database = tmp_path / "copom_tone.duckdb"
    _write_fixture_db(database)
    paths = ProjectPaths(
        database=database,
        raw=tmp_path / "raw",
        processed=tmp_path / "processed",
        figures=tmp_path / "figures",
        reports=tmp_path / "reports" / "meeting_notes",
    )
    output_dir = tmp_path / "v2"
    output_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "sentence_id": "s1",
                "meeting_id": "copom_1",
                "nro_reuniao": 1,
                "document_type": "ata",
                "priority_score": 4.2,
                "topic_label": "forward_guidance",
                "predicted_topic": "policy_decision",
                "stance_label": "hawkish",
                "predicted_stance": "neutral",
                "is_informative_label": True,
                "predicted_is_informative": True,
                "taxonomy_boundary_flag": "",
                "error_interpretation": "likely_baseline_error",
                "human_conflict_types": "",
                "text": "O Copom indicou pressao relevante e postura cautelosa.",
            },
            {
                "sentence_id": "s2",
                "meeting_id": "copom_2",
                "nro_reuniao": 2,
                "document_type": "ata",
                "priority_score": 2.0,
                "topic_label": "policy_decision",
                "predicted_topic": "forward_guidance",
                "stance_label": "neutral",
                "predicted_stance": "neutral",
                "is_informative_label": True,
                "predicted_is_informative": True,
                "taxonomy_boundary_flag": "policy_decision_vs_forward_guidance",
                "error_interpretation": "likely_baseline_error",
                "human_conflict_types": "",
                "text": "O Copom decidiu manter a Selic e avaliara proximos passos.",
            },
        ]
    ).to_csv(output_dir / "model_audit_error_classification.csv", index=False)
    monkeypatch.setattr("copom_tone_index.v2.get_paths", lambda _settings=None: paths)

    result = v2_review_remaining_errors_command(limit=2)

    assert result.output_path and result.output_path.exists()
    review = pd.read_csv(result.output_path)
    assert {"generalizable_rule_candidate", "taxonomy_boundary"}.issubset(set(review["review_category"]))
    assert (tmp_path / "reports" / "v2" / "remaining_error_review.html").exists()


def test_v2_remaining_error_review_handles_empty() -> None:
    review = build_remaining_error_review(pd.DataFrame(), "run_empty", limit=177)

    assert review.empty
    assert "review_category" in review.columns


def _supervised_labels(scores: pd.DataFrame) -> pd.DataFrame:
    sentence_ids = scores["sentence_id"].head(6).tolist()
    rows = [
        (sentence_ids[0], "inflation_expectations", "hawkish", True, "claude"),
        (sentence_ids[1], "activity_growth", "dovish", True, "claude"),
        (sentence_ids[2], "institutional", "neutral", False, "claude"),
        (sentence_ids[3], "inflation_current", "hawkish", True, "claude"),
        (sentence_ids[4], "inflation_expectations", "hawkish", True, "claude_holdout_002"),
        (sentence_ids[4], "inflation_expectations", "hawkish", True, "gpt_holdout_002"),
        (sentence_ids[5], "activity_growth", "dovish", True, "claude_holdout_002"),
        (sentence_ids[5], "activity_growth", "dovish", True, "gpt_holdout_002"),
    ]
    return normalize_v2_labels(
        pd.DataFrame(
            [
                {
                    "sentence_id": sentence_id,
                    "human_topic": topic,
                    "human_stance": stance,
                    "human_is_informative": informative,
                    "reviewer_id": reviewer,
                    "accepted": True,
                }
                for sentence_id, topic, stance, informative, reviewer in rows
            ]
        ),
        label_source="human",
    )


def test_v2_train_supervised_outputs_and_keeps_official_scores(tmp_path, monkeypatch) -> None:
    database = tmp_path / "copom_tone.duckdb"
    tables = _build_v2_tables()
    tables["v2_labels"] = _supervised_labels(tables["v2_sentence_scores"])
    write_tables(database, tables)
    paths = ProjectPaths(
        database=database,
        raw=tmp_path / "raw",
        processed=tmp_path / "processed",
        figures=tmp_path / "figures",
        reports=tmp_path / "reports" / "meeting_notes",
    )
    monkeypatch.setattr("copom_tone_index.v2.get_paths", lambda _settings=None: paths)
    before = duckdb.connect(str(database), read_only=True).execute("SELECT COUNT(*) FROM v2_sentence_scores").fetchone()[0]

    result = v2_train_supervised_command()

    after = duckdb.connect(str(database), read_only=True).execute("SELECT COUNT(*) FROM v2_sentence_scores").fetchone()[0]
    assert result.output_path and result.output_path.exists()
    assert before == after
    audit = pd.read_csv(result.output_path)
    assert {"stance", "topic", "is_informative"}.issubset(set(audit["target"]))
    predictions = duckdb.connect(str(database), read_only=True).execute("SELECT * FROM v2_supervised_predictions").df()
    assert not predictions.empty
    assert set(predictions["model_status"]) == {"experimental"}
    assert (tmp_path / "reports" / "v2" / "supervised_model_report.html").exists()


def test_v2_supervised_insufficient_labels_report() -> None:
    predictions, audit = build_v2_supervised_model_artifacts(empty_v2_labels(), pd.DataFrame(), pd.DataFrame(), "run_supervised")

    assert predictions.empty
    assert set(audit["status"]) == {"insufficient_labels"}


def _build_v21_tables_for_freeze() -> dict[str, pd.DataFrame]:
    focus_vintages = pd.DataFrame(
        [
            {
                "focus_release_date": "2024-01-08",
                "focus_reference_date": "2024-12-31",
                "indicator": "Selic",
                "reference_year": 2024,
                "horizon": "current_year",
                "statistic": "median",
                "value": 11.5,
                "source": "focus_odata",
                "source_date": "",
                "collected_at": "2024-01-08T12:00:00",
                "query_signature": "selic-2024",
                "data_access_tier": "PUBLIC_API",
            }
        ]
    )
    focus_event_features = pd.DataFrame(
        [
            {
                "meeting_id": "copom_1",
                "event_type": "comunicado",
                "event_date": "2024-01-10",
                "known_at_timestamp": "2024-01-10 18:30:00",
                "indicator": "Selic",
                "reference_year": 2024,
                "horizon": "current_year",
                "statistic": "median",
                "pre_date": "2024-01-08",
                "pre_value": 11.5,
                "post_1_date": "2024-01-11",
                "post_1_value": 11.25,
                "post_2_date": "2024-01-12",
                "post_2_value": 11.0,
                "delta_post_1": -0.25,
                "delta_post_2": -0.5,
                "missing_reason": "",
            }
        ]
    )
    market_observations = pd.DataFrame(
        [
            {
                "asset": "USD_BRL_PTAX",
                "asset_class": "fx",
                "vertex": "spot",
                "timestamp": "2024-01-10 17:00:00",
                "value": 5.0,
                "source": "bcb_ptax",
                "source_file": "public",
                "data_access_tier": "PUBLIC_API",
            },
            {
                "asset": "USD_BRL_PTAX",
                "asset_class": "fx",
                "vertex": "spot",
                "timestamp": "2024-01-11 17:00:00",
                "value": 5.1,
                "source": "bcb_ptax",
                "source_file": "public",
                "data_access_tier": "PUBLIC_API",
            },
        ]
    )
    market_event_windows = pd.DataFrame(
        [
            {
                "meeting_id": "copom_1",
                "document_type": "comunicado",
                "asset": "USD_BRL_PTAX",
                "vertex": "spot",
                "window": "close_to_next_close",
                "status": "ok",
                "pre_timestamp": "2024-01-10 17:00:00",
                "known_at_timestamp": "2024-01-10 18:30:00",
                "post_timestamp": "2024-01-11 17:00:00",
                "market_reaction": 0.1,
            }
        ]
    )
    decision_expectations = pd.DataFrame(
        [
            {
                "meeting_id": "copom_1",
                "source": "focus_selic_proxy",
                "data_access_tier": "DERIVED",
                "as_of_timestamp": "2024-01-09 08:00:00",
                "expected_selic_change_bps": -25.0,
                "scenario_selic_change_bps": -25.0,
                "probability": 1.0,
                "is_proxy": True,
                "proxy_method": "focus_selic_proxy",
                "license_note": "",
            }
        ]
    )
    v21_event_panel = pd.DataFrame(
        [
            {
                "meeting_id": "copom_1",
                "nro_reuniao": 1,
                "data_referencia": "2024-01-10",
                "decision_surprise_status": "proxy",
                "decision_surprise_official_bps": pd.NA,
                "decision_surprise_proxy_bps": 0.0,
                "focus_status": "ok",
                "focus_feature_rows": 1,
                "focus_delta_coverage": 1.0,
                "market_status": "ready",
                "market_window_rows": 1,
                "market_ok_windows": 1,
            }
        ]
    )
    return {
        "focus_vintages": focus_vintages,
        "focus_event_features": focus_event_features,
        "market_observations": market_observations,
        "market_event_windows": market_event_windows,
        "decision_expectations": decision_expectations,
        "public_market_source_audit": pd.DataFrame([{"source": "ptax:USD_BRL", "status": "ok", "rows": 2, "detail": ""}]),
        "decision_expectation_source_audit": pd.DataFrame(
            [
                {"source": "b3:opcao_copom", "status": "unavailable", "rows": 0, "detail": "No structured public history."},
                {"source": "public:proxy", "status": "ok", "rows": 1, "detail": "Proxy only."},
            ]
        ),
        "public_market_coverage": pd.DataFrame([{"section": "market", "metric": "rows", "value": 2, "status": "ok"}]),
        "v21_event_panel": v21_event_panel,
    }


def _prepare_v22_fixture_database(tmp_path: Path) -> Path:
    database = tmp_path / "copom_tone.duckdb"
    tables = _build_v2_tables()
    tables.update(_build_v21_tables_for_freeze())
    tables["semantic_chunks"] = build_semantic_chunks(tables["v2_sentence_scores"].head(8), method="tfidf")
    write_tables(database, tables)
    return database


def _patch_v22_environment(tmp_path: Path, monkeypatch, database: Path) -> ProjectPaths:
    paths = ProjectPaths(
        database=database,
        raw=tmp_path / "raw",
        processed=tmp_path / "outputs" / "processed",
        figures=tmp_path / "outputs" / "figures",
        reports=tmp_path / "reports" / "meeting_notes",
    )
    app_data = tmp_path / "app_data"
    monkeypatch.setattr("copom_tone_index.v22.get_paths", lambda: paths)
    monkeypatch.setattr("copom_tone_index.v2.get_paths", lambda _settings=None: paths)
    monkeypatch.setattr("copom_tone_index.v2_health.get_paths", lambda: paths)
    monkeypatch.setattr("copom_tone_index.v21.get_paths", lambda: paths)
    monkeypatch.setattr("copom_tone_index.v22.ROOT", tmp_path)
    monkeypatch.setattr("copom_tone_index.v22.PUBLIC_DATA_DIR", app_data)
    monkeypatch.setattr("copom_tone_index.v22.PUBLIC_DATA_DB", app_data / "copom_watch_public.duckdb")
    monkeypatch.setattr("copom_tone_index.v22.PUBLIC_DATA_MANIFEST", app_data / "public_data_manifest.json")
    (tmp_path / ".streamlit").mkdir(parents=True)
    (tmp_path / "streamlit_app.py").write_text("from copom_tone_index.dashboard.app import main\n", encoding="utf-8")
    (tmp_path / "requirements.txt").write_text("streamlit>=1.35\n", encoding="utf-8")
    (tmp_path / ".streamlit" / "config.toml").write_text("[server]\nheadless=true\n", encoding="utf-8")
    return paths


def test_v2_freeze_release_manifest(tmp_path, monkeypatch) -> None:
    database = tmp_path / "copom_tone.duckdb"
    tables = _build_v2_tables()
    tables["v2_labels"] = _supervised_labels(tables["v2_sentence_scores"])
    write_tables(database, tables)
    paths = ProjectPaths(
        database=database,
        raw=tmp_path / "raw",
        processed=tmp_path / "processed",
        figures=tmp_path / "figures",
        reports=tmp_path / "reports" / "meeting_notes",
    )
    monkeypatch.setattr("copom_tone_index.v2.get_paths", lambda _settings=None: paths)
    monkeypatch.setattr("copom_tone_index.v2_health.get_paths", lambda: paths)
    v2_audit_command()
    v2_benchmark_baseline_command()

    result = v2_freeze_release_command(version="v2.0.4-test")

    assert result.status == "completed"
    assert result.output_path and result.output_path.exists()
    manifest = pd.read_json(result.output_path, typ="series")
    assert manifest["version"] == "v2.0.4-test"
    assert manifest["rule_engine_version"] == "taxonomy-rules-v2.0.4"
    assert manifest["health_errors"] == 0
    assert (tmp_path / "reports" / "v2" / "v2_0_methodology_report.html").exists()
    assert (tmp_path / "reports" / "v2" / "releases" / "v2.0.4-test" / "release_manifest.json").exists()


def test_v21_freeze_release_manifest(tmp_path, monkeypatch) -> None:
    database = tmp_path / "copom_tone.duckdb"
    tables = _build_v2_tables()
    tables["v2_labels"] = _supervised_labels(tables["v2_sentence_scores"])
    tables.update(_build_v21_tables_for_freeze())
    write_tables(database, tables)
    paths = ProjectPaths(
        database=database,
        raw=tmp_path / "raw",
        processed=tmp_path / "outputs" / "processed",
        figures=tmp_path / "outputs" / "figures",
        reports=tmp_path / "reports" / "meeting_notes",
    )
    monkeypatch.setattr("copom_tone_index.v2.get_paths", lambda _settings=None: paths)
    monkeypatch.setattr("copom_tone_index.v2_health.get_paths", lambda: paths)
    monkeypatch.setattr("copom_tone_index.v21.get_paths", lambda: paths)
    output_dir = tmp_path / "outputs" / "v2"
    report_dir = tmp_path / "reports" / "v2"
    output_dir.mkdir(parents=True)
    report_dir.mkdir(parents=True)
    tables["focus_event_features"].to_csv(output_dir / "focus_event_features.csv", index=False)
    tables["market_event_windows"].to_csv(output_dir / "market_event_study.csv", index=False)
    tables["v21_event_panel"].to_csv(output_dir / "v21_event_panel.csv", index=False)

    result = v21_freeze_release_command(version="v2.1-test")

    assert result.status == "completed"
    manifest = pd.read_json(result.output_path, typ="series")
    assert manifest["version"] == "v2.1-test"
    assert manifest["rule_engine_version"] == "taxonomy-rules-v2.0.4"
    assert manifest["v2_health"]["errors"] == 0
    assert manifest["v21_health"]["errors"] == 0
    assert manifest["surprise_metrics"]["official_surprises"] == 0
    assert manifest["surprise_metrics"]["proxy_surprises"] == 1
    assert (tmp_path / "reports" / "v2" / "v21_release_summary.html").exists()
    assert (tmp_path / "reports" / "v2" / "releases" / "v2.1-test" / "v21_release_manifest.json").exists()


def test_v21_freeze_release_fails_without_event_panel(tmp_path, monkeypatch) -> None:
    database = tmp_path / "copom_tone.duckdb"
    tables = _build_v2_tables()
    tables["v2_labels"] = _supervised_labels(tables["v2_sentence_scores"])
    tables.update(_build_v21_tables_for_freeze())
    tables["v21_event_panel"] = tables["v21_event_panel"].iloc[0:0].copy()
    write_tables(database, tables)
    paths = ProjectPaths(
        database=database,
        raw=tmp_path / "raw",
        processed=tmp_path / "outputs" / "processed",
        figures=tmp_path / "outputs" / "figures",
        reports=tmp_path / "reports" / "meeting_notes",
    )
    monkeypatch.setattr("copom_tone_index.v2.get_paths", lambda _settings=None: paths)
    monkeypatch.setattr("copom_tone_index.v2_health.get_paths", lambda: paths)
    monkeypatch.setattr("copom_tone_index.v21.get_paths", lambda: paths)
    output_dir = tmp_path / "outputs" / "v2"
    output_dir.mkdir(parents=True)
    tables["focus_event_features"].to_csv(output_dir / "focus_event_features.csv", index=False)
    tables["market_event_windows"].to_csv(output_dir / "market_event_study.csv", index=False)

    result = v21_freeze_release_command(version="v2.1-missing-panel")

    assert result.status == "fail"
    manifest = pd.read_json(result.output_path, typ="series")
    assert any("event panel" in error.lower() for error in manifest["errors"])


def test_v21_freeze_detects_proxy_official_contamination() -> None:
    panel = pd.DataFrame(
        [
            {"decision_surprise_status": "proxy", "decision_surprise_official_bps": 10.0},
            {"decision_surprise_status": "official", "decision_surprise_official_bps": 5.0},
        ]
    )

    assert v21_release_official_proxy_contamination(panel) == 1


def test_v22_package_public_data_and_health(tmp_path, monkeypatch) -> None:
    database = _prepare_v22_fixture_database(tmp_path)
    paths = _patch_v22_environment(tmp_path, monkeypatch, database)
    report_dir = paths.reports.parent / "v2"
    report_dir.mkdir(parents=True)
    (report_dir / "release_manifest.json").write_text('{"version":"v2.0.4-test"}', encoding="utf-8")
    (report_dir / "v21_release_manifest.json").write_text(
        '{"version":"v2.1-test","surprise_metrics":{"official_surprises":0}}',
        encoding="utf-8",
    )

    package = package_public_data_command()
    health = v22_health_command()

    assert package.status == "completed"
    assert package.output_path.exists()
    assert health.status == "warning"
    assert health.errors == 0
    payload = pd.read_json(health.json_path, typ="series")
    assert "v2_labels" not in payload["package"]["tables"]
    assert "semantic_chunks" in payload["package"]["tables"]
    assert payload["semantic"]["status"] == "ready"


def test_v22_health_fails_without_public_package(tmp_path, monkeypatch) -> None:
    database = tmp_path / "copom_tone.duckdb"
    write_tables(database, _build_v2_tables())
    _patch_v22_environment(tmp_path, monkeypatch, database)

    health = v22_health_command()

    assert health.status == "fail"
    payload = pd.read_json(health.json_path, typ="series")
    assert any("Public DuckDB package" in error for error in payload["errors"])


def test_v22_freeze_release_manifest(tmp_path, monkeypatch) -> None:
    database = _prepare_v22_fixture_database(tmp_path)
    paths = _patch_v22_environment(tmp_path, monkeypatch, database)
    report_dir = paths.reports.parent / "v2"
    report_dir.mkdir(parents=True)
    (report_dir / "release_manifest.json").write_text('{"version":"v2.0.4-test"}', encoding="utf-8")
    (report_dir / "v21_release_manifest.json").write_text(
        '{"version":"v2.1-test","surprise_metrics":{"official_surprises":0}}',
        encoding="utf-8",
    )
    (report_dir / "semantic_ask_report.html").write_text("<html>Ask</html>", encoding="utf-8")
    package_public_data_command()

    result = v22_freeze_release_command(version="v2.2-test")

    assert result.status == "completed"
    assert result.output_path and result.output_path.exists()
    manifest = pd.read_json(result.output_path, typ="series")
    assert manifest["version"] == "v2.2-test"
    assert manifest["rule_engine_version"] == "taxonomy-rules-v2.0.4"
    assert manifest["public_package"]["sha256"]
    assert manifest["public_package"]["bytes"] > 0
    assert manifest["rag"]["status"] == "ready"
    assert manifest["v22_health"]["errors"] == 0
    assert (tmp_path / "reports" / "v2" / "v22_release_summary.html").exists()
    assert (tmp_path / "reports" / "v2" / "releases" / "v2.2-test" / "v22_release_manifest.json").exists()


def test_v22_freeze_release_fails_without_package(tmp_path, monkeypatch) -> None:
    database = _prepare_v22_fixture_database(tmp_path)
    _patch_v22_environment(tmp_path, monkeypatch, database)

    result = v22_freeze_release_command(version="v2.2-missing-package")

    assert result.status == "fail"
    manifest = pd.read_json(result.output_path, typ="series")
    assert any("Public DuckDB package" in error for error in manifest["errors"])


def test_v2_health_label_metrics_include_human_agreement_and_informativeness() -> None:
    predictions = pd.DataFrame(
        [
            {"sentence_id": "s1", "predicted_topic": "inflation_expectations", "predicted_stance": "hawkish", "predicted_is_informative": True},
            {"sentence_id": "s2", "predicted_topic": "activity_growth", "predicted_stance": "dovish", "predicted_is_informative": False},
        ]
    )
    labels = normalize_v2_labels(
        pd.DataFrame(
            [
                {
                    "sentence_id": "s1",
                    "human_topic": "inflation_expectations",
                    "human_stance": "hawkish",
                    "human_is_informative": True,
                    "reviewer_id": "a",
                    "accepted": True,
                },
                {
                    "sentence_id": "s1",
                    "human_topic": "inflation_expectations",
                    "human_stance": "hawkish",
                    "human_is_informative": True,
                    "reviewer_id": "b",
                    "accepted": True,
                },
                {
                    "sentence_id": "s2",
                    "human_topic": "activity_growth",
                    "human_stance": "neutral",
                    "human_is_informative": False,
                    "reviewer_id": "a",
                    "accepted": True,
                },
            ]
        ),
        label_source="human",
    )
    section = labels_section(
        {
            "v2_labels": labels,
            "v2_model_predictions": predictions,
            "v2_model_audit": build_v2_model_audit(labels, predictions, "run_audit"),
        }
    )

    assert section["metrics"]["informativeness_accuracy"] == 1.0
    assert section["metrics"]["human_stance_agreement"] == 1.0
    assert section["metrics"]["human_agreement"] == 1.0


def test_v2_audit_error_analysis_prioritizes_mismatches() -> None:
    predictions = pd.DataFrame(
        [
            {"sentence_id": "s1", "predicted_topic": "inflation_expectations", "predicted_stance": "neutral", "predicted_is_informative": True},
            {"sentence_id": "s2", "predicted_topic": "credit_conditions", "predicted_stance": "hawkish", "predicted_is_informative": True},
            {"sentence_id": "s3", "predicted_topic": "institutional", "predicted_stance": "neutral", "predicted_is_informative": False},
        ]
    )
    labels = normalize_v2_labels(
        pd.DataFrame(
            [
                {
                    "sentence_id": "s1",
                    "human_topic": "inflation_expectations",
                    "human_stance": "hawkish",
                    "human_is_informative": True,
                    "reviewer_id": "a",
                    "accepted": True,
                },
                {
                    "sentence_id": "s2",
                    "human_topic": "institutional",
                    "human_stance": "neutral",
                    "human_is_informative": False,
                    "reviewer_id": "a",
                    "accepted": True,
                },
                {
                    "sentence_id": "s3",
                    "human_topic": "institutional",
                    "human_stance": "neutral",
                    "human_is_informative": False,
                    "reviewer_id": "a",
                    "accepted": True,
                },
            ]
        ),
        label_source="human",
    )
    sentence_scores = pd.DataFrame(
        [
            {
                "sentence_id": "s1",
                "meeting_id": "copom_1",
                "nro_reuniao": 1,
                "document_type": "ata",
                "sentence_order": 1,
                "text": "Expectativas desancoradas.",
                "tone_level": 0.0,
                "confidence": 0.8,
                "information_weight": 0.9,
                "evidence_terms": "[]",
            },
            {
                "sentence_id": "s2",
                "meeting_id": "copom_1",
                "nro_reuniao": 1,
                "document_type": "ata",
                "sentence_order": 2,
                "text": "Presentes: membros do Copom.",
                "tone_level": 0.5,
                "confidence": 0.9,
                "information_weight": 0.85,
                "evidence_terms": "[]",
            },
        ]
    )

    errors = build_v2_model_audit_error_analysis(labels, predictions, sentence_scores, "run_audit")
    summary = build_v2_model_audit_error_summary(errors, "run_audit")
    html = build_v2_model_audit_report_html(build_v2_model_audit(labels, predictions, "run_audit"), errors, summary, "run_audit")

    assert len(errors) == 2
    assert errors.iloc[0]["sentence_id"] == "s2"
    assert "informativeness_mismatch" in errors.iloc[0]["issue_types"]
    assert "tighten_low_information_or_header_filter" in set(errors["suggested_action"])
    assert {"stance_mismatch", "topic_mismatch", "informativeness_mismatch"}.intersection(set(summary["issue_type"]))
    assert "COPOM Watch V2 Model Audit" in html


def test_v2_disagreement_report_separates_baseline_error_from_taxonomy_ambiguity() -> None:
    predictions = pd.DataFrame(
        [
            {"sentence_id": "s1", "predicted_topic": "policy_decision", "predicted_stance": "hawkish", "predicted_is_informative": True},
            {"sentence_id": "s2", "predicted_topic": "institutional", "predicted_stance": "neutral", "predicted_is_informative": False},
        ]
    )
    labels = normalize_v2_labels(
        pd.DataFrame(
            [
                {
                    "sentence_id": "s1",
                    "human_topic": "forward_guidance",
                    "human_stance": "hawkish",
                    "human_is_informative": True,
                    "reviewer_id": "a",
                    "accepted": True,
                },
                {
                    "sentence_id": "s1",
                    "human_topic": "policy_decision",
                    "human_stance": "hawkish",
                    "human_is_informative": True,
                    "reviewer_id": "b",
                    "accepted": True,
                },
                {
                    "sentence_id": "s2",
                    "human_topic": "activity_growth",
                    "human_stance": "dovish",
                    "human_is_informative": True,
                    "reviewer_id": "a",
                    "accepted": True,
                },
                {
                    "sentence_id": "s2",
                    "human_topic": "activity_growth",
                    "human_stance": "dovish",
                    "human_is_informative": True,
                    "reviewer_id": "b",
                    "accepted": True,
                },
            ]
        ),
        label_source="human",
    )
    sentence_scores = pd.DataFrame(
        [
            {
                "sentence_id": "s1",
                "meeting_id": "copom_1",
                "nro_reuniao": 1,
                "document_type": "ata",
                "sentence_order": 1,
                "text": "O Comite avaliara os proximos passos.",
                "tone_level": 0.5,
                "confidence": 0.8,
                "information_weight": 0.9,
                "evidence_terms": "[]",
            },
            {
                "sentence_id": "s2",
                "meeting_id": "copom_1",
                "nro_reuniao": 1,
                "document_type": "ata",
                "sentence_order": 2,
                "text": "A atividade perdeu tracao.",
                "tone_level": 0.0,
                "confidence": 0.2,
                "information_weight": 0.0,
                "evidence_terms": "[]",
            },
        ]
    )

    errors = build_v2_model_audit_error_analysis(labels, predictions, sentence_scores, "run_audit")
    disagreements = build_v2_reviewer_disagreements(labels, predictions, sentence_scores, "run_audit")
    classified = build_v2_model_audit_error_classification(errors, disagreements, "run_audit")
    html = build_v2_validation_disagreement_report_html(
        build_v2_model_audit(labels, predictions, "run_audit"),
        disagreements,
        classified,
        "run_audit",
    )

    assert len(disagreements) == 1
    assert disagreements.iloc[0]["conflict_types"] == "topic_conflict"
    interpretations = dict(zip(classified["sentence_id"], classified["error_interpretation"], strict=False))
    assert interpretations["s1"] == "legitimate_taxonomy_ambiguity"
    assert interpretations["s2"] == "likely_baseline_error"
    assert "Validation Disagreement Report" in html


def test_v2_llm_bootstrap_labels_do_not_count_as_formal_truth() -> None:
    predictions = pd.DataFrame(
        [
            {
                "sentence_id": "s1",
                "predicted_topic": "inflation_expectations",
                "predicted_stance": "hawkish",
                "predicted_is_informative": True,
            }
        ]
    )
    imported = pd.DataFrame(
        [
            {
                "sentence_id": "s1",
                "human_topic": "inflation_expectations",
                "human_stance": "hawkish",
                "human_is_informative": True,
                "reviewer_id": "llm",
                "accepted": True,
            }
        ]
    )

    labels = normalize_v2_labels(imported, label_source="llm_bootstrap")
    audit = build_v2_model_audit(labels, predictions, "run_audit")

    assert accepted_v2_labels(labels).empty
    assert audit["status"].iloc[0] == "needs_human_acceptance"


def test_v2_label_visual_status_distinguishes_ready_from_done() -> None:
    ready = {
        "audit_status": "ready",
        "accepted_labels": 300,
        "unique_accepted_sentences": 300,
        "metrics": {"stance_accuracy": 0.517},
    }
    passed = {
        "audit_status": "passed",
        "accepted_labels": 300,
        "unique_accepted_sentences": 300,
        "metrics": {"stance_accuracy": 0.72},
    }

    assert label_visual_status(ready) == "READY_WITH_WARNINGS"
    assert label_visual_status(passed) == "DONE"


def test_v2_no_v1_breakage() -> None:
    sentences = pd.DataFrame(
        [
            {
                "sentence_id": "s1",
                "document_id": "d1",
                "meeting_id": "m1",
                "nro_reuniao": 1,
                "document_type": "comunicado",
                "sentence_order": 1,
                "text": "As expectativas seguem desancoradas.",
            }
        ]
    )
    classified = classify_sentences_baseline(sentences, TOPICS, LEXICON, "p1")
    assert classified["stance"].iloc[0] == "hawkish"
    assert callable(validate_existing_outputs)

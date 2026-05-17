import numpy as np
import pandas as pd

from copom_tone_index.bcb import attach_selic_to_meetings, build_focus_revisions
from copom_tone_index.econometric_review import (
    audit_data_quality,
    determine_overall_status,
    leave_one_out_metrics,
    run_regression_suite,
    validate_regression_inputs,
)


def test_selic_audit_accepts_changing_decision_fixture() -> None:
    meetings = pd.DataFrame(
        [
            {
                "meeting_id": "m1",
                "nro_reuniao": 1,
                "data_referencia": pd.Timestamp("2024-01-10"),
                "titulo_comunicado": "Copom reduz a taxa Selic para 10,50% a.a.",
                "in_operational_window": True,
            },
            {
                "meeting_id": "m2",
                "nro_reuniao": 2,
                "data_referencia": pd.Timestamp("2024-02-10"),
                "titulo_comunicado": "Copom eleva a taxa Selic para 10,75% a.a.",
                "in_operational_window": True,
            },
        ]
    )
    selic = pd.DataFrame(
        [
            {"date": pd.Timestamp("2024-01-09"), "selic_target": 11.00},
            {"date": pd.Timestamp("2024-01-10"), "selic_target": 10.50},
            {"date": pd.Timestamp("2024-02-09"), "selic_target": 10.50},
            {"date": pd.Timestamp("2024-02-10"), "selic_target": 10.75},
        ]
    )
    meetings = attach_selic_to_meetings(meetings, selic)
    documents = pd.DataFrame(
        [
            {"document_id": "m1c", "meeting_id": "m1"},
            {"document_id": "m1a", "meeting_id": "m1"},
            {"document_id": "m2c", "meeting_id": "m2"},
            {"document_id": "m2a", "meeting_id": "m2"},
        ]
    )
    sentences = pd.DataFrame(
        [
            {"sentence_id": "s1", "stance_score": 0.2},
            {"sentence_id": "s2", "stance_score": -0.1},
        ]
    )
    scores = meetings[["meeting_id", "delta_selic", "selic_pos", "titulo_comunicado"]].copy()
    scores["communication_surprise"] = [0.1, -0.1]
    focus = pd.DataFrame(
        [
            {"delta_post_comunicado": 0.1, "delta_post_ata": 0.2},
            {"delta_post_comunicado": -0.1, "delta_post_ata": -0.2},
        ]
    )
    audit = audit_data_quality(
        {
            "copom_meetings": meetings,
            "copom_documents": documents,
            "copom_sentences": sentences,
            "copom_scores": scores,
            "focus_revisions": focus,
        }
    )
    delta_row = audit[audit["check"] == "delta_selic_variation"].iloc[0]
    assert delta_row["status"] == "ok"
    assert set(meetings["delta_selic"].round(2)) == {-0.50, 0.25}


def test_validate_regression_inputs_blocks_constant_predictor() -> None:
    data = pd.DataFrame({"target": np.arange(10, dtype=float), "delta_selic": 0.0})
    reason = validate_regression_inputs(data, "target", ["delta_selic"], min_exploratory_obs=5)
    assert reason == "predictor_no_variation:delta_selic"


def test_focus_revisions_handle_weekend_gap() -> None:
    meetings = pd.DataFrame(
        [
            {
                "meeting_id": "m1",
                "data_referencia": pd.Timestamp("2024-03-22"),
                "data_comunicado": pd.Timestamp("2024-03-22"),
                "data_ata": pd.Timestamp("2024-03-26"),
            }
        ]
    )
    focus = pd.DataFrame(
        [
            {"variable": "IPCA", "date": pd.Timestamp("2024-03-21"), "reference_year": 2024, "median": 4.0},
            {"variable": "IPCA", "date": pd.Timestamp("2024-03-25"), "reference_year": 2024, "median": 4.1},
            {"variable": "IPCA", "date": pd.Timestamp("2024-03-27"), "reference_year": 2024, "median": 4.2},
        ]
    )
    revisions = build_focus_revisions(meetings, focus, ["IPCA"])
    row = revisions[revisions["reference_year"] == 2024].iloc[0]
    assert row["focus_pre_date"] == pd.Timestamp("2024-03-21")
    assert row["focus_post_comunicado_date"] == pd.Timestamp("2024-03-25")
    assert row["focus_post_ata_date"] == pd.Timestamp("2024-03-27")


def test_leave_one_out_metrics_on_synthetic_linear_data() -> None:
    data = pd.DataFrame({"target": np.arange(1, 11, dtype=float) * 2, "tone_raw": np.arange(1, 11, dtype=float)})
    metrics = leave_one_out_metrics(data, "target", ["tone_raw"])
    assert metrics["rmse"] < 1e-10
    assert metrics["mae"] < 1e-10
    assert metrics["directional_accuracy"] == 1.0


def test_regression_suite_estimates_valid_specs_and_blocks_invalid_specs() -> None:
    n = 10
    scores = pd.DataFrame(
        {
            "meeting_id": [f"m{i}" for i in range(n)],
            "tone_raw": np.linspace(-1, 1, n),
            "delta_tone": np.linspace(-0.5, 0.5, n),
            "delta_selic": np.linspace(-0.25, 0.25, n),
            "communication_surprise": np.linspace(-0.2, 0.2, n),
        }
    )
    focus = pd.DataFrame(
        {
            "meeting_id": [f"m{i}" for i in range(n)],
            "variable": "Selic",
            "reference_year": 2024,
            "delta_post_comunicado": np.linspace(-0.1, 0.1, n),
            "delta_post_ata": np.linspace(-0.2, 0.2, n),
        }
    )
    diagnostics = run_regression_suite(scores, focus, min_exploratory_obs=5, min_formal_obs=30)
    assert ((diagnostics["model"] == "tone_raw") & (diagnostics["status"] == "estimated")).any()
    blocked_comm = diagnostics[diagnostics["model"] == "communication_surprise"]
    assert (blocked_comm["status"] == "blocked").all()
    assert blocked_comm["block_reason"].str.contains("communication_surprise_requires_min_30_obs").all()


def test_overall_status_invalid_when_focus_is_unavailable() -> None:
    data_quality = pd.DataFrame(
        [
            {"status": "ok", "check": "meeting_coverage"},
            {"status": "invalid_for_inference", "check": "focus_revision_coverage"},
        ]
    )
    diagnostics = pd.DataFrame([{"status": "blocked"}])
    assert determine_overall_status(data_quality, diagnostics) == "INVALID_FOR_INFERENCE"

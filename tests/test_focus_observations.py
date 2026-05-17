from pathlib import Path

import numpy as np
import pandas as pd

from copom_tone_index.focus import (
    audit_focus_coverage,
    build_focus_revisions_from_observations,
    coverage_status,
    fetch_focus_observations_for_meetings,
    import_focus_snapshot,
    normalize_focus_observations,
    overall_focus_coverage_status,
)


class FakeFocusClient:
    def __init__(self) -> None:
        self.calls = []

    def get_json(self, url: str, cache_name: str, params: dict | None = None) -> dict:
        self.calls.append({"url": url, "cache_name": cache_name, "params": params})
        filter_text = params["$filter"]
        if "DataReferencia eq '2024'" in filter_text:
            return {
                "value": [
                    {"Indicador": "Selic", "Data": "2024-03-18", "DataReferencia": "2024", "Mediana": 10.0},
                    {"Indicador": "Selic", "Data": "2024-03-21", "DataReferencia": "2024", "Mediana": 10.25},
                ]
            }
        return {"value": []}


def test_fetch_focus_observations_event_queries_from_odata_fixture() -> None:
    meetings = pd.DataFrame(
        [
            {
                "meeting_id": "m1",
                "data_referencia": pd.Timestamp("2024-03-20"),
                "data_comunicado": pd.Timestamp("2024-03-20"),
                "data_ata": pd.Timestamp("2024-03-26"),
            }
        ]
    )
    observations = fetch_focus_observations_for_meetings(FakeFocusClient(), meetings, ["Selic"])
    assert len(observations) == 2
    assert set(observations["source"]) == {"focus_odata"}
    assert observations["query_signature"].str.len().gt(0).all()


def test_import_focus_snapshot_csv_fallback(tmp_path: Path) -> None:
    snapshot = tmp_path / "focus_snapshot.csv"
    snapshot.write_text(
        "Indicador,DataReferencia,Mediana\nSelic,2024,10.5\nIPCA,2024,4.1\n",
        encoding="utf-8",
    )
    observations = import_focus_snapshot(snapshot, "2024-03-25")
    assert len(observations) == 2
    assert set(observations["source"]) == {"focus_report_fallback"}
    assert set(observations["date"]) == {pd.Timestamp("2024-03-25")}


def test_focus_revision_selection_and_missing_reason() -> None:
    meetings = pd.DataFrame(
        [
            {
                "meeting_id": "m1",
                "data_referencia": pd.Timestamp("2024-03-20"),
                "data_comunicado": pd.Timestamp("2024-03-20"),
                "data_ata": pd.Timestamp("2024-03-26"),
            }
        ]
    )
    observations = normalize_focus_observations(
        pd.DataFrame(
            [
                {"variable": "Selic", "reference_year": 2024, "date": "2024-03-18", "median": 10.0, "source": "fixture"},
                {"variable": "Selic", "reference_year": 2024, "date": "2024-03-21", "median": 10.25, "source": "fixture"},
            ]
        )
    )
    revisions = build_focus_revisions_from_observations(meetings, observations, ["Selic"], post_days=3)
    row = revisions[revisions["reference_year"] == 2024].iloc[0]
    assert row["focus_pre_value"] == 10.0
    assert row["focus_post_comunicado_value"] == 10.25
    assert np.isnan(row["focus_post_ata_value"])
    assert row["post_ata_missing_reason"] == "no_observation_within_3_days_after_event"


def test_focus_coverage_status_thresholds() -> None:
    assert coverage_status(0.81) == "ok"
    assert coverage_status(0.40) == "limited_data"
    assert coverage_status(0.39) == "invalid_for_inference"


def test_focus_coverage_audit_ok_with_high_coverage() -> None:
    revisions = pd.DataFrame(
        [
            {
                "meeting_id": f"m{i}",
                "variable": "Selic",
                "reference_year": 2024,
                "focus_pre_value": 10.0,
                "delta_post_comunicado": 0.1,
                "delta_post_ata": 0.2,
            }
            for i in range(5)
        ]
    )
    coverage = audit_focus_coverage(revisions)
    assert coverage.iloc[0]["status"] == "ok"
    assert overall_focus_coverage_status(coverage) == "ok"

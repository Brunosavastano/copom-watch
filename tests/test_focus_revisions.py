import pandas as pd

from copom_tone_index.bcb import build_focus_revisions


def test_build_focus_revisions_selects_pre_and_post_dates() -> None:
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
    focus = pd.DataFrame(
        [
            {"variable": "Selic", "date": pd.Timestamp("2024-03-18"), "reference_year": 2024, "median": 10.0},
            {"variable": "Selic", "date": pd.Timestamp("2024-03-21"), "reference_year": 2024, "median": 10.25},
            {"variable": "Selic", "date": pd.Timestamp("2024-03-27"), "reference_year": 2024, "median": 10.50},
            {"variable": "Selic", "date": pd.Timestamp("2024-03-18"), "reference_year": 2025, "median": 9.0},
        ]
    )
    revisions = build_focus_revisions(meetings, focus, ["Selic"])
    current_year = revisions[revisions["reference_year"] == 2024].iloc[0]
    assert current_year["focus_pre_value"] == 10.0
    assert current_year["focus_post_comunicado_value"] == 10.25
    assert current_year["focus_post_ata_value"] == 10.50
    assert current_year["delta_post_comunicado"] == 0.25

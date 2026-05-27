from __future__ import annotations

import hashlib
import html
import logging
from dataclasses import dataclass
from datetime import timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlencode

import duckdb
import numpy as np
import pandas as pd

from copom_tone_index.config import get_paths, load_settings
from copom_tone_index.http_client import CachedHttpClient, FetchError
from copom_tone_index.storage import export_tables, read_table, write_tables

LOGGER = logging.getLogger(__name__)

BCB_FOCUS_ANNUAL = "https://olinda.bcb.gov.br/olinda/servico/Expectativas/versao/v1/odata/ExpectativasMercadoAnuais"
FOCUS_ODATA_MIN_DATE = pd.Timestamp("2017-01-01")
FOCUS_OBSERVATION_COLUMNS = [
    "variable",
    "reference_year",
    "date",
    "median",
    "mean",
    "std",
    "minimum",
    "maximum",
    "source",
    "source_date",
    "fetched_at",
    "query_signature",
]
FOCUS_V21_VARIABLES = ["Selic", "IPCA", "PIB Total", "Câmbio"]
FOCUS_VINTAGE_COLUMNS = [
    "focus_release_date",
    "focus_reference_date",
    "indicator",
    "reference_year",
    "horizon",
    "statistic",
    "value",
    "source",
    "source_date",
    "collected_at",
    "query_signature",
    "data_access_tier",
]
FOCUS_EVENT_FEATURE_COLUMNS = [
    "meeting_id",
    "nro_reuniao",
    "event_type",
    "event_date",
    "known_at_timestamp",
    "indicator",
    "reference_year",
    "horizon",
    "statistic",
    "pre_date",
    "pre_value",
    "post_1_date",
    "post_1_value",
    "post_2_date",
    "post_2_value",
    "delta_post_1",
    "delta_post_2",
    "missing_reason",
    "source",
    "data_access_tier",
]


@dataclass(frozen=True)
class FocusCommandResult:
    observations: int
    revisions: int
    coverage_status: str
    output_path: Path | None = None


@dataclass(frozen=True)
class FocusV21CommandResult:
    vintages: int
    event_features: int
    coverage_status: str
    output_path: Path | None = None


def fetch_focus_observations_for_meetings(
    client: CachedHttpClient,
    meetings: pd.DataFrame,
    variables: list[str],
    pre_days: int = 30,
    post_days: int = 14,
    max_consecutive_failures: int = 3,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    consecutive_failures = 0
    for query in build_focus_event_queries(meetings, variables, pre_days=pre_days, post_days=post_days):
        LOGGER.info(
            "Fetching Focus %s %s from %s to %s.",
            query["variable"],
            query["reference_year"],
            query["start_date"].date(),
            query["end_date"].date(),
        )
        frame = _fetch_focus_query(client, query)
        if frame.attrs.get("fetch_failed"):
            consecutive_failures += 1
            if consecutive_failures >= max_consecutive_failures:
                LOGGER.warning("Stopping Focus OData fetch after %s consecutive failures.", consecutive_failures)
                break
        else:
            consecutive_failures = 0
        frames.append(frame)
    if not frames:
        return empty_focus_observations()
    non_empty_frames = [frame for frame in frames if not frame.empty]
    if not non_empty_frames:
        return empty_focus_observations()
    observations = pd.concat(non_empty_frames, ignore_index=True)
    observations = normalize_focus_observations(observations)
    return observations


def build_focus_event_queries(
    meetings: pd.DataFrame,
    variables: list[str],
    pre_days: int = 30,
    post_days: int = 14,
) -> list[dict[str, Any]]:
    queries: dict[str, dict[str, Any]] = {}
    for _, meeting in meetings.iterrows():
        meeting_date = pd.Timestamp(meeting["data_referencia"])
        end_event = meeting.get("data_ata", pd.NaT)
        if pd.isna(end_event):
            end_event = meeting.get("data_comunicado", meeting_date)
        end_event = pd.Timestamp(end_event)
        start_date = meeting_date - pd.Timedelta(days=pre_days)
        end_date = end_event + pd.Timedelta(days=post_days)
        for variable in variables:
            for reference_year in [meeting_date.year, meeting_date.year + 1]:
                signature = _query_signature(variable, int(reference_year), start_date, end_date)
                if signature not in queries:
                    queries[signature] = {
                        "variable": variable,
                        "reference_year": int(reference_year),
                        "start_date": start_date.normalize(),
                        "end_date": end_date.normalize(),
                        "query_signature": signature,
                    }
    return list(queries.values())


def fetch_focus_observations_backfill(
    client: CachedHttpClient,
    meetings: pd.DataFrame,
    variables: list[str],
    pre_days: int = 30,
    post_days: int = 14,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for query in build_focus_backfill_queries(meetings, variables, pre_days=pre_days, post_days=post_days):
        LOGGER.info(
            "Fetching Focus V2.1 %s %s from %s to %s.",
            query["variable"],
            query["reference_year"],
            query["start_date"].date(),
            query["end_date"].date(),
        )
        frame = _fetch_focus_query(client, query)
        frames.append(frame)
    non_empty_frames = [frame for frame in frames if not frame.empty]
    if not non_empty_frames:
        return empty_focus_observations()
    return normalize_focus_observations(pd.concat(non_empty_frames, ignore_index=True))


def build_focus_backfill_queries(
    meetings: pd.DataFrame,
    variables: list[str],
    pre_days: int = 30,
    post_days: int = 14,
) -> list[dict[str, Any]]:
    if meetings.empty:
        return []
    frame = meetings.copy()
    frame["data_referencia"] = pd.to_datetime(frame["data_referencia"], errors="coerce")
    frame = frame[frame["data_referencia"].notna()].copy()
    if frame.empty:
        return []
    start_date = max(frame["data_referencia"].min() - pd.Timedelta(days=pre_days), FOCUS_ODATA_MIN_DATE)
    end_candidates = []
    for column in ["data_ata", "data_comunicado", "data_referencia"]:
        if column in frame:
            end_candidates.append(pd.to_datetime(frame[column], errors="coerce").max())
    end_date = max([candidate for candidate in end_candidates if pd.notna(candidate)] or [frame["data_referencia"].max()])
    end_date = pd.Timestamp(end_date) + pd.Timedelta(days=post_days)
    reference_years = sorted(set(frame["data_referencia"].dt.year.dropna().astype(int)) | set((frame["data_referencia"].dt.year + 1).dropna().astype(int)))
    queries = []
    for variable in variables:
        for reference_year in reference_years:
            if reference_year < FOCUS_ODATA_MIN_DATE.year:
                continue
            signature = _query_signature(variable, int(reference_year), start_date, end_date)
            queries.append(
                {
                    "variable": variable,
                    "reference_year": int(reference_year),
                    "start_date": start_date.normalize(),
                    "end_date": end_date.normalize(),
                    "query_signature": signature,
                }
            )
    return queries


def normalize_focus_observations(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return empty_focus_observations()
    renamed = frame.rename(
        columns={
            "Indicador": "variable",
            "Data": "date",
            "DataReferencia": "reference_year",
            "Mediana": "median",
            "Media": "mean",
            "DesvioPadrao": "std",
            "Minimo": "minimum",
            "Maximo": "maximum",
            "sourceDate": "source_date",
        }
    ).copy()
    if "date" not in renamed and "source_date" in renamed:
        renamed["date"] = renamed["source_date"]
    for column in FOCUS_OBSERVATION_COLUMNS:
        if column not in renamed:
            renamed[column] = np.nan
    renamed["variable"] = renamed["variable"].astype(str).str.strip()
    renamed["reference_year"] = pd.to_numeric(renamed["reference_year"], errors="coerce").astype("Int64")
    renamed["date"] = pd.to_datetime(renamed["date"], errors="coerce")
    renamed["source_date"] = pd.to_datetime(renamed["source_date"], errors="coerce")
    for column in ["median", "mean", "std", "minimum", "maximum"]:
        renamed[column] = pd.to_numeric(renamed[column], errors="coerce")
    renamed["source"] = renamed["source"].fillna("focus_odata").astype(str)
    renamed["fetched_at"] = pd.to_datetime(renamed["fetched_at"], errors="coerce")
    if renamed["fetched_at"].isna().any():
        renamed.loc[renamed["fetched_at"].isna(), "fetched_at"] = pd.Timestamp.now(tz=timezone.utc).tz_localize(None)
    renamed["query_signature"] = renamed["query_signature"].fillna("").astype(str)
    normalized = renamed[FOCUS_OBSERVATION_COLUMNS].dropna(subset=["variable", "reference_year", "date", "median"])
    normalized = normalized.drop_duplicates(["variable", "reference_year", "date", "source", "query_signature"])
    normalized = normalized.sort_values(["variable", "reference_year", "date", "source"]).reset_index(drop=True)
    return normalized


def empty_focus_observations() -> pd.DataFrame:
    return pd.DataFrame(columns=FOCUS_OBSERVATION_COLUMNS)


def build_focus_vintages_from_observations(observations: pd.DataFrame) -> pd.DataFrame:
    observations = normalize_focus_observations(observations)
    if observations.empty:
        return empty_focus_vintages()
    rows: list[dict[str, Any]] = []
    statistic_columns = {
        "median": "median",
        "mean": "mean",
        "std": "std",
        "minimum": "minimum",
        "maximum": "maximum",
    }
    for _, row in observations.iterrows():
        reference_year = int(row["reference_year"])
        reference_date = pd.Timestamp(year=reference_year, month=12, day=31)
        for statistic, column in statistic_columns.items():
            value = row.get(column)
            if pd.isna(value):
                continue
            rows.append(
                {
                    "focus_release_date": row["date"],
                    "focus_reference_date": reference_date,
                    "indicator": row["variable"],
                    "reference_year": reference_year,
                    "horizon": f"year_{reference_year}",
                    "statistic": statistic,
                    "value": float(value),
                    "source": row["source"],
                    "source_date": row["source_date"],
                    "collected_at": row["fetched_at"],
                    "query_signature": row["query_signature"],
                    "data_access_tier": focus_data_access_tier(row["source"]),
                }
            )
    if not rows:
        return empty_focus_vintages()
    vintages = pd.DataFrame(rows, columns=FOCUS_VINTAGE_COLUMNS)
    vintages["focus_release_date"] = pd.to_datetime(vintages["focus_release_date"], errors="coerce")
    vintages["focus_reference_date"] = pd.to_datetime(vintages["focus_reference_date"], errors="coerce")
    vintages["source_date"] = pd.to_datetime(vintages["source_date"], errors="coerce")
    vintages["collected_at"] = pd.to_datetime(vintages["collected_at"], errors="coerce")
    vintages = vintages.drop_duplicates(
        ["focus_release_date", "indicator", "reference_year", "statistic", "source", "query_signature"],
        keep="last",
    )
    return vintages.sort_values(["indicator", "reference_year", "statistic", "focus_release_date"]).reset_index(drop=True)


def empty_focus_vintages() -> pd.DataFrame:
    return pd.DataFrame(columns=FOCUS_VINTAGE_COLUMNS)


def build_focus_event_features(
    events: pd.DataFrame,
    vintages: pd.DataFrame,
    post_days: int = 14,
) -> pd.DataFrame:
    vintages = normalize_focus_vintages(vintages)
    events = normalize_focus_events(events)
    if events.empty or vintages.empty:
        return empty_focus_event_features()
    rows: list[dict[str, Any]] = []
    for _, event in events.dropna(subset=["event_date"]).iterrows():
        event_date = pd.Timestamp(event["event_date"]).normalize()
        event_year = int(event_date.year)
        for reference_year, horizon in [(event_year, "current_year"), (event_year + 1, "next_year")]:
            subset_year = vintages[vintages["reference_year"] == reference_year].copy()
            if subset_year.empty:
                continue
            for (indicator, statistic), subset in subset_year.groupby(["indicator", "statistic"], dropna=False):
                selected = select_focus_event_values(subset, event_date, post_days=post_days)
                missing_reason = focus_event_missing_reason(selected, post_days)
                rows.append(
                    {
                        "meeting_id": event["meeting_id"],
                        "nro_reuniao": event.get("nro_reuniao", np.nan),
                        "event_type": event["event_type"],
                        "event_date": event_date,
                        "known_at_timestamp": event["known_at_timestamp"],
                        "indicator": indicator,
                        "reference_year": reference_year,
                        "horizon": horizon,
                        "statistic": statistic,
                        "pre_date": selected["pre_date"],
                        "pre_value": selected["pre_value"],
                        "post_1_date": selected["post_1_date"],
                        "post_1_value": selected["post_1_value"],
                        "post_2_date": selected["post_2_date"],
                        "post_2_value": selected["post_2_value"],
                        "delta_post_1": _safe_delta(selected["post_1_value"], selected["pre_value"]),
                        "delta_post_2": _safe_delta(selected["post_2_value"], selected["pre_value"]),
                        "missing_reason": missing_reason,
                        "source": selected["source"],
                        "data_access_tier": selected["data_access_tier"],
                    }
                )
    if not rows:
        return empty_focus_event_features()
    features = pd.DataFrame(rows, columns=FOCUS_EVENT_FEATURE_COLUMNS)
    for column in ["event_date", "known_at_timestamp", "pre_date", "post_1_date", "post_2_date"]:
        features[column] = pd.to_datetime(features[column], errors="coerce")
    return features.sort_values(["meeting_id", "event_type", "indicator", "horizon", "statistic"]).reset_index(drop=True)


def empty_focus_event_features() -> pd.DataFrame:
    return pd.DataFrame(columns=FOCUS_EVENT_FEATURE_COLUMNS)


def normalize_focus_vintages(vintages: pd.DataFrame) -> pd.DataFrame:
    if vintages.empty:
        return empty_focus_vintages()
    frame = vintages.copy()
    for column in FOCUS_VINTAGE_COLUMNS:
        if column not in frame:
            frame[column] = np.nan
    frame["focus_release_date"] = pd.to_datetime(frame["focus_release_date"], errors="coerce")
    frame["focus_reference_date"] = pd.to_datetime(frame["focus_reference_date"], errors="coerce")
    frame["indicator"] = frame["indicator"].astype(str).str.strip()
    frame["reference_year"] = pd.to_numeric(frame["reference_year"], errors="coerce").astype("Int64")
    frame["horizon"] = frame["horizon"].fillna("").astype(str)
    frame["statistic"] = frame["statistic"].fillna("median").astype(str).str.lower()
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame["source"] = frame["source"].fillna("").astype(str)
    frame["source_date"] = pd.to_datetime(frame["source_date"], errors="coerce")
    frame["collected_at"] = pd.to_datetime(frame["collected_at"], errors="coerce")
    frame["query_signature"] = frame["query_signature"].fillna("").astype(str)
    frame["data_access_tier"] = frame["data_access_tier"].fillna("").astype(str)
    frame.loc[frame["data_access_tier"].str.strip() == "", "data_access_tier"] = frame["source"].map(focus_data_access_tier)
    normalized = frame[FOCUS_VINTAGE_COLUMNS].dropna(subset=["focus_release_date", "indicator", "reference_year", "statistic", "value"])
    return normalized.reset_index(drop=True)


def normalize_focus_events(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(columns=["meeting_id", "nro_reuniao", "event_type", "event_date", "known_at_timestamp"])
    frame = events.copy()
    if "event_type" not in frame:
        frame["event_type"] = frame.get("document_type", "")
    if "event_date" not in frame:
        if "release_date" in frame:
            frame["event_date"] = frame["release_date"]
        elif "data_referencia" in frame:
            frame["event_date"] = frame["data_referencia"]
        else:
            frame["event_date"] = pd.NaT
    if "known_at_timestamp" not in frame:
        frame["known_at_timestamp"] = frame["event_date"]
    frame["event_date"] = pd.to_datetime(frame["event_date"], errors="coerce").dt.normalize()
    frame["known_at_timestamp"] = pd.to_datetime(frame["known_at_timestamp"], errors="coerce")
    missing_known = frame["known_at_timestamp"].isna() & frame["event_date"].notna()
    frame.loc[missing_known, "known_at_timestamp"] = frame.loc[missing_known, "event_date"]
    for column in ["meeting_id", "event_type"]:
        if column not in frame:
            frame[column] = ""
        frame[column] = frame[column].astype(str)
    if "nro_reuniao" not in frame:
        frame["nro_reuniao"] = np.nan
    return frame[["meeting_id", "nro_reuniao", "event_type", "event_date", "known_at_timestamp"]].drop_duplicates()


def select_focus_event_values(subset: pd.DataFrame, event_date: pd.Timestamp, post_days: int = 14) -> dict[str, Any]:
    subset = subset.sort_values("focus_release_date").copy()
    pre = subset[subset["focus_release_date"] < event_date]
    post = subset[
        (subset["focus_release_date"] > event_date)
        & (subset["focus_release_date"] <= event_date + pd.Timedelta(days=post_days))
    ]
    selected_pre = pre.iloc[-1] if not pre.empty else None
    selected_post_1 = post.iloc[0] if len(post) >= 1 else None
    selected_post_2 = post.iloc[1] if len(post) >= 2 else None
    source_row = selected_post_1 if selected_post_1 is not None else selected_pre
    return {
        "pre_date": selected_pre["focus_release_date"] if selected_pre is not None else pd.NaT,
        "pre_value": selected_pre["value"] if selected_pre is not None else np.nan,
        "post_1_date": selected_post_1["focus_release_date"] if selected_post_1 is not None else pd.NaT,
        "post_1_value": selected_post_1["value"] if selected_post_1 is not None else np.nan,
        "post_2_date": selected_post_2["focus_release_date"] if selected_post_2 is not None else pd.NaT,
        "post_2_value": selected_post_2["value"] if selected_post_2 is not None else np.nan,
        "source": source_row["source"] if source_row is not None else "",
        "data_access_tier": source_row["data_access_tier"] if source_row is not None else "",
    }


def focus_event_missing_reason(selected: dict[str, Any], post_days: int) -> str:
    missing = []
    if pd.isna(selected["pre_value"]):
        missing.append("missing_pre_event")
    if pd.isna(selected["post_1_value"]):
        missing.append(f"missing_post_event_1_within_{post_days}_days")
    if pd.isna(selected["post_2_value"]):
        missing.append(f"missing_post_event_2_within_{post_days}_days")
    return "|".join(missing)


def focus_data_access_tier(source: Any) -> str:
    source_text = str(source).lower()
    if "odata" in source_text:
        return "PUBLIC_API"
    if "fallback" in source_text or "snapshot" in source_text:
        return "MANUAL_UPLOAD"
    if not source_text or source_text == "nan":
        return "PUBLIC_API"
    return "DERIVED"


def build_focus_revisions_from_observations(
    meetings: pd.DataFrame,
    observations: pd.DataFrame,
    variables: list[str],
    post_days: int = 14,
) -> pd.DataFrame:
    observations = normalize_focus_observations(observations)
    rows: list[dict[str, Any]] = []
    for _, meeting in meetings.iterrows():
        meeting_date = pd.Timestamp(meeting["data_referencia"])
        comunicado_date = pd.Timestamp(meeting.get("data_comunicado", meeting_date))
        ata_date = meeting.get("data_ata", pd.NaT)
        for variable in variables:
            for reference_year in [meeting_date.year, meeting_date.year + 1]:
                subset = observations[
                    (observations["variable"].str.lower() == str(variable).lower())
                    & (observations["reference_year"] == int(reference_year))
                ].copy()
                row = _empty_focus_revision(meeting["meeting_id"], variable, int(reference_year))
                row.update(_pick_focus_observation(subset, "pre", before=meeting_date))
                row.update(_pick_focus_observation(subset, "post_comunicado", after=comunicado_date, max_days=post_days))
                if pd.notna(ata_date):
                    row.update(_pick_focus_observation(subset, "post_ata", after=pd.Timestamp(ata_date), max_days=post_days))
                else:
                    row["post_ata_missing_reason"] = "missing_event_date"
                row["delta_post_comunicado"] = _safe_delta(row["focus_post_comunicado_value"], row["focus_pre_value"])
                row["delta_post_ata"] = _safe_delta(row["focus_post_ata_value"], row["focus_pre_value"])
                rows.append(row)
    return pd.DataFrame(rows)


def audit_focus_coverage(focus_revisions: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "variable",
        "reference_year",
        "rows",
        "pre_coverage",
        "post_comunicado_delta_coverage",
        "post_ata_delta_coverage",
        "any_delta_coverage",
        "missing_pre",
        "missing_post_comunicado",
        "missing_post_ata",
        "status",
    ]
    if focus_revisions.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for (variable, reference_year), group in focus_revisions.groupby(["variable", "reference_year"], dropna=False):
        total = len(group)
        pre = float(group["focus_pre_value"].notna().mean()) if total else 0.0
        post_com = float(group["delta_post_comunicado"].notna().mean()) if total else 0.0
        post_ata = float(group["delta_post_ata"].notna().mean()) if total else 0.0
        any_delta = float(group[["delta_post_comunicado", "delta_post_ata"]].notna().any(axis=1).mean()) if total else 0.0
        status = coverage_status(any_delta)
        rows.append(
            {
                "variable": variable,
                "reference_year": int(reference_year) if pd.notna(reference_year) else np.nan,
                "rows": total,
                "pre_coverage": pre,
                "post_comunicado_delta_coverage": post_com,
                "post_ata_delta_coverage": post_ata,
                "any_delta_coverage": any_delta,
                "missing_pre": int(group["focus_pre_value"].isna().sum()),
                "missing_post_comunicado": int(group["delta_post_comunicado"].isna().sum()),
                "missing_post_ata": int(group["delta_post_ata"].isna().sum()),
                "status": status,
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values(["variable", "reference_year"]).reset_index(drop=True)


def build_focus_missing_events(focus_revisions: pd.DataFrame) -> pd.DataFrame:
    if focus_revisions.empty:
        return pd.DataFrame()
    missing = focus_revisions[
        focus_revisions[["focus_pre_value", "delta_post_comunicado", "delta_post_ata"]].isna().any(axis=1)
    ].copy()
    columns = [
        "meeting_id",
        "variable",
        "reference_year",
        "pre_missing_reason",
        "post_comunicado_missing_reason",
        "post_ata_missing_reason",
    ]
    return missing[[column for column in columns if column in missing.columns]]


def coverage_status(coverage: float) -> str:
    if coverage >= 0.80:
        return "ok"
    if coverage >= 0.40:
        return "limited_data"
    return "invalid_for_inference"


def overall_focus_coverage_status(coverage: pd.DataFrame) -> str:
    if coverage.empty:
        return "invalid_for_inference"
    statuses = set(coverage["status"].dropna())
    if "invalid_for_inference" in statuses:
        return "invalid_for_inference"
    if "limited_data" in statuses:
        return "limited_data"
    return "ok"


def import_focus_snapshot(path: Path, source_date: str | pd.Timestamp) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        frame = pd.read_excel(path)
    elif path.suffix.lower() in {".csv", ".txt"}:
        frame = pd.read_csv(path, sep=None, engine="python")
    else:
        raise ValueError("Focus snapshot import supports CSV/TXT or Excel files. Convert PDFs before importing.")
    frame = _normalize_snapshot_columns(frame)
    source_timestamp = pd.Timestamp(source_date)
    if "date" not in frame or frame["date"].isna().all():
        frame["date"] = source_timestamp
    frame["source_date"] = source_timestamp
    frame["source"] = "focus_report_fallback"
    frame["fetched_at"] = pd.Timestamp.now(tz=timezone.utc).tz_localize(None)
    frame["query_signature"] = "manual_snapshot_" + _hash_text(str(path.resolve()) + str(source_timestamp.date()))
    return normalize_focus_observations(frame)


def fetch_focus_command(months: int | None = None) -> FocusCommandResult:
    settings = load_settings()
    paths = get_paths(settings)
    months = months or int(settings["pipeline"]["months"])
    meetings = read_table(paths.database, "copom_meetings")
    meetings = _filter_meetings_by_months(meetings, months)
    client = CachedHttpClient(
        cache_dir=paths.raw / "focus",
        timeout_seconds=int(settings["pipeline"]["request_timeout_seconds"]),
        retries=int(settings["pipeline"]["retries"]),
        retry_backoff_seconds=float(settings["pipeline"]["retry_backoff_seconds"]),
    )
    observations = fetch_focus_observations_for_meetings(client, meetings, settings["pipeline"]["focus_variables"])
    observations = merge_existing_focus_observations(paths.database, observations)
    revisions = build_focus_revisions_from_observations(meetings, observations, settings["pipeline"]["focus_variables"])
    write_tables(paths.database, {"focus_observations": observations, "focus_revisions": revisions})
    export_tables(paths.database, paths.processed, ["focus_observations", "focus_revisions"])
    coverage = write_focus_coverage_outputs(revisions)
    return FocusCommandResult(
        observations=len(observations),
        revisions=len(revisions),
        coverage_status=overall_focus_coverage_status(coverage),
        output_path=paths.processed / "focus_observations.csv",
    )


def rebuild_focus_revisions_command(months: int | None = None) -> FocusCommandResult:
    settings = load_settings()
    paths = get_paths(settings)
    meetings = read_table(paths.database, "copom_meetings")
    if months:
        meetings = _filter_meetings_by_months(meetings, months)
    observations = read_optional_table(paths.database, "focus_observations", empty_focus_observations())
    revisions = build_focus_revisions_from_observations(meetings, observations, settings["pipeline"]["focus_variables"])
    write_tables(paths.database, {"focus_revisions": revisions})
    export_tables(paths.database, paths.processed, ["focus_revisions"])
    coverage = write_focus_coverage_outputs(revisions)
    return FocusCommandResult(
        observations=len(observations),
        revisions=len(revisions),
        coverage_status=overall_focus_coverage_status(coverage),
        output_path=paths.processed / "focus_revisions.csv",
    )


def import_focus_snapshot_command(path: Path, source_date: str) -> FocusCommandResult:
    settings = load_settings()
    paths = get_paths(settings)
    imported = import_focus_snapshot(Path(path), source_date)
    observations = merge_existing_focus_observations(paths.database, imported)
    write_tables(paths.database, {"focus_observations": observations})
    export_tables(paths.database, paths.processed, ["focus_observations"])
    rebuild = rebuild_focus_revisions_command()
    return FocusCommandResult(
        observations=len(observations),
        revisions=rebuild.revisions,
        coverage_status=rebuild.coverage_status,
        output_path=paths.processed / "focus_observations.csv",
    )


def focus_coverage_command() -> FocusCommandResult:
    paths = get_paths()
    revisions = read_table(paths.database, "focus_revisions")
    coverage = write_focus_coverage_outputs(revisions)
    return FocusCommandResult(
        observations=0,
        revisions=len(revisions),
        coverage_status=overall_focus_coverage_status(coverage),
        output_path=paths.processed.parent / "econometrics" / "focus_coverage_audit.csv",
    )


def focus_v21_refresh_command(months: int = 400) -> FocusV21CommandResult:
    settings = load_settings()
    paths = get_paths(settings)
    meetings = read_v21_meetings(paths.database)
    if months:
        meetings = _filter_meetings_by_months(meetings, months)
    variables = focus_v21_variables(settings)
    client = CachedHttpClient(
        cache_dir=paths.raw / "focus",
        timeout_seconds=int(settings["pipeline"]["request_timeout_seconds"]),
        retries=int(settings["pipeline"]["retries"]),
        retry_backoff_seconds=float(settings["pipeline"]["retry_backoff_seconds"]),
    )
    fetched = fetch_focus_observations_backfill(client, meetings, variables)
    observations = merge_existing_focus_observations(paths.database, fetched)
    legacy_variables = list(dict.fromkeys(settings["pipeline"].get("focus_variables", ["Selic", "IPCA"])))
    revisions = build_focus_revisions_from_observations(meetings, observations, legacy_variables)
    event_calendar = read_optional_table(paths.database, "event_calendar", pd.DataFrame())
    events = event_calendar if not event_calendar.empty else fallback_focus_events_from_meetings(meetings)
    vintages = build_focus_vintages_from_observations(observations)
    features = build_focus_event_features(events, vintages)
    coverage = audit_focus_v21_coverage(features)
    write_tables(
        paths.database,
        {
            "focus_observations": observations,
            "focus_revisions": revisions,
            "focus_vintages": vintages,
            "focus_event_features": features,
        },
    )
    export_tables(paths.database, paths.processed, ["focus_observations", "focus_revisions", "focus_vintages", "focus_event_features"])
    write_focus_coverage_outputs(revisions)
    output_path = write_focus_v21_outputs(vintages, features, coverage)
    return FocusV21CommandResult(
        vintages=len(vintages),
        event_features=len(features),
        coverage_status=overall_focus_v21_status(coverage),
        output_path=output_path,
    )


def focus_v21_audit_command() -> FocusV21CommandResult:
    paths = get_paths()
    observations = read_optional_table(paths.database, "focus_observations", empty_focus_observations())
    vintages = read_optional_table(paths.database, "focus_vintages", empty_focus_vintages())
    if vintages.empty and not observations.empty:
        vintages = build_focus_vintages_from_observations(observations)
    features = read_optional_table(paths.database, "focus_event_features", empty_focus_event_features())
    if features.empty and not vintages.empty:
        events = read_optional_table(paths.database, "event_calendar", pd.DataFrame())
        if events.empty:
            events = fallback_focus_events_from_meetings(read_v21_meetings(paths.database))
        features = build_focus_event_features(events, vintages)
    coverage = audit_focus_v21_coverage(features)
    write_tables(paths.database, {"focus_vintages": vintages, "focus_event_features": features})
    export_tables(paths.database, paths.processed, ["focus_vintages", "focus_event_features"])
    output_path = write_focus_v21_outputs(vintages, features, coverage)
    return FocusV21CommandResult(
        vintages=len(vintages),
        event_features=len(features),
        coverage_status=overall_focus_v21_status(coverage),
        output_path=output_path,
    )


def audit_focus_v21_coverage(features: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "indicator",
        "horizon",
        "statistic",
        "event_type",
        "rows",
        "pre_coverage",
        "post_1_delta_coverage",
        "post_2_delta_coverage",
        "any_delta_coverage",
        "status",
    ]
    if features.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for keys, group in features.groupby(["indicator", "horizon", "statistic", "event_type"], dropna=False):
        indicator, horizon, statistic, event_type = keys
        total = len(group)
        pre = float(group["pre_value"].notna().mean()) if total else 0.0
        post_1 = float(group["delta_post_1"].notna().mean()) if total else 0.0
        post_2 = float(group["delta_post_2"].notna().mean()) if total else 0.0
        any_delta = float(group[["delta_post_1", "delta_post_2"]].notna().any(axis=1).mean()) if total else 0.0
        rows.append(
            {
                "indicator": indicator,
                "horizon": horizon,
                "statistic": statistic,
                "event_type": event_type,
                "rows": total,
                "pre_coverage": pre,
                "post_1_delta_coverage": post_1,
                "post_2_delta_coverage": post_2,
                "any_delta_coverage": any_delta,
                "status": coverage_status(any_delta),
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values(["indicator", "horizon", "statistic", "event_type"]).reset_index(drop=True)


def overall_focus_v21_status(coverage: pd.DataFrame) -> str:
    if coverage.empty:
        return "invalid_for_inference"
    statuses = set(coverage["status"].dropna())
    if "ok" in statuses:
        return "ok" if statuses <= {"ok"} else "limited_data"
    if "limited_data" in statuses:
        return "limited_data"
    return "invalid_for_inference"


def write_focus_v21_outputs(vintages: pd.DataFrame, features: pd.DataFrame, coverage: pd.DataFrame) -> Path:
    paths = get_paths()
    output_dir = paths.processed.parent / "v2"
    report_dir = paths.reports.parent / "v2"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    vintages.to_csv(output_dir / "focus_vintages.csv", index=False)
    features.to_csv(output_dir / "focus_event_features.csv", index=False)
    coverage.to_csv(output_dir / "focus_v21_coverage.csv", index=False)
    vintages.to_parquet(output_dir / "focus_vintages.parquet", index=False)
    features.to_parquet(output_dir / "focus_event_features.parquet", index=False)
    coverage.to_parquet(output_dir / "focus_v21_coverage.parquet", index=False)
    report_path = report_dir / "focus_v21_report.html"
    report_path.write_text(render_focus_v21_report_html(vintages, features, coverage), encoding="utf-8")
    return output_dir / "focus_event_features.csv"


def write_focus_coverage_outputs(focus_revisions: pd.DataFrame) -> pd.DataFrame:
    paths = get_paths()
    output_dir = paths.processed.parent / "econometrics"
    output_dir.mkdir(parents=True, exist_ok=True)
    coverage = audit_focus_coverage(focus_revisions)
    missing = build_focus_missing_events(focus_revisions)
    coverage.to_csv(output_dir / "focus_coverage_audit.csv", index=False)
    coverage.to_parquet(output_dir / "focus_coverage_audit.parquet", index=False)
    missing.to_csv(output_dir / "focus_missing_events.csv", index=False)
    missing.to_parquet(output_dir / "focus_missing_events.parquet", index=False)
    return coverage


def render_focus_v21_report_html(vintages: pd.DataFrame, features: pd.DataFrame, coverage: pd.DataFrame) -> str:
    status = overall_focus_v21_status(coverage)
    coverage_html = coverage.to_html(index=False, escape=True) if not coverage.empty else "<p>Sem cobertura Focus V2.1.</p>"
    recent_features = (
        features.sort_values(["event_date", "meeting_id"], ascending=[False, True])
        .head(30)
        .to_html(index=False, escape=True)
        if not features.empty
        else "<p>Sem features de evento Focus.</p>"
    )
    return "\n".join(
        [
            "<!doctype html><html><head><meta charset='utf-8'>",
            "<title>COPOM Watch V2.1 Focus Report</title>",
            "<style>body{font-family:Arial,sans-serif;margin:32px;line-height:1.45;color:#111827}table{border-collapse:collapse;width:100%;margin:12px 0}td,th{border:1px solid #ddd;padding:6px}th{background:#f3f4f6}.badge{display:inline-block;padding:6px 10px;border-radius:4px;background:#0f766e;color:white;font-weight:bold}.muted{color:#6b7280}</style>",
            "</head><body>",
            "<h1>COPOM Watch V2.1 Focus Report</h1>",
            f"<p><span class='badge'>{html.escape(status)}</span></p>",
            "<p>Focus V2.1 usa disciplina de vintage: observacoes pre-evento e pos-evento sao selecionadas apenas por datas observaveis, sem forward-fill para deltas.</p>",
            f"<p>Vintages: <strong>{len(vintages)}</strong>. Event features: <strong>{len(features)}</strong>.</p>",
            "<h2>Cobertura</h2>",
            coverage_html,
            "<h2>Features recentes</h2>",
            recent_features,
            "</body></html>",
        ]
    )


def read_v21_meetings(database: Path) -> pd.DataFrame:
    meetings = read_optional_table(database, "v2_meetings", pd.DataFrame())
    if meetings.empty:
        meetings = read_optional_table(database, "copom_meetings", pd.DataFrame())
    return meetings.copy()


def fallback_focus_events_from_meetings(meetings: pd.DataFrame) -> pd.DataFrame:
    if meetings.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for _, meeting in meetings.iterrows():
        for event_type, date_column, hour in [
            ("comunicado", "data_comunicado", 18.5),
            ("ata", "data_ata", 8.0),
        ]:
            event_date = pd.to_datetime(meeting.get(date_column), errors="coerce")
            if pd.isna(event_date):
                continue
            hours = int(hour)
            minutes = int(round((hour - hours) * 60))
            rows.append(
                {
                    "meeting_id": meeting.get("meeting_id", ""),
                    "nro_reuniao": meeting.get("nro_reuniao", np.nan),
                    "event_type": event_type,
                    "event_date": event_date.normalize(),
                    "known_at_timestamp": event_date.normalize() + pd.Timedelta(hours=hours, minutes=minutes),
                }
            )
    return pd.DataFrame(rows)


def focus_v21_variables(settings: dict[str, Any]) -> list[str]:
    base = list(settings.get("pipeline", {}).get("focus_variables", []))
    return list(dict.fromkeys([*base, *FOCUS_V21_VARIABLES]))


def merge_existing_focus_observations(database: Path, new_observations: pd.DataFrame) -> pd.DataFrame:
    existing = read_optional_table(database, "focus_observations", empty_focus_observations())
    merged = pd.concat([existing, new_observations], ignore_index=True)
    return normalize_focus_observations(merged)


def read_optional_table(database: Path, table: str, default: pd.DataFrame) -> pd.DataFrame:
    if not Path(database).exists():
        return default.copy()
    with duckdb.connect(str(database), read_only=True) as con:
        exists = con.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
            [table],
        ).fetchone()[0]
        if not exists:
            return default.copy()
        return con.execute(f"SELECT * FROM {table}").df()


def _fetch_focus_query(client: CachedHttpClient, query: dict[str, Any]) -> pd.DataFrame:
    variable = query["variable"]
    reference_year = int(query["reference_year"])
    start_date = pd.Timestamp(query["start_date"])
    end_date = pd.Timestamp(query["end_date"])
    query_signature = query["query_signature"]
    params = {
        "$format": "json",
        "$select": "Indicador,Data,DataReferencia,Media,Mediana,DesvioPadrao,Minimo,Maximo",
        "$filter": (
            f"Indicador eq '{variable}' and DataReferencia eq '{reference_year}' "
            f"and Data ge '{start_date.date()}' and Data le '{end_date.date()}'"
        ),
        "$orderby": "Data asc",
        "$top": 5000,
    }
    cache_name = f"{variable.lower()}_{reference_year}_{start_date.date()}_{end_date.date()}_{query_signature}"
    try:
        if isinstance(client, CachedHttpClient):
            request_url = BCB_FOCUS_ANNUAL + "?" + urlencode(params, quote_via=quote, safe="$,")
            data = client.get_json(request_url, cache_name)
        else:
            data = client.get_json(BCB_FOCUS_ANNUAL, cache_name, params)
    except FetchError as exc:
        LOGGER.warning("Focus query failed for %s %s: %s", variable, reference_year, exc)
        frame = empty_focus_observations()
        frame.attrs["fetch_failed"] = True
        return frame
    rows = data.get("value", []) if isinstance(data, dict) else []
    frame = pd.DataFrame(rows)
    if frame.empty:
        return empty_focus_observations()
    frame["source"] = "focus_odata"
    frame["source_date"] = pd.NaT
    frame["fetched_at"] = pd.Timestamp.now(tz=timezone.utc).tz_localize(None)
    frame["query_signature"] = query_signature
    return normalize_focus_observations(frame)


def _pick_focus_observation(
    subset: pd.DataFrame,
    label: str,
    before: pd.Timestamp | None = None,
    after: pd.Timestamp | None = None,
    max_days: int | None = None,
) -> dict[str, Any]:
    if subset.empty:
        return {f"{label}_missing_reason": "no_observations_for_variable_year"}
    if before is not None:
        eligible = subset[subset["date"] < before].sort_values("date")
        if eligible.empty:
            return {f"{label}_missing_reason": "no_observation_before_meeting"}
        selected = eligible.iloc[-1]
    elif after is not None:
        upper_bound = after + pd.Timedelta(days=max_days) if max_days is not None else pd.Timestamp.max
        eligible = subset[(subset["date"] > after) & (subset["date"] <= upper_bound)].sort_values("date")
        if eligible.empty:
            return {f"{label}_missing_reason": f"no_observation_within_{max_days}_days_after_event"}
        selected = eligible.iloc[0]
    else:
        return {f"{label}_missing_reason": "missing_selection_rule"}
    return {
        f"focus_{label}_date": selected["date"],
        f"focus_{label}_value": selected["median"],
        f"focus_{label}_source": selected["source"],
        f"{label}_missing_reason": "",
    }


def _empty_focus_revision(meeting_id: str, variable: str, reference_year: int) -> dict[str, Any]:
    return {
        "meeting_id": meeting_id,
        "variable": variable,
        "reference_year": reference_year,
        "focus_pre_date": pd.NaT,
        "focus_pre_value": np.nan,
        "focus_pre_source": "",
        "focus_post_comunicado_date": pd.NaT,
        "focus_post_comunicado_value": np.nan,
        "focus_post_comunicado_source": "",
        "focus_post_ata_date": pd.NaT,
        "focus_post_ata_value": np.nan,
        "focus_post_ata_source": "",
        "delta_post_comunicado": np.nan,
        "delta_post_ata": np.nan,
        "pre_missing_reason": "",
        "post_comunicado_missing_reason": "",
        "post_ata_missing_reason": "",
    }


def _safe_delta(post: Any, pre: Any) -> float:
    if pd.isna(post) or pd.isna(pre):
        return np.nan
    return float(post) - float(pre)


def _query_signature(variable: str, reference_year: int, start_date: pd.Timestamp, end_date: pd.Timestamp) -> str:
    payload = f"{variable}|{reference_year}|{start_date.date()}|{end_date.date()}"
    return _hash_text(payload)[:12]


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _normalize_snapshot_columns(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    normalized.columns = [str(column).strip() for column in normalized.columns]
    aliases = {
        "Indicador": "variable",
        "indicador": "variable",
        "Variable": "variable",
        "variable": "variable",
        "Data": "date",
        "data": "date",
        "Date": "date",
        "date": "date",
        "DataReferencia": "reference_year",
        "data_referencia": "reference_year",
        "reference_year": "reference_year",
        "Ano": "reference_year",
        "ano": "reference_year",
        "Mediana": "median",
        "mediana": "median",
        "Median": "median",
        "median": "median",
        "Media": "mean",
        "media": "mean",
        "DesvioPadrao": "std",
        "Minimo": "minimum",
        "Maximo": "maximum",
    }
    normalized = normalized.rename(columns={column: aliases.get(column, column) for column in normalized.columns})
    required = {"variable", "reference_year", "median"}
    missing = required - set(normalized.columns)
    if missing:
        raise ValueError(f"Focus snapshot missing required columns: {sorted(missing)}")
    return normalized


def _filter_meetings_by_months(meetings: pd.DataFrame, months: int) -> pd.DataFrame:
    if meetings.empty:
        return meetings
    meetings = meetings.copy()
    meetings["data_referencia"] = pd.to_datetime(meetings["data_referencia"])
    cutoff = meetings["data_referencia"].max() - pd.DateOffset(months=months)
    return meetings[meetings["data_referencia"] >= cutoff].reset_index(drop=True)

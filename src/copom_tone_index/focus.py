from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import timezone
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from copom_tone_index.config import get_paths, load_settings
from copom_tone_index.http_client import CachedHttpClient, FetchError
from copom_tone_index.storage import export_tables, read_table, write_tables

LOGGER = logging.getLogger(__name__)

BCB_FOCUS_ANNUAL = "https://olinda.bcb.gov.br/olinda/servico/Expectativas/versao/v1/odata/ExpectativasMercadoAnuais"
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


@dataclass(frozen=True)
class FocusCommandResult:
    observations: int
    revisions: int
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

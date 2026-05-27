from __future__ import annotations

import io
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlencode

import httpx
import numpy as np
import pandas as pd

from copom_tone_index.config import get_paths
from copom_tone_index.storage import export_tables, write_tables
from copom_tone_index.v2 import read_optional_table, utc_now_naive

MARKET_OBSERVATION_COLUMNS = [
    "asset",
    "asset_class",
    "vertex",
    "timestamp",
    "date",
    "value",
    "unit",
    "currency",
    "frequency",
    "market_timezone",
    "source",
    "source_file",
    "as_of_date",
    "data_access_tier",
    "license_note",
    "source_url",
    "source_method",
    "source_status",
    "is_proxy",
    "proxy_method",
    "collected_at",
]
DECISION_EXPECTATION_COLUMNS = [
    "meeting_id",
    "source",
    "data_access_tier",
    "as_of_timestamp",
    "expected_selic_change_bps",
    "scenario_selic_change_bps",
    "probability",
    "license_note",
    "source_file",
    "source_url",
    "source_method",
    "source_status",
    "is_proxy",
    "proxy_method",
    "collected_at",
]
PUBLIC_MARKET_AUDIT_COLUMNS = [
    "source",
    "status",
    "rows",
    "start_date",
    "end_date",
    "source_url",
    "detail",
]
SGS_SERIES = {
    "SELIC_META": {"code": 432, "asset_class": "rates", "vertex": "target", "unit": "percent_annual"},
    "SELIC_OVER": {"code": 11, "asset_class": "rates", "vertex": "overnight", "unit": "percent_daily"},
    "CDI": {"code": 12, "asset_class": "rates", "vertex": "overnight", "unit": "percent_daily"},
}
ANBIMA_TARGET_VERTICES = {
    "3m": 63,
    "6m": 126,
    "1y": 252,
    "2y": 504,
    "5y": 1260,
}
SGS_URL_TEMPLATE = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados"
PTAX_URL = "https://olinda.bcb.gov.br/olinda/servico/PTAX/versao/v1/odata/CotacaoMoedaPeriodo(moeda=@moeda,dataInicial=@dataInicial,dataFinalCotacao=@dataFinalCotacao)"
ANBIMA_INTRADAY_URL = "https://www.anbima.com.br/informacoes/curvas-intradiarias/CIntra.asp"
B3_COPOM_OPTION_URL = "https://www.b3.com.br/pt_br/produtos-e-servicos/negociacao/juros/opcao-de-copom.htm"
HTTP_RETRY_ATTEMPTS = 3
HTTP_RETRY_BACKOFF_SECONDS = 1.5


@dataclass(frozen=True)
class MarketCommandResult:
    database: Path
    output_path: Path | None
    rows: int
    status: str


def import_market_csv_command(
    path: str | Path,
    source: str = "user_csv",
    data_access_tier: str = "USER_CSV",
    license_note: str = "",
) -> MarketCommandResult:
    paths = get_paths()
    imported = import_market_csv(path, source=source, data_access_tier=data_access_tier, license_note=license_note)
    existing = read_optional_table(paths.database, "market_observations", empty_market_observations())
    merged = normalize_market_observations(concat_nonempty([existing, imported], empty_market_observations()))
    write_tables(paths.database, {"market_observations": merged})
    export_tables(paths.database, paths.processed, ["market_observations"])
    return MarketCommandResult(paths.database, paths.processed / "market_observations.csv", len(merged), "completed")


def run_event_study_command() -> MarketCommandResult:
    paths = get_paths()
    events = read_optional_table(paths.database, "event_calendar", pd.DataFrame())
    observations = read_optional_table(paths.database, "market_observations", empty_market_observations())
    windows = build_market_event_windows(events, observations)
    write_tables(paths.database, {"market_event_windows": windows})
    export_tables(paths.database, paths.processed, ["market_event_windows"])
    write_market_event_outputs(windows)
    status = "completed" if not windows.empty else "no_market_windows"
    return MarketCommandResult(paths.database, paths.processed / "market_event_windows.csv", len(windows), status)


def import_decision_expectations_command(
    path: str | Path,
    source: str = "user_csv",
    data_access_tier: str = "USER_CSV",
    license_note: str = "",
) -> MarketCommandResult:
    paths = get_paths()
    imported = import_decision_expectations(path, source=source, data_access_tier=data_access_tier, license_note=license_note)
    existing = read_optional_table(paths.database, "decision_expectations", empty_decision_expectations())
    merged = normalize_decision_expectations(concat_nonempty([existing, imported], empty_decision_expectations()))
    write_tables(paths.database, {"decision_expectations": merged})
    export_tables(paths.database, paths.processed, ["decision_expectations"])
    output_dir = paths.processed.parent / "v2"
    output_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_dir / "decision_expectations.csv", index=False)
    merged.to_parquet(output_dir / "decision_expectations.parquet", index=False)
    return MarketCommandResult(paths.database, output_dir / "decision_expectations.csv", len(merged), "completed")


def fetch_public_market_command(sources: str = "bcb-sgs,ptax,anbima", months: int = 400) -> MarketCommandResult:
    paths = get_paths()
    start_date, end_date = public_market_date_range(paths.database, months)
    source_list = [source.strip().lower() for source in sources.split(",") if source.strip()]
    frames: list[pd.DataFrame] = []
    audit_rows: list[dict[str, Any]] = []
    if "bcb-sgs" in source_list:
        frame, audit = fetch_bcb_sgs_market_observations(start_date, end_date)
        frames.append(frame)
        audit_rows.extend(audit)
    if "ptax" in source_list:
        frame, audit = fetch_ptax_market_observations(start_date, end_date)
        frames.append(frame)
        audit_rows.extend(audit)
    if "anbima" in source_list:
        frame, audit = fetch_anbima_market_observations(end_date)
        frames.append(frame)
        audit_rows.extend(audit)
    if "b3" in source_list or "b3-public" in source_list:
        audit_rows.append(public_audit_row("b3-public", "unavailable", 0, start_date, end_date, "No stable public DI futures adapter implemented yet.", B3_COPOM_OPTION_URL))
    public_observations = normalize_market_observations(pd.concat([frame for frame in frames if not frame.empty], ignore_index=True)) if frames else empty_market_observations()
    existing = read_optional_table(paths.database, "market_observations", empty_market_observations())
    merged = normalize_market_observations(concat_nonempty([existing, public_observations], empty_market_observations()))
    audit = pd.DataFrame(audit_rows, columns=PUBLIC_MARKET_AUDIT_COLUMNS)
    write_tables(paths.database, {"market_observations": merged, "public_market_source_audit": audit})
    export_tables(paths.database, paths.processed, ["market_observations", "public_market_source_audit"])
    output_path = write_public_market_outputs(public_observations, audit)
    status = "completed" if not public_observations.empty else "no_public_market_data"
    return MarketCommandResult(paths.database, output_path, len(public_observations), status)


def derive_decision_expectations_command(method: str = "public") -> MarketCommandResult:
    paths = get_paths()
    meetings = read_optional_table(paths.database, "v2_meetings", pd.DataFrame())
    focus_features = read_optional_table(paths.database, "focus_event_features", pd.DataFrame())
    market_observations = read_optional_table(paths.database, "market_observations", empty_market_observations())
    expectations, audit = derive_public_decision_expectations(meetings, focus_features, market_observations, method=method)
    existing = read_optional_table(paths.database, "decision_expectations", empty_decision_expectations())
    if not existing.empty and "source" in existing:
        generated_sources = {"focus_selic_proxy", "curve_short_rate_proxy"}
        existing = existing[~existing["source"].astype(str).isin(generated_sources)].copy()
    merged = normalize_decision_expectations(concat_nonempty([existing, expectations], empty_decision_expectations()))
    write_tables(paths.database, {"decision_expectations": merged, "decision_expectation_source_audit": audit})
    export_tables(paths.database, paths.processed, ["decision_expectations", "decision_expectation_source_audit"])
    output_path = write_decision_expectation_outputs(expectations, audit)
    status = "completed" if not expectations.empty else "no_public_decision_expectations"
    return MarketCommandResult(paths.database, output_path, len(expectations), status)


def public_market_coverage_command() -> MarketCommandResult:
    paths = get_paths()
    observations = read_optional_table(paths.database, "market_observations", empty_market_observations())
    windows = read_optional_table(paths.database, "market_event_windows", pd.DataFrame())
    expectations = read_optional_table(paths.database, "decision_expectations", empty_decision_expectations())
    market_audit = read_optional_table(paths.database, "public_market_source_audit", pd.DataFrame(columns=PUBLIC_MARKET_AUDIT_COLUMNS))
    decision_audit = read_optional_table(paths.database, "decision_expectation_source_audit", pd.DataFrame(columns=PUBLIC_MARKET_AUDIT_COLUMNS))
    coverage = build_public_market_coverage(observations, windows, expectations, market_audit, decision_audit)
    write_tables(paths.database, {"public_market_coverage": coverage})
    export_tables(paths.database, paths.processed, ["public_market_coverage"])
    output_path = write_public_market_coverage_report(coverage)
    return MarketCommandResult(paths.database, output_path, len(coverage), "completed")


def import_market_csv(
    path: str | Path,
    source: str = "user_csv",
    data_access_tier: str = "USER_CSV",
    license_note: str = "",
) -> pd.DataFrame:
    path = Path(path)
    frame = pd.read_csv(path)
    frame["source"] = frame.get("source", source)
    frame["source_file"] = str(path)
    frame["data_access_tier"] = frame.get("data_access_tier", data_access_tier)
    frame["license_note"] = frame.get("license_note", license_note)
    frame["collected_at"] = utc_now_naive()
    return normalize_market_observations(frame)


def import_decision_expectations(
    path: str | Path,
    source: str = "user_csv",
    data_access_tier: str = "USER_CSV",
    license_note: str = "",
) -> pd.DataFrame:
    path = Path(path)
    frame = pd.read_csv(path)
    frame["source"] = frame.get("source", source)
    frame["source_file"] = str(path)
    frame["data_access_tier"] = frame.get("data_access_tier", data_access_tier)
    frame["license_note"] = frame.get("license_note", license_note)
    frame["collected_at"] = utc_now_naive()
    return normalize_decision_expectations(frame)


def normalize_market_observations(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return empty_market_observations()
    renamed = frame.rename(
        columns={
            "datetime": "timestamp",
            "time": "timestamp",
            "price": "value",
            "rate": "value",
            "ticker": "asset",
            "tenor": "vertex",
        }
    ).copy()
    for column in MARKET_OBSERVATION_COLUMNS:
        if column not in renamed:
            renamed[column] = np.nan
    renamed["asset"] = renamed["asset"].astype(str).str.strip()
    renamed["asset_class"] = renamed["asset_class"].fillna("").astype(str).str.strip()
    renamed["vertex"] = renamed["vertex"].fillna("").astype(str).str.strip()
    renamed["timestamp"] = pd.to_datetime(renamed["timestamp"], errors="coerce")
    if renamed["timestamp"].isna().all() and "date" in renamed:
        renamed["timestamp"] = pd.to_datetime(renamed["date"], errors="coerce")
    renamed["date"] = pd.to_datetime(renamed["timestamp"], errors="coerce").dt.normalize()
    renamed["value"] = pd.to_numeric(renamed["value"], errors="coerce")
    for column in [
        "unit",
        "currency",
        "frequency",
        "market_timezone",
        "source",
        "source_file",
        "data_access_tier",
        "license_note",
        "source_url",
        "source_method",
        "source_status",
        "proxy_method",
    ]:
        renamed[column] = renamed[column].fillna("").astype(str)
    renamed["is_proxy"] = normalize_bool_series(renamed["is_proxy"])
    renamed["as_of_date"] = pd.to_datetime(renamed["as_of_date"], errors="coerce")
    renamed["collected_at"] = pd.to_datetime(renamed["collected_at"], errors="coerce")
    renamed.loc[renamed["collected_at"].isna(), "collected_at"] = utc_now_naive()
    normalized = renamed[MARKET_OBSERVATION_COLUMNS].dropna(subset=["asset", "timestamp", "value"])
    normalized = normalized.drop_duplicates(["asset", "vertex", "timestamp", "source_file"], keep="last")
    return normalized.sort_values(["asset", "vertex", "timestamp"]).reset_index(drop=True)


def empty_market_observations() -> pd.DataFrame:
    return pd.DataFrame(columns=MARKET_OBSERVATION_COLUMNS)


def normalize_decision_expectations(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return empty_decision_expectations()
    renamed = frame.rename(
        columns={
            "meeting": "meeting_id",
            "as_of": "as_of_timestamp",
            "timestamp": "as_of_timestamp",
            "expected_change_bps": "expected_selic_change_bps",
            "expected_selic_bps": "expected_selic_change_bps",
            "scenario_bps": "scenario_selic_change_bps",
            "prob": "probability",
        }
    ).copy()
    for column in DECISION_EXPECTATION_COLUMNS:
        if column not in renamed:
            renamed[column] = np.nan
    renamed["meeting_id"] = renamed["meeting_id"].astype(str).str.strip()
    for column in ["source", "data_access_tier", "license_note", "source_file", "source_url", "source_method", "source_status", "proxy_method"]:
        renamed[column] = renamed[column].fillna("").astype(str)
    renamed["is_proxy"] = normalize_bool_series(renamed["is_proxy"])
    renamed["as_of_timestamp"] = pd.to_datetime(renamed["as_of_timestamp"], errors="coerce")
    for column in ["expected_selic_change_bps", "scenario_selic_change_bps", "probability"]:
        renamed[column] = pd.to_numeric(renamed[column], errors="coerce")
    has_scenario = renamed["expected_selic_change_bps"].isna() & renamed["scenario_selic_change_bps"].notna() & renamed["probability"].notna()
    if has_scenario.any():
        grouped = (
            renamed[has_scenario]
            .assign(weighted=lambda data: data["scenario_selic_change_bps"] * data["probability"])
            .groupby(["meeting_id", "as_of_timestamp"], dropna=False)["weighted"]
            .sum()
        )
        for (meeting_id, as_of), value in grouped.items():
            mask = (renamed["meeting_id"] == meeting_id) & (renamed["as_of_timestamp"] == as_of) & renamed["expected_selic_change_bps"].isna()
            renamed.loc[mask, "expected_selic_change_bps"] = float(value)
    renamed["collected_at"] = pd.to_datetime(renamed["collected_at"], errors="coerce")
    renamed.loc[renamed["collected_at"].isna(), "collected_at"] = utc_now_naive()
    normalized = renamed[DECISION_EXPECTATION_COLUMNS].dropna(subset=["meeting_id", "as_of_timestamp"])
    normalized = normalized.drop_duplicates(
        ["meeting_id", "as_of_timestamp", "expected_selic_change_bps", "scenario_selic_change_bps", "source_file"],
        keep="last",
    )
    return normalized.sort_values(["meeting_id", "as_of_timestamp"]).reset_index(drop=True)


def empty_decision_expectations() -> pd.DataFrame:
    return pd.DataFrame(columns=DECISION_EXPECTATION_COLUMNS)


def public_market_date_range(database: Path, months: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    meetings = read_optional_table(database, "v2_meetings", pd.DataFrame())
    if not meetings.empty and "data_referencia" in meetings:
        dates = pd.to_datetime(meetings["data_referencia"], errors="coerce").dropna()
        if not dates.empty:
            end_date = pd.Timestamp(dates.max()).normalize() + pd.Timedelta(days=14)
            start_date = max(pd.Timestamp(dates.min()).normalize() - pd.Timedelta(days=7), end_date - pd.DateOffset(months=months))
            return pd.Timestamp(start_date), pd.Timestamp(end_date)
    end_date = pd.Timestamp.today().normalize()
    return end_date - pd.DateOffset(months=months), end_date


def fetch_bcb_sgs_market_observations(start_date: pd.Timestamp, end_date: pd.Timestamp) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    frames: list[pd.DataFrame] = []
    audit: list[dict[str, Any]] = []
    for asset, config in SGS_SERIES.items():
        try:
            frame = fetch_sgs_series(
                code=int(config["code"]),
                asset=asset,
                asset_class=str(config["asset_class"]),
                vertex=str(config["vertex"]),
                unit=str(config["unit"]),
                start_date=start_date,
                end_date=end_date,
            )
            frames.append(frame)
            audit.append(public_audit_row(f"bcb-sgs:{asset}", "ok", len(frame), start_date, end_date, f"SGS code {config['code']}", SGS_URL_TEMPLATE.format(code=config["code"])))
        except Exception as exc:  # noqa: BLE001 - public source availability is reported.
            audit.append(public_audit_row(f"bcb-sgs:{asset}", "unavailable", 0, start_date, end_date, str(exc), SGS_URL_TEMPLATE.format(code=config["code"])))
    observations = normalize_market_observations(pd.concat(frames, ignore_index=True)) if frames else empty_market_observations()
    return observations, audit


def fetch_sgs_series(
    code: int,
    asset: str,
    asset_class: str,
    vertex: str,
    unit: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for block_start, block_end in date_blocks(start_date, end_date, years=5):
        url = SGS_URL_TEMPLATE.format(code=code)
        params = {
            "formato": "json",
            "dataInicial": block_start.strftime("%d/%m/%Y"),
            "dataFinal": block_end.strftime("%d/%m/%Y"),
        }
        response = http_get_with_retry(url, params=params, timeout=60)
        response.raise_for_status()
        for item in response.json():
            rows.append(
                {
                    "asset": asset,
                    "asset_class": asset_class,
                    "vertex": vertex,
                    "timestamp": pd.to_datetime(item.get("data"), format="%d/%m/%Y", errors="coerce"),
                    "value": parse_number(item.get("valor")),
                    "unit": unit,
                    "currency": "BRL",
                    "frequency": "daily",
                    "market_timezone": "America/Sao_Paulo",
                    "source": "bcb_sgs",
                    "source_url": url,
                    "source_method": f"sgs_code_{code}",
                    "source_status": "ok",
                    "data_access_tier": "PUBLIC_API",
                    "license_note": "BCB SGS public API",
                    "collected_at": utc_now_naive(),
                }
            )
    return normalize_market_observations(pd.DataFrame(rows))


def fetch_ptax_market_observations(start_date: pd.Timestamp, end_date: pd.Timestamp) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    frames: list[pd.DataFrame] = []
    audit: list[dict[str, Any]] = []
    try:
        for block_start, block_end in date_blocks(start_date, end_date, years=1):
            frames.append(fetch_ptax_block(block_start, block_end))
        observations = normalize_market_observations(pd.concat(frames, ignore_index=True)) if frames else empty_market_observations()
        audit.append(public_audit_row("ptax:USD_BRL", "ok", len(observations), start_date, end_date, "BCB PTAX OData USD close.", PTAX_URL))
        return observations, audit
    except Exception as exc:  # noqa: BLE001
        audit.append(public_audit_row("ptax:USD_BRL", "unavailable", 0, start_date, end_date, str(exc), PTAX_URL))
        return empty_market_observations(), audit


def fetch_ptax_block(start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    params = {
        "@moeda": "'USD'",
        "@dataInicial": f"'{start_date.strftime('%m-%d-%Y')}'",
        "@dataFinalCotacao": f"'{end_date.strftime('%m-%d-%Y')}'",
        "$format": "json",
    }
    url = PTAX_URL + "?" + urlencode(params, quote_via=quote, safe="$@()',")
    response = http_get_with_retry(url, timeout=30)
    response.raise_for_status()
    data = response.json().get("value", [])
    frame = pd.DataFrame(data)
    if frame.empty:
        return empty_market_observations()
    frame["timestamp"] = pd.to_datetime(frame["dataHoraCotacao"], errors="coerce")
    frame["date"] = frame["timestamp"].dt.normalize()
    frame = frame.sort_values("timestamp").groupby("date", as_index=False).tail(1)
    return normalize_market_observations(
        pd.DataFrame(
            {
                "asset": "USD_BRL_PTAX",
                "asset_class": "fx",
                "vertex": "spot",
                "timestamp": frame["timestamp"],
                "value": pd.to_numeric(frame["cotacaoVenda"], errors="coerce"),
                "unit": "BRL_per_USD",
                "currency": "BRL",
                "frequency": "daily",
                "market_timezone": "America/Sao_Paulo",
                "source": "bcb_ptax",
                "source_url": PTAX_URL,
                "source_method": "PTAX CotacaoMoedaPeriodo USD last daily quote",
                "source_status": "ok",
                "data_access_tier": "PUBLIC_API",
                "license_note": "BCB PTAX public OData",
                "collected_at": utc_now_naive(),
            }
        )
    )


def fetch_anbima_market_observations(reference_date: pd.Timestamp) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    try:
        html_text = fetch_anbima_intraday_html(reference_date)
        frame = parse_anbima_intraday_html(html_text, reference_date)
        status = "ok" if not frame.empty else "unavailable"
        detail = "ANBIMA intraday page parsed. Historical access is limited to recent business days."
        return frame, [public_audit_row("anbima:ettj_intraday", status, len(frame), reference_date, reference_date, detail, ANBIMA_INTRADAY_URL)]
    except Exception as exc:  # noqa: BLE001
        return empty_market_observations(), [
            public_audit_row("anbima:ettj_intraday", "unavailable", 0, reference_date, reference_date, str(exc), ANBIMA_INTRADAY_URL)
        ]


def fetch_anbima_intraday_html(reference_date: pd.Timestamp) -> str:
    response = http_post_with_retry(
        ANBIMA_INTRADAY_URL,
        data={"escolha": "2", "saida": "csv", "Dt_Ref": pd.Timestamp(reference_date).strftime("%d/%m/%Y")},
        timeout=30,
    )
    response.raise_for_status()
    return response.text


def parse_anbima_intraday_html(html_text: str, reference_date: pd.Timestamp) -> pd.DataFrame:
    try:
        tables = pd.read_html(io.StringIO(html_text), decimal=",", thousands=".")
    except ValueError:
        return empty_market_observations()
    rows: list[dict[str, Any]] = []
    table_kinds = ["PRE", "REAL"]
    for idx, table in enumerate(tables):
        if table.shape[1] < 3:
            continue
        kind = table_kinds[min(idx, len(table_kinds) - 1)]
        parsed = table.copy()
        parsed.columns = ["vertex_days", "d_minus_1", "d0"][: len(parsed.columns)]
        parsed["vertex_days"] = pd.to_numeric(parsed["vertex_days"], errors="coerce")
        parsed["d0"] = pd.to_numeric(parsed["d0"], errors="coerce")
        parsed = parsed.dropna(subset=["vertex_days", "d0"])
        for vertex, target_days in ANBIMA_TARGET_VERTICES.items():
            selected = parsed.iloc[(parsed["vertex_days"] - target_days).abs().argsort()].iloc[0]
            rows.append(
                {
                    "asset": f"ANBIMA_ETTJ_{kind}",
                    "asset_class": "rates_curve",
                    "vertex": vertex,
                    "timestamp": pd.Timestamp(reference_date).normalize() + pd.Timedelta(hours=12),
                    "value": selected["d0"],
                    "unit": "percent_annual_252",
                    "currency": "BRL",
                    "frequency": "intraday",
                    "market_timezone": "America/Sao_Paulo",
                    "source": "anbima_ettj_intraday",
                    "source_url": ANBIMA_INTRADAY_URL,
                    "source_method": "anbima_intraday_html_table_nearest_vertex",
                    "source_status": "limited_recent_history",
                    "data_access_tier": "PUBLIC_API",
                    "license_note": "ANBIMA public intraday curve page; historical coverage is limited by source page.",
                    "collected_at": utc_now_naive(),
                }
            )
    return normalize_market_observations(pd.DataFrame(rows))


def derive_public_decision_expectations(
    meetings: pd.DataFrame,
    focus_features: pd.DataFrame,
    market_observations: pd.DataFrame,
    method: str = "public",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    audit_rows = [
        public_audit_row(
            "b3:opcao_copom",
            "unavailable",
            0,
            pd.NaT,
            pd.NaT,
            "No structured public historical probability feed was found; official decision surprise is not filled.",
            B3_COPOM_OPTION_URL,
        )
    ]
    rows: list[dict[str, Any]] = []
    rows.extend(derive_focus_selic_proxy_rows(meetings, focus_features))
    rows.extend(derive_curve_short_rate_proxy_rows(meetings, market_observations))
    expectations = normalize_decision_expectations(pd.DataFrame(rows))
    proxy_count = int(expectations["is_proxy"].sum()) if not expectations.empty else 0
    audit_rows.append(public_audit_row(f"{method}:proxy", "ok" if proxy_count else "unavailable", proxy_count, pd.NaT, pd.NaT, "Proxy expectations are separated from official decision surprise.", ""))
    audit = pd.DataFrame(audit_rows, columns=PUBLIC_MARKET_AUDIT_COLUMNS)
    return expectations, audit


def derive_focus_selic_proxy_rows(meetings: pd.DataFrame, focus_features: pd.DataFrame) -> list[dict[str, Any]]:
    if meetings.empty or focus_features.empty:
        return []
    features = focus_features.copy()
    features = features[
        (features.get("indicator", pd.Series(dtype=str)).astype(str).str.lower() == "selic")
        & (features.get("statistic", pd.Series(dtype=str)).astype(str).str.lower() == "median")
        & (features.get("horizon", pd.Series(dtype=str)).astype(str) == "current_year")
        & features.get("pre_value", pd.Series(dtype=float)).notna()
    ].copy()
    if features.empty:
        return []
    meetings = meetings.copy()
    rows: list[dict[str, Any]] = []
    for _, meeting in meetings.iterrows():
        meeting_id = str(meeting.get("meeting_id", ""))
        selic_pre = meeting.get("selic_pre")
        meeting_date = pd.to_datetime(meeting.get("data_referencia"), errors="coerce")
        if pd.isna(selic_pre) or pd.isna(meeting_date):
            continue
        subset = features[features["meeting_id"].astype(str) == meeting_id].copy()
        if "event_type" in subset:
            comunicado = subset[subset["event_type"].astype(str) == "comunicado"].copy()
            if not comunicado.empty:
                subset = comunicado
        subset["pre_date"] = pd.to_datetime(subset["pre_date"], errors="coerce")
        subset = subset[subset["pre_date"] < pd.Timestamp(meeting_date).normalize()].copy()
        if subset.empty:
            continue
        selected = subset.sort_values("pre_date").iloc[-1]
        rows.append(
            decision_expectation_row(
                meeting_id=meeting_id,
                as_of_timestamp=pd.Timestamp(selected["pre_date"]).normalize() + pd.Timedelta(hours=8),
                expected_selic_change_bps=(float(selected["pre_value"]) - float(selic_pre)) * 100.0,
                source="focus_selic_proxy",
                data_access_tier="DERIVED",
                source_method="focus_current_year_selic_pre_minus_selic_pre",
                source_status="proxy",
                is_proxy=True,
                proxy_method="focus_selic_year_end_minus_current_selic",
                license_note="Derived from BCB Focus vintages; not a point decision expectation.",
            )
        )
    return rows


def derive_curve_short_rate_proxy_rows(meetings: pd.DataFrame, market_observations: pd.DataFrame) -> list[dict[str, Any]]:
    if meetings.empty or market_observations.empty:
        return []
    observations = normalize_market_observations(market_observations)
    curve = observations[
        observations["asset"].isin(["ANBIMA_ETTJ_PRE", "SELIC_META", "SELIC_OVER", "CDI"])
        & observations["value"].notna()
    ].copy()
    if curve.empty:
        return []
    rows: list[dict[str, Any]] = []
    for _, meeting in meetings.iterrows():
        meeting_id = str(meeting.get("meeting_id", ""))
        event_time = pd.to_datetime(meeting.get("data_referencia"), errors="coerce")
        selic_pre = meeting.get("selic_pre")
        if pd.isna(event_time) or pd.isna(selic_pre):
            continue
        cutoff = event_time.normalize() + pd.Timedelta(hours=18, minutes=30)
        eligible = curve[curve["timestamp"] < cutoff].sort_values("timestamp")
        if eligible.empty:
            continue
        preferred = eligible[eligible["vertex"].isin(["3m", "target", "overnight"])]
        selected = (preferred if not preferred.empty else eligible).iloc[-1]
        rows.append(
            decision_expectation_row(
                meeting_id=meeting_id,
                as_of_timestamp=selected["timestamp"],
                expected_selic_change_bps=(float(selected["value"]) - float(selic_pre)) * 100.0,
                source="curve_short_rate_proxy",
                data_access_tier="DERIVED",
                source_method=f"{selected['asset']}_{selected['vertex']}_minus_selic_pre",
                source_status="proxy",
                is_proxy=True,
                proxy_method="short_rate_level_minus_current_selic",
                license_note="Derived from public market rate level; not an official decision probability.",
            )
        )
    return rows


def decision_expectation_row(
    meeting_id: str,
    as_of_timestamp: Any,
    expected_selic_change_bps: float,
    source: str,
    data_access_tier: str,
    source_method: str,
    source_status: str,
    is_proxy: bool,
    proxy_method: str,
    license_note: str,
) -> dict[str, Any]:
    return {
        "meeting_id": meeting_id,
        "source": source,
        "data_access_tier": data_access_tier,
        "as_of_timestamp": as_of_timestamp,
        "expected_selic_change_bps": expected_selic_change_bps,
        "scenario_selic_change_bps": np.nan,
        "probability": np.nan,
        "license_note": license_note,
        "source_file": "",
        "source_url": "",
        "source_method": source_method,
        "source_status": source_status,
        "is_proxy": is_proxy,
        "proxy_method": proxy_method,
        "collected_at": utc_now_naive(),
    }


def build_market_event_windows(events: pd.DataFrame, observations: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "meeting_id",
        "nro_reuniao",
        "document_type",
        "asset",
        "asset_class",
        "vertex",
        "window",
        "pre_timestamp",
        "post_timestamp",
        "pre_value",
        "post_value",
        "market_reaction",
        "known_at_timestamp",
        "market_close_convention",
        "data_access_tier",
        "status",
    ]
    if events.empty or observations.empty:
        return pd.DataFrame(columns=columns)
    events = normalize_market_events(events)
    observations = normalize_market_observations(observations)
    grouped_series = [prepare_market_series(asset, vertex, series) for (asset, vertex), series in observations.groupby(["asset", "vertex"], dropna=False)]
    rows: list[dict[str, Any]] = []
    for _, event in events.iterrows():
        if pd.isna(event["known_at_timestamp"]):
            rows.append(empty_market_window_row(event, columns, status="ambiguous_event_timing"))
            continue
        known_at = pd.Timestamp(event["known_at_timestamp"])
        for series_info in grouped_series:
            rows.extend(build_market_window_rows_fast(event, known_at, series_info))
    return pd.DataFrame(rows, columns=columns)


def prepare_market_series(asset: str, vertex: str, series: pd.DataFrame) -> dict[str, Any]:
    ordered = series.sort_values("timestamp").reset_index(drop=True)
    timestamps = pd.to_datetime(ordered["timestamp"], errors="coerce")
    return {
        "asset": asset,
        "vertex": vertex,
        "series": ordered,
        "timestamps_ns": timestamps.astype("int64").to_numpy(),
        "dates_ns": timestamps.dt.normalize().astype("int64").to_numpy(),
        "asset_class": first_non_empty(ordered, "asset_class"),
        "data_access_tier": first_non_empty(ordered, "data_access_tier"),
    }


def build_market_window_rows_fast(event: pd.Series, known_at: pd.Timestamp, series_info: dict[str, Any]) -> list[dict[str, Any]]:
    series = series_info["series"]
    timestamps_ns = series_info["timestamps_ns"]
    if len(timestamps_ns) == 0:
        return []
    known_ns = pd.Timestamp(known_at).value
    pre_position = int(np.searchsorted(timestamps_ns, known_ns, side="left") - 1)
    first_post_position = int(np.searchsorted(timestamps_ns, known_ns, side="right"))
    rows = [
        build_market_window_row_by_position(event, series_info, known_at, "close_to_next_close", pre_position, first_post_position),
        build_market_window_row_by_position(event, series_info, known_at, "close_to_second_close", pre_position, first_post_position + 1),
    ]
    same_day = np.flatnonzero(series_info["dates_ns"] == pd.Timestamp(known_at).normalize().value)
    if len(same_day) > 0:
        intraday_pre = same_day[same_day < first_post_position]
        intraday_post = same_day[same_day >= first_post_position]
        if len(intraday_pre) > 0 or len(intraday_post) > 0:
            rows.append(
                build_market_window_row_by_position(
                    event,
                    series_info,
                    known_at,
                    "intraday_before_after",
                    int(intraday_pre[-1]) if len(intraday_pre) else -1,
                    int(intraday_post[0]) if len(intraday_post) else len(series),
                )
            )
    return rows


def build_market_window_row_by_position(
    event: pd.Series,
    series_info: dict[str, Any],
    known_at: pd.Timestamp,
    label: str,
    pre_position: int,
    post_position: int,
) -> dict[str, Any]:
    series = series_info["series"]
    base = {
        "meeting_id": event.get("meeting_id", ""),
        "nro_reuniao": event.get("nro_reuniao", np.nan),
        "document_type": event.get("document_type", ""),
        "asset": series_info["asset"],
        "asset_class": series_info["asset_class"],
        "vertex": series_info["vertex"],
        "window": label,
        "known_at_timestamp": known_at,
        "market_close_convention": event.get("market_close_convention", ""),
        "data_access_tier": series_info["data_access_tier"],
    }
    if pre_position < 0 or post_position >= len(series):
        return {
            **base,
            "pre_timestamp": pd.NaT,
            "post_timestamp": pd.NaT,
            "pre_value": np.nan,
            "post_value": np.nan,
            "market_reaction": np.nan,
            "status": "insufficient_market_observations",
        }
    pre_row = series.iloc[pre_position]
    post_row = series.iloc[post_position]
    if not (pd.Timestamp(pre_row["timestamp"]) < known_at < pd.Timestamp(post_row["timestamp"])):
        return {
            **base,
            "pre_timestamp": pre_row["timestamp"],
            "post_timestamp": post_row["timestamp"],
            "pre_value": pre_row["value"],
            "post_value": post_row["value"],
            "market_reaction": np.nan,
            "status": "ambiguous_event_timing",
        }
    return {
        **base,
        "pre_timestamp": pre_row["timestamp"],
        "post_timestamp": post_row["timestamp"],
        "pre_value": pre_row["value"],
        "post_value": post_row["value"],
        "market_reaction": float(post_row["value"]) - float(pre_row["value"]),
        "status": "ok",
    }


def normalize_market_events(events: pd.DataFrame) -> pd.DataFrame:
    frame = events.copy()
    if "document_type" not in frame:
        frame["document_type"] = frame.get("event_type", "")
    if "release_date" not in frame:
        frame["release_date"] = frame.get("event_date", pd.NaT)
    if "market_close_convention" not in frame:
        frame["market_close_convention"] = np.where(frame["document_type"] == "comunicado", "after_close", "before_or_during_session")
    if "known_at_timestamp" not in frame:
        frame["known_at_timestamp"] = pd.NaT
    frame["release_date"] = pd.to_datetime(frame["release_date"], errors="coerce").dt.normalize()
    frame["known_at_timestamp"] = pd.to_datetime(frame["known_at_timestamp"], errors="coerce")
    missing = frame["known_at_timestamp"].isna() & frame["release_date"].notna()
    if missing.any():
        frame.loc[missing, "known_at_timestamp"] = pd.to_datetime(frame.loc[missing].apply(_known_at_from_event_row, axis=1))
    for column in ["meeting_id", "document_type", "market_close_convention"]:
        if column not in frame:
            frame[column] = ""
        frame[column] = frame[column].astype(str)
    if "nro_reuniao" not in frame:
        frame["nro_reuniao"] = np.nan
    return frame


def build_market_window_row(
    event: pd.Series,
    asset: str,
    vertex: str,
    series: pd.DataFrame,
    pre: pd.DataFrame,
    post: pd.DataFrame,
    label: str,
    post_index: int,
) -> dict[str, Any]:
    known_at = pd.Timestamp(event["known_at_timestamp"])
    base = {
        "meeting_id": event.get("meeting_id", ""),
        "nro_reuniao": event.get("nro_reuniao", np.nan),
        "document_type": event.get("document_type", ""),
        "asset": asset,
        "asset_class": first_non_empty(series, "asset_class"),
        "vertex": vertex,
        "window": label,
        "known_at_timestamp": known_at,
        "market_close_convention": event.get("market_close_convention", ""),
        "data_access_tier": first_non_empty(series, "data_access_tier"),
    }
    if pre.empty or len(post) <= post_index:
        return {
            **base,
            "pre_timestamp": pd.NaT,
            "post_timestamp": pd.NaT,
            "pre_value": np.nan,
            "post_value": np.nan,
            "market_reaction": np.nan,
            "status": "insufficient_market_observations",
        }
    pre_row = pre.iloc[-1]
    post_row = post.iloc[post_index]
    if not (pd.Timestamp(pre_row["timestamp"]) < known_at < pd.Timestamp(post_row["timestamp"])):
        return {
            **base,
            "pre_timestamp": pre_row["timestamp"],
            "post_timestamp": post_row["timestamp"],
            "pre_value": pre_row["value"],
            "post_value": post_row["value"],
            "market_reaction": np.nan,
            "status": "ambiguous_event_timing",
        }
    return {
        **base,
        "pre_timestamp": pre_row["timestamp"],
        "post_timestamp": post_row["timestamp"],
        "pre_value": pre_row["value"],
        "post_value": post_row["value"],
        "market_reaction": float(post_row["value"]) - float(pre_row["value"]),
        "status": "ok",
    }


def empty_market_window_row(event: pd.Series, columns: list[str], status: str) -> dict[str, Any]:
    row = {column: np.nan for column in columns}
    row.update(
        {
            "meeting_id": event.get("meeting_id", ""),
            "nro_reuniao": event.get("nro_reuniao", np.nan),
            "document_type": event.get("document_type", ""),
            "known_at_timestamp": event.get("known_at_timestamp", pd.NaT),
            "market_close_convention": event.get("market_close_convention", ""),
            "status": status,
        }
    )
    return row


def first_non_empty(frame: pd.DataFrame, column: str) -> str:
    if column not in frame:
        return ""
    values = frame[column].dropna().astype(str)
    values = values[values.str.strip() != ""]
    return values.iloc[0] if not values.empty else ""


def build_public_market_coverage(
    observations: pd.DataFrame,
    windows: pd.DataFrame,
    expectations: pd.DataFrame,
    market_audit: pd.DataFrame,
    decision_audit: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rows.append({"section": "market_observations", "metric": "rows", "value": len(observations), "status": "ok" if not observations.empty else "missing"})
    rows.append({"section": "market_observations", "metric": "assets", "value": observations["asset"].nunique() if not observations.empty and "asset" in observations else 0, "status": "info"})
    rows.append({"section": "market_event_windows", "metric": "rows", "value": len(windows), "status": "ok" if not windows.empty else "missing"})
    ok_windows = int((windows.get("status", pd.Series(dtype=str)) == "ok").sum()) if not windows.empty else 0
    rows.append({"section": "market_event_windows", "metric": "ok_windows", "value": ok_windows, "status": "ok" if ok_windows else "warning"})
    official = expectations[~normalize_bool_series(expectations.get("is_proxy", pd.Series(dtype=bool)))] if not expectations.empty else pd.DataFrame()
    proxy = expectations[normalize_bool_series(expectations.get("is_proxy", pd.Series(dtype=bool)))] if not expectations.empty else pd.DataFrame()
    rows.append({"section": "decision_expectations", "metric": "official_rows", "value": len(official), "status": "ok" if not official.empty else "warning"})
    rows.append({"section": "decision_expectations", "metric": "proxy_rows", "value": len(proxy), "status": "ok" if not proxy.empty else "warning"})
    for frame, section in [(market_audit, "source_audit"), (decision_audit, "decision_source_audit")]:
        if frame.empty:
            rows.append({"section": section, "metric": "sources_reported", "value": 0, "status": "warning"})
            continue
        for _, row in frame.iterrows():
            rows.append(
                {
                    "section": section,
                    "metric": row.get("source", ""),
                    "value": row.get("rows", 0),
                    "status": row.get("status", ""),
                    "detail": row.get("detail", ""),
                }
            )
    return pd.DataFrame(rows)


def write_public_market_outputs(public_observations: pd.DataFrame, audit: pd.DataFrame) -> Path:
    paths = get_paths()
    data_dir = paths.database.parents[1] / "data" / "market" / "generated"
    output_dir = paths.processed.parent / "v2"
    report_dir = paths.reports.parent / "v2"
    for directory in [data_dir, output_dir, report_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / "market_observations_public.csv"
    public_observations.to_csv(csv_path, index=False)
    public_observations.to_csv(output_dir / "market_observations_public.csv", index=False)
    public_observations.to_parquet(output_dir / "market_observations_public.parquet", index=False)
    audit.to_csv(output_dir / "public_market_source_audit.csv", index=False)
    audit.to_parquet(output_dir / "public_market_source_audit.parquet", index=False)
    (report_dir / "public_market_data_report.html").write_text(render_public_market_data_report_html(public_observations, audit), encoding="utf-8")
    return csv_path


def write_decision_expectation_outputs(expectations: pd.DataFrame, audit: pd.DataFrame) -> Path:
    paths = get_paths()
    data_dir = paths.database.parents[1] / "data" / "market" / "generated"
    output_dir = paths.processed.parent / "v2"
    for directory in [data_dir, output_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / "decision_expectations_public.csv"
    expectations.to_csv(csv_path, index=False)
    expectations.to_csv(output_dir / "decision_expectations.csv", index=False)
    expectations.to_parquet(output_dir / "decision_expectations.parquet", index=False)
    audit.to_csv(output_dir / "decision_expectation_source_audit.csv", index=False)
    audit.to_parquet(output_dir / "decision_expectation_source_audit.parquet", index=False)
    return csv_path


def write_public_market_coverage_report(coverage: pd.DataFrame) -> Path:
    paths = get_paths()
    output_dir = paths.processed.parent / "v2"
    report_dir = paths.reports.parent / "v2"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(output_dir / "public_market_coverage.csv", index=False)
    coverage.to_parquet(output_dir / "public_market_coverage.parquet", index=False)
    report_path = report_dir / "public_market_coverage_report.html"
    report_path.write_text(render_public_market_coverage_report_html(coverage), encoding="utf-8")
    return output_dir / "public_market_coverage.csv"


def write_market_event_outputs(windows: pd.DataFrame) -> Path:
    paths = get_paths()
    output_dir = paths.processed.parent / "v2"
    report_dir = paths.reports.parent / "v2"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "market_event_study.csv"
    windows.to_csv(output_path, index=False)
    windows.to_parquet(output_dir / "market_event_study.parquet", index=False)
    report_path = report_dir / "market_event_study_report.html"
    report_path.write_text(render_market_event_study_report_html(windows), encoding="utf-8")
    return output_path


def render_public_market_data_report_html(observations: pd.DataFrame, audit: pd.DataFrame) -> str:
    asset_counts = observations.get("asset", pd.Series(dtype=str)).value_counts().reset_index().to_html(index=False, escape=True) if not observations.empty else "<p>Sem observacoes publicas.</p>"
    audit_html = audit.to_html(index=False, escape=True) if not audit.empty else "<p>Sem auditoria de fontes.</p>"
    return "\n".join(
        [
            "<!doctype html><html><head><meta charset='utf-8'>",
            "<title>COPOM Watch Public Market Data</title>",
            "<style>body{font-family:Arial,sans-serif;margin:32px;line-height:1.45;color:#111827}table{border-collapse:collapse;width:100%;margin:12px 0}td,th{border:1px solid #ddd;padding:6px}th{background:#f3f4f6}</style>",
            "</head><body>",
            "<h1>COPOM Watch Public Market Data</h1>",
            "<p>Dados publicos coletados para event study. Fontes indisponiveis sao registradas sem quebrar o pipeline.</p>",
            f"<p>Observacoes publicas: <strong>{len(observations)}</strong>.</p>",
            "<h2>Ativos</h2>",
            asset_counts,
            "<h2>Auditoria de fontes</h2>",
            audit_html,
            "</body></html>",
        ]
    )


def render_public_market_coverage_report_html(coverage: pd.DataFrame) -> str:
    table = coverage.to_html(index=False, escape=True) if not coverage.empty else "<p>Sem cobertura.</p>"
    return "\n".join(
        [
            "<!doctype html><html><head><meta charset='utf-8'>",
            "<title>COPOM Watch Public Market Coverage</title>",
            "<style>body{font-family:Arial,sans-serif;margin:32px;line-height:1.45;color:#111827}table{border-collapse:collapse;width:100%;margin:12px 0}td,th{border:1px solid #ddd;padding:6px}th{background:#f3f4f6}</style>",
            "</head><body>",
            "<h1>COPOM Watch Public Market Coverage</h1>",
            "<p>Cobertura de mercado publico e expectativas de decisao. Proxies permanecem separados de surpresa oficial.</p>",
            table,
            "</body></html>",
        ]
    )


def render_market_event_study_report_html(windows: pd.DataFrame) -> str:
    ok = int((windows.get("status", pd.Series(dtype=str)) == "ok").sum()) if not windows.empty else 0
    status_counts = windows.get("status", pd.Series(dtype=str)).value_counts().reset_index().to_html(index=False, escape=True) if not windows.empty else "<p>Sem janelas de mercado.</p>"
    sample = windows.head(50).to_html(index=False, escape=True) if not windows.empty else "<p>Sem dados.</p>"
    return "\n".join(
        [
            "<!doctype html><html><head><meta charset='utf-8'>",
            "<title>COPOM Watch V2.1 Market Event Study</title>",
            "<style>body{font-family:Arial,sans-serif;margin:32px;line-height:1.45;color:#111827}table{border-collapse:collapse;width:100%;margin:12px 0}td,th{border:1px solid #ddd;padding:6px}th{background:#f3f4f6}</style>",
            "</head><body>",
            "<h1>COPOM Watch V2.1 Market Event Study</h1>",
            "<p>Estudo de evento descritivo. As janelas respeitam known_at_timestamp e nao devem ser lidas como causalidade.</p>",
            f"<p>Janelas totais: <strong>{len(windows)}</strong>. Janelas ok: <strong>{ok}</strong>.</p>",
            "<h2>Status</h2>",
            status_counts,
            "<h2>Amostra</h2>",
            sample,
            "</body></html>",
        ]
    )


def _known_at_from_event_row(row: pd.Series) -> pd.Timestamp:
    release_date = pd.to_datetime(row.get("release_date"), errors="coerce")
    if pd.isna(release_date):
        return pd.NaT
    convention = str(row.get("market_close_convention", ""))
    document_type = str(row.get("document_type", ""))
    if convention == "after_close" or document_type == "comunicado":
        return release_date.normalize() + pd.Timedelta(hours=18, minutes=30)
    return release_date.normalize() + pd.Timedelta(hours=8)


def date_blocks(start_date: pd.Timestamp, end_date: pd.Timestamp, years: int) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    blocks = []
    current = start
    while current <= end:
        block_end = min(current + pd.DateOffset(years=years) - pd.Timedelta(days=1), end)
        blocks.append((current, pd.Timestamp(block_end)))
        current = pd.Timestamp(block_end) + pd.Timedelta(days=1)
    return blocks


def parse_number(value: Any) -> float:
    if pd.isna(value):
        return np.nan
    return float(str(value).replace(".", "").replace(",", ".") if "," in str(value) else value)


def normalize_bool_series(series: pd.Series) -> pd.Series:
    if series.empty:
        return pd.Series(dtype=bool)
    if series.dtype == bool:
        return series.astype(bool)
    clean = pd.Series(np.where(series.notna(), series, False), index=series.index)
    return clean.astype(str).str.lower().isin({"true", "1", "yes", "sim"})


def concat_nonempty(frames: list[pd.DataFrame], empty: pd.DataFrame) -> pd.DataFrame:
    nonempty = [frame for frame in frames if not frame.empty]
    return pd.concat(nonempty, ignore_index=True) if nonempty else empty


def http_get_with_retry(url: str, **kwargs: Any) -> httpx.Response:
    return http_request_with_retry("GET", url, **kwargs)


def http_post_with_retry(url: str, **kwargs: Any) -> httpx.Response:
    return http_request_with_retry("POST", url, **kwargs)


def http_request_with_retry(method: str, url: str, **kwargs: Any) -> httpx.Response:
    last_error: Exception | None = None
    for attempt in range(1, HTTP_RETRY_ATTEMPTS + 1):
        try:
            response = httpx.request(method, url, **kwargs)
            response.raise_for_status()
            return response
        except (httpx.HTTPError, httpx.TimeoutException) as exc:
            last_error = exc
            if attempt == HTTP_RETRY_ATTEMPTS:
                break
            time.sleep(HTTP_RETRY_BACKOFF_SECONDS * attempt)
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"HTTP request failed without exception: {method} {url}")


def public_audit_row(
    source: str,
    status: str,
    rows: int,
    start_date: Any,
    end_date: Any,
    detail: str,
    source_url: str,
) -> dict[str, Any]:
    return {
        "source": source,
        "status": status,
        "rows": int(rows),
        "start_date": pd.to_datetime(start_date, errors="coerce"),
        "end_date": pd.to_datetime(end_date, errors="coerce"),
        "source_url": source_url,
        "detail": detail,
    }

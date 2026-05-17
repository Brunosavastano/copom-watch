from __future__ import annotations

import logging
import re
from datetime import date, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from copom_tone_index.http_client import CachedHttpClient, FetchError

LOGGER = logging.getLogger(__name__)

BCB_COPOM_BASE = "https://www.bcb.gov.br/api/servico/sitebcb/copom"
BCB_SGS_432 = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.432/dados"
BCB_FOCUS_ANNUAL = "https://olinda.bcb.gov.br/olinda/servico/Expectativas/versao/v1/odata/ExpectativasMercadoAnuais"


def _as_date(value: Any) -> pd.Timestamp | pd.NaT:
    if value in (None, "", np.nan):
        return pd.NaT
    return pd.to_datetime(value, errors="coerce")


def _normalize_meeting_number(row: dict[str, Any]) -> int:
    number = row.get("nro_reuniao", row.get("nroReuniao"))
    if number is None:
        raise ValueError(f"Missing meeting number in row: {row}")
    return int(number)


def _content(data: dict[str, Any]) -> list[dict[str, Any]]:
    value = data.get("conteudo", data.get("value", []))
    if isinstance(value, list):
        return value
    return []


def fetch_copom_documents(client: CachedHttpClient, quantity: int = 100) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch COPOM meetings and document details from official BCB endpoints."""
    LOGGER.info("Fetching COPOM lists from BCB.")
    minutes = _content(client.get_json(f"{BCB_COPOM_BASE}/atas", "copom_atas_list", {"quantidade": quantity}))
    statements = _content(
        client.get_json(f"{BCB_COPOM_BASE}/comunicados", "copom_comunicados_list", {"quantidade": quantity})
    )

    meeting_numbers = sorted({_normalize_meeting_number(row) for row in minutes + statements}, reverse=True)
    document_rows: list[dict[str, Any]] = []
    meeting_rows: dict[int, dict[str, Any]] = {}

    for nro_reuniao in meeting_numbers:
        LOGGER.info("Fetching COPOM meeting %s details.", nro_reuniao)
        try:
            statement_detail = _content(client.get_json(
                f"{BCB_COPOM_BASE}/comunicados_detalhes",
                f"copom_comunicado_{nro_reuniao}",
                {"nro_reuniao": nro_reuniao},
            ))
        except FetchError as exc:
            LOGGER.warning("Skipping comunicado details for meeting %s after fetch failure: %s", nro_reuniao, exc)
            statement_detail = []
        try:
            minutes_detail = _content(client.get_json(
                f"{BCB_COPOM_BASE}/atas_detalhes",
                f"copom_ata_{nro_reuniao}",
                {"nro_reuniao": nro_reuniao},
            ))
        except FetchError as exc:
            LOGGER.warning("Skipping ata details for meeting %s after fetch failure: %s", nro_reuniao, exc)
            minutes_detail = []

        for item in statement_detail:
            meeting_id = f"copom_{nro_reuniao}"
            data_referencia = _as_date(item.get("dataReferencia", item.get("data_referencia")))
            meeting_rows[nro_reuniao] = {
                "meeting_id": meeting_id,
                "nro_reuniao": nro_reuniao,
                "data_referencia": data_referencia,
                "data_comunicado": data_referencia,
            } | meeting_rows.get(nro_reuniao, {})
            document_rows.append(
                {
                    "document_id": f"{meeting_id}_comunicado",
                    "meeting_id": meeting_id,
                    "nro_reuniao": nro_reuniao,
                    "document_type": "comunicado",
                    "publication_date": data_referencia,
                    "title": item.get("titulo", ""),
                    "url": None,
                    "raw_text": item.get("textoComunicado", ""),
                    "source": "bcb_copom_comunicados_detalhes",
                }
            )

        for item in minutes_detail:
            meeting_id = f"copom_{nro_reuniao}"
            data_referencia = _as_date(item.get("dataReferencia", item.get("data_referencia")))
            data_publicacao = _as_date(item.get("dataPublicacao", item.get("data_publicacao")))
            meeting_rows[nro_reuniao] = meeting_rows.get(nro_reuniao, {}) | {
                "meeting_id": meeting_id,
                "nro_reuniao": nro_reuniao,
                "data_referencia": data_referencia,
                "data_ata": data_publicacao,
            }
            document_rows.append(
                {
                    "document_id": f"{meeting_id}_ata",
                    "meeting_id": meeting_id,
                    "nro_reuniao": nro_reuniao,
                    "document_type": "ata",
                    "publication_date": data_publicacao,
                    "title": item.get("titulo", ""),
                    "url": item.get("urlPdfAta"),
                    "raw_text": item.get("textoAta", ""),
                    "source": "bcb_copom_atas_detalhes",
                }
            )

    meetings = pd.DataFrame(meeting_rows.values())
    documents = pd.DataFrame(document_rows)
    if meetings.empty:
        raise FetchError("No COPOM meetings returned by BCB endpoints.")
    meetings["data_referencia"] = pd.to_datetime(meetings["data_referencia"])
    if "data_comunicado" not in meetings:
        meetings["data_comunicado"] = meetings["data_referencia"]
    if "data_ata" not in meetings:
        meetings["data_ata"] = pd.NaT
    meetings["data_comunicado"] = pd.to_datetime(meetings["data_comunicado"], errors="coerce")
    meetings["data_ata"] = pd.to_datetime(meetings["data_ata"], errors="coerce")
    meetings = meetings.sort_values("data_referencia").drop_duplicates("meeting_id").reset_index(drop=True)

    documents["publication_date"] = pd.to_datetime(documents["publication_date"], errors="coerce")
    documents = documents.drop_duplicates("document_id").reset_index(drop=True)
    return meetings, documents


def fetch_selic(client: CachedHttpClient, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    params = {
        "formato": "json",
        "dataInicial": start_date.strftime("%d/%m/%Y"),
        "dataFinal": end_date.strftime("%d/%m/%Y"),
    }
    LOGGER.info("Fetching Selic target SGS 432.")
    data = client.get_json(BCB_SGS_432, "sgs_432_selic", params)
    rows = data if isinstance(data, list) else data.get("value", [])
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["date", "selic_target"])
    df = df.rename(columns={"data": "date", "valor": "selic_target"})
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df["selic_target"] = pd.to_numeric(df["selic_target"], errors="coerce")
    df = df.dropna(subset=["date", "selic_target"]).drop_duplicates("date").sort_values("date")
    return df


def attach_selic_to_meetings(meetings: pd.DataFrame, selic: pd.DataFrame) -> pd.DataFrame:
    meetings = meetings.copy()
    meetings["selic_pre"] = np.nan
    meetings["selic_pos"] = np.nan
    if selic.empty:
        meetings["delta_selic"] = np.nan
        return meetings

    selic = selic.sort_values("date").reset_index(drop=True)
    for idx, row in meetings.iterrows():
        reference = row["data_referencia"]
        before = selic[selic["date"] < reference]
        on_or_after = selic[selic["date"] >= reference]
        selic_pre = before.iloc[-1]["selic_target"] if not before.empty else np.nan
        selic_pos = on_or_after.iloc[0]["selic_target"] if not on_or_after.empty else np.nan
        title_rate = _extract_rate_from_documents_title(row.get("titulo_comunicado"))
        if pd.isna(selic_pos) and title_rate is not None:
            selic_pos = title_rate
        meetings.loc[idx, "selic_pre"] = selic_pre
        meetings.loc[idx, "selic_pos"] = selic_pos
    meetings["delta_selic"] = meetings["selic_pos"] - meetings["selic_pre"]
    return meetings


def _extract_rate_from_documents_title(title: Any) -> float | None:
    if not isinstance(title, str):
        return None
    match = re.search(r"(\d{1,2},\d{1,2}|\d{1,2}\.\d{1,2})\s*%", title)
    if not match:
        return None
    return float(match.group(1).replace(",", "."))


def fetch_focus_annual(
    client: CachedHttpClient,
    variables: list[str],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for variable in variables:
        params = {
            "$format": "json",
            "$select": "Indicador,Data,DataReferencia,Media,Mediana,DesvioPadrao,Minimo,Maximo",
            "$filter": (
                f"Indicador eq '{variable}' and Data ge '{start_date.date()}' "
                f"and Data le '{end_date.date()}'"
            ),
            "$orderby": "Data asc",
            "$top": 20000,
        }
        LOGGER.info("Fetching Focus annual expectations for %s.", variable)
        try:
            data = client.get_json(BCB_FOCUS_ANNUAL, f"focus_annual_{variable.lower()}", params)
        except FetchError as exc:
            LOGGER.warning("Focus fetch failed for %s: %s", variable, exc)
            continue
        df = pd.DataFrame(data.get("value", []))
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame(
            columns=["variable", "date", "reference_year", "mean", "median", "std", "minimum", "maximum"]
        )

    focus = pd.concat(frames, ignore_index=True)
    focus = focus.rename(
        columns={
            "Indicador": "variable",
            "Data": "date",
            "DataReferencia": "reference_year",
            "Media": "mean",
            "Mediana": "median",
            "DesvioPadrao": "std",
            "Minimo": "minimum",
            "Maximo": "maximum",
        }
    )
    focus["date"] = pd.to_datetime(focus["date"], errors="coerce")
    focus["reference_year"] = pd.to_numeric(focus["reference_year"], errors="coerce").astype("Int64")
    for col in ["mean", "median", "std", "minimum", "maximum"]:
        focus[col] = pd.to_numeric(focus[col], errors="coerce")
    focus = focus.dropna(subset=["variable", "date", "reference_year", "median"])
    focus = focus.drop_duplicates(["variable", "date", "reference_year"]).sort_values(["variable", "reference_year", "date"])
    return focus


def build_focus_revisions(meetings: pd.DataFrame, focus: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if focus.empty:
        for _, meeting in meetings.iterrows():
            for variable in variables:
                for reference_year in [meeting["data_referencia"].year, meeting["data_referencia"].year + 1]:
                    rows.append(_empty_focus_revision(meeting["meeting_id"], variable, reference_year))
        return pd.DataFrame(rows)

    for _, meeting in meetings.iterrows():
        meeting_date = meeting["data_referencia"]
        comunicado_date = meeting.get("data_comunicado", meeting_date)
        ata_date = meeting.get("data_ata", pd.NaT)
        for variable in variables:
            for reference_year in [meeting_date.year, meeting_date.year + 1]:
                subset = focus[
                    (focus["variable"] == variable)
                    & (focus["reference_year"] == reference_year)
                    & focus["median"].notna()
                ].copy()
                row = _empty_focus_revision(meeting["meeting_id"], variable, int(reference_year))
                row.update(_pick_focus_value(subset, "pre", before=meeting_date))
                row.update(_pick_focus_value(subset, "post_comunicado", after=comunicado_date))
                if pd.notna(ata_date):
                    row.update(_pick_focus_value(subset, "post_ata", after=ata_date))
                row["delta_post_comunicado"] = _safe_delta(row["focus_post_comunicado_value"], row["focus_pre_value"])
                row["delta_post_ata"] = _safe_delta(row["focus_post_ata_value"], row["focus_pre_value"])
                rows.append(row)
    return pd.DataFrame(rows)


def _empty_focus_revision(meeting_id: str, variable: str, reference_year: int) -> dict[str, Any]:
    return {
        "meeting_id": meeting_id,
        "variable": variable,
        "reference_year": reference_year,
        "focus_pre_date": pd.NaT,
        "focus_pre_value": np.nan,
        "focus_post_comunicado_date": pd.NaT,
        "focus_post_comunicado_value": np.nan,
        "focus_post_ata_date": pd.NaT,
        "focus_post_ata_value": np.nan,
        "delta_post_comunicado": np.nan,
        "delta_post_ata": np.nan,
    }


def _pick_focus_value(
    subset: pd.DataFrame,
    label: str,
    before: pd.Timestamp | None = None,
    after: pd.Timestamp | None = None,
) -> dict[str, Any]:
    if subset.empty:
        return {}
    if before is not None:
        eligible = subset[subset["date"] < before]
        if eligible.empty:
            return {}
        selected = eligible.iloc[-1]
    elif after is not None:
        eligible = subset[subset["date"] > after]
        if eligible.empty:
            return {}
        selected = eligible.iloc[0]
    else:
        return {}
    return {f"focus_{label}_date": selected["date"], f"focus_{label}_value": selected["median"]}


def _safe_delta(post: Any, pre: Any) -> float:
    if pd.isna(post) or pd.isna(pre):
        return np.nan
    return float(post) - float(pre)


def date_window_for_sources(meetings: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    min_date = meetings["data_referencia"].min() - timedelta(days=60)
    max_date = max(meetings["data_referencia"].max(), pd.Timestamp(date.today())) + timedelta(days=30)
    return pd.Timestamp(min_date), pd.Timestamp(max_date)

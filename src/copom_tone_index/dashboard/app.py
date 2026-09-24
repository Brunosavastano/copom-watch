from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import plotly.express as px
import streamlit as st

from copom_tone_index.dashboard.english import (
    ask_english_chunks as ask_semantic_chunks,
    english_citation,
    search_english_chunks as search_semantic_chunks,
)
from copom_tone_index.dashboard.ui_copy_en import (
    APP_POSITIONING,
    ASK_EXAMPLES,
    GLOSSARY,
    METHODOLOGY_OVERVIEW,
    PAGE_COPY,
)


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DB = ROOT / "data" / "copom_tone.duckdb"
PUBLIC_DB = ROOT / "app_data" / "copom_watch_public.duckdb"

APP_TABS = (
    ("Latest meeting", "latest"),
    ("Tone over time", "timeline"),
    ("Topic breakdown", "decomposition"),
    ("Text changes", "text_changes"),
    ("Key sentences", "evidence"),
    ("Focus expectations", "focus"),
    ("Market reaction", "market"),
    ("Audit", "audit"),
    ("Questions with evidence", "ask"),
    ("Reports", "reports"),
    ("Classic view", "legacy"),
)

COLUMN_LABELS = {
    "nro_reuniao": "Meeting",
    "data_referencia": "Date",
    "copom_tone_index": "Tone index",
    "copom_tone_index_v2": "Tone index",
    "classification": "Classification",
    "classification_v2": "Classification",
    "communication_surprise_naive": "Simple textual change",
    "directional_intensity": "Directional intensity",
    "calibration_status": "Calibration status",
    "variable": "Variable",
    "reference_year": "Reference year",
    "focus_pre_value": "Pre-event Focus",
    "focus_post_comunicado_value": "Post-statement Focus",
    "focus_post_ata_value": "Post-minutes Focus",
    "delta_post_comunicado": "Post-statement change",
    "delta_post_ata": "Post-minutes change",
    "evidence_type": "Type",
    "document_type": "Document",
    "topic": "Topic",
    "primary_topic": "Topic",
    "score": "Score",
    "confidence": "Confidence",
    "text": "Text",
    "rationale": "Rationale",
    "citation": "Citation",
    "tone_level": "Tone",
    "stance": "Stance",
    "label": "Subindex",
    "tone_raw": "Raw tone",
    "sentence_count": "Sentences",
    "change_type": "Change",
    "similarity": "Similarity",
    "tone_delta": "Tone change",
    "current_text": "Current text",
    "previous_text": "Previous text",
    "event_type": "Event",
    "indicator": "Indicator",
    "horizon": "Horizon",
    "statistic": "Statistic",
    "pre_value": "Pre-event",
    "post_1_value": "First post-event",
    "post_2_value": "Second post-event",
    "delta_post_1": "First post-event change",
    "delta_post_2": "Second post-event change",
    "missing_reason": "Missing-data reason",
    "asset": "Asset",
    "vertex": "Tenor",
    "window": "Window",
    "market_reaction": "Market reaction",
    "status": "Status",
    "known_at_timestamp": "Known at",
    "manifest_name": "Manifest",
    "sha256": "SHA-256",
    "loaded_at": "Loaded at",
    "rank": "Rank",
    "retrieval_method": "Retrieval method",
}

VALUE_LABELS = {
    "Inflation Pressure Index": "Inflation Pressure Index",
    "Expectations Anchoring Index": "Expectations Anchoring Index",
    "Risk Balance Index": "Risk Balance Index",
    "Activity Slack Index": "Activity Slack Index",
    "Fiscal Concern Index": "Fiscal Concern Index",
    "External Constraint Index": "External Constraint Index",
    "Forward Guidance Index": "Forward Guidance Index",
    "Text-Implied Reaction Function Index": "Text-Implied Reaction Function Index",
    "inflation_current": "current inflation",
    "inflation_expectations": "inflation expectations",
    "activity_growth": "activity",
    "labor_market": "labor market",
    "external_environment": "external environment",
    "fx_commodities": "FX and commodities",
    "fiscal_risk": "fiscal risk",
    "policy_decision": "monetary policy decision",
    "forward_guidance": "forward guidance",
    "credit_conditions": "credit conditions",
    "risk_balance": "risk balance",
    "institutional": "institutional",
    "uncertainty": "uncertainty",
    "hawkish": "hawkish",
    "dovish": "dovish",
    "neutral": "neutral",
    "added": "added",
    "removed": "removed",
    "tone_changed": "tone change",
    "rewritten": "rewritten",
    "maintained": "maintained",
    "ready": "ready",
    "partial": "partial",
    "not_available": "unavailable",
    "not_built": "not generated",
    "no_market_data": "no market data",
    "limited_data": "limited data",
    "invalid_for_inference": "invalid for inference",
    "insufficient_market_observations": "insufficient observations",
    "ambiguous_event_timing": "ambiguous event timing",
    "ok": "ok",
    "comunicado": "statement",
    "ata": "minutes",
    "neutro/equilibrado": "neutral/balanced",
    "neutro": "neutral",
    "contracionista": "hawkish",
    "expansionista": "dovish",
    "restritivo": "hawkish",
    "restritiva": "hawkish",
    "muito hawkish": "strongly hawkish",
    "muito dovish": "strongly dovish",
    "moderadamente dovish": "moderately dovish",
    "moderadamente hawkish": "moderately hawkish",
    "neutro / balanceado": "neutral / balanced",
    "claramente dovish": "clearly dovish",
    "claramente hawkish": "clearly hawkish",
}


@st.cache_data(show_spinner=False)
def load_data(database: str) -> dict[str, pd.DataFrame]:
    db_path = Path(database)
    if not db_path.exists():
        return {}
    with duckdb.connect(str(db_path), read_only=True) as con:
        tables = {
            "scores": _read_optional(con, "copom_scores", "SELECT * FROM copom_scores ORDER BY data_referencia"),
            "meetings": _read_optional(con, "copom_meetings", "SELECT * FROM copom_meetings ORDER BY data_referencia"),
            "topic_scores": _read_optional(con, "copom_topic_scores", "SELECT * FROM copom_topic_scores"),
            "evidence": _read_optional(con, "evidence_sentences", "SELECT * FROM evidence_sentences"),
            "focus": _read_optional(con, "focus_revisions", "SELECT * FROM focus_revisions"),
            "v2_scores": _read_optional(con, "v2_meeting_scores", "SELECT * FROM v2_meeting_scores ORDER BY data_referencia"),
            "v2_subindices": _read_optional(con, "v2_subindices", "SELECT * FROM v2_subindices"),
            "v2_evidence": _read_optional(con, "v2_evidence", "SELECT * FROM v2_evidence"),
            "v2_redline": _read_optional(con, "v2_redline", "SELECT * FROM v2_redline"),
            "v2_model_audit": _read_optional(con, "v2_model_audit", "SELECT * FROM v2_model_audit"),
            "v2_model_audit_details": _read_optional(
                con,
                "v2_model_audit_details",
                "SELECT * FROM v2_model_audit_details",
            ),
            "focus_event_features": _read_optional(con, "focus_event_features", "SELECT * FROM focus_event_features"),
            "market_event_windows": _read_optional(con, "market_event_windows", "SELECT * FROM market_event_windows"),
            "decision_expectations": _read_optional(con, "decision_expectations", "SELECT * FROM decision_expectations"),
            "v21_event_panel": _read_optional(con, "v21_event_panel", "SELECT * FROM v21_event_panel"),
            "semantic_chunks": _read_optional(con, "semantic_chunks", "SELECT * FROM semantic_chunks"),
            "app_manifests": _read_optional(con, "app_manifests", "SELECT * FROM app_manifests"),
        }
    for key in ["scores", "meetings", "v2_scores"]:
        if key in tables and "data_referencia" in tables[key]:
            tables[key]["data_referencia"] = pd.to_datetime(tables[key]["data_referencia"])
    return tables


def main() -> None:
    st.set_page_config(page_title="COPOM Watch", layout="wide")
    inject_dark_theme()
    st.title("COPOM Watch")
    st.caption(APP_POSITIONING)
    st.caption("Interface in English. Official quotations remain in their original Portuguese.")

    database = os.getenv("COPOM_TONE_DB", str(default_database_path()))
    data = load_data(database)
    if not data:
        st.warning("DuckDB database not found. Run `copom-watch run-pipeline --use-llm never` before opening the dashboard.")
        st.stop()

    scores = data["scores"]
    topic_scores = data["topic_scores"]
    evidence = data["evidence"]
    focus = data["focus"]
    v2_scores = data.get("v2_scores", pd.DataFrame())
    v2_subindices = data.get("v2_subindices", pd.DataFrame())
    v2_evidence = data.get("v2_evidence", pd.DataFrame())
    v2_redline = data.get("v2_redline", pd.DataFrame())
    v2_model_audit = data.get("v2_model_audit", pd.DataFrame())
    v2_model_audit_details = data.get("v2_model_audit_details", pd.DataFrame())
    focus_event_features = data.get("focus_event_features", pd.DataFrame())
    market_event_windows = data.get("market_event_windows", pd.DataFrame())
    decision_expectations = data.get("decision_expectations", pd.DataFrame())
    v21_event_panel = data.get("v21_event_panel", pd.DataFrame())
    semantic_chunks = data.get("semantic_chunks", pd.DataFrame())
    app_manifests = data.get("app_manifests", pd.DataFrame())

    with st.sidebar:
        st.header("Controls")
        period_choice = st.radio(
            "Analysis period",
            ["Recent operational window", "Full history"],
            index=0,
            help=GLOSSARY["Analysis period"],
        )
        st.toggle(
            "Show technical details",
            value=False,
            key="show_technical_details",
            help="Shows internal audit columns, including identifiers, versions and control fields.",
        )
        operational_only = period_choice == "Recent operational window"
        meeting_source = v2_scores if not v2_scores.empty else scores
        if operational_only and "in_operational_window" in meeting_source:
            scores_view = meeting_source[meeting_source["in_operational_window"]].copy()
        else:
            scores_view = meeting_source.copy()
        if scores_view.empty:
            st.warning("No indicators are available in the selected database.")
            st.stop()
        meeting_labels = scores_view.sort_values("data_referencia", ascending=False).apply(
            lambda row: f"{int(row['nro_reuniao'])} - {pd.Timestamp(row['data_referencia']).date()}",
            axis=1,
        )
        selected_label = st.selectbox("Meeting", meeting_labels.tolist())
        selected_meeting = scores_view.loc[meeting_labels[meeting_labels == selected_label].index[0], "meeting_id"]

    if not v2_scores.empty:
        v2_selected = v2_scores[v2_scores["meeting_id"] == selected_meeting]
        if not v2_selected.empty:
            legacy_latest = scores[scores["meeting_id"] == selected_meeting].iloc[0] if not scores.empty and not scores[scores["meeting_id"] == selected_meeting].empty else pd.Series(dtype=object)
            _render_v22_product_dashboard(
                v2_selected.iloc[0],
                v2_scores,
                scores,
                topic_scores,
                evidence,
                focus,
                v2_subindices,
                v2_evidence,
                v2_redline,
                v2_model_audit,
                v2_model_audit_details,
                focus_event_features,
                market_event_windows,
                decision_expectations,
                v21_event_panel,
                semantic_chunks,
                app_manifests,
                legacy_latest,
            )
            return

    latest = scores_view[scores_view["meeting_id"] == selected_meeting].iloc[0]
    _render_legacy_v1_dashboard(scores_view, topic_scores, evidence, focus, latest, selected_meeting)


def default_database_path() -> Path:
    if PUBLIC_DB.exists():
        return PUBLIC_DB
    return DEFAULT_DB


def inject_dark_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --copom-bg: #090d12;
            --copom-panel: #111827;
            --copom-panel-2: #0f1720;
            --copom-border: #273244;
            --copom-text: #e5edf5;
            --copom-muted: #9aa8b8;
            --copom-blue: #38bdf8;
            --copom-green: #34d399;
            --copom-amber: #fbbf24;
            --copom-red: #fb7185;
        }
        .stApp {
            background: var(--copom-bg);
            color: var(--copom-text);
        }
        [data-testid="stSidebar"] {
            background: #070b10;
            border-right: 1px solid var(--copom-border);
        }
        h1, h2, h3, h4, h5, h6, p, label, span, div {
            color: var(--copom-text);
        }
        .stCaptionContainer, .stMarkdown small, [data-testid="stCaptionContainer"] {
            color: var(--copom-muted);
        }
        [data-testid="stMetric"] {
            background: linear-gradient(180deg, var(--copom-panel), var(--copom-panel-2));
            border: 1px solid var(--copom-border);
            border-radius: 8px;
            padding: 14px 16px;
        }
        [data-testid="stMetricLabel"] p {
            color: var(--copom-muted);
            font-size: 0.78rem;
            letter-spacing: 0;
        }
        [data-testid="stMetricValue"] {
            color: var(--copom-text);
        }
        [data-testid="stTabs"] button {
            color: var(--copom-muted);
        }
        [data-testid="stTabs"] button[aria-selected="true"] {
            color: var(--copom-blue);
            border-bottom-color: var(--copom-blue);
        }
        [data-testid="stDataFrame"], [data-testid="stTable"] {
            border: 1px solid var(--copom-border);
            border-radius: 8px;
            overflow: hidden;
        }
        .stAlert {
            background: #101820;
            border: 1px solid var(--copom-border);
            color: var(--copom-text);
        }
        a {
            color: var(--copom-blue);
        }
        .copom-guide {
            background: linear-gradient(180deg, rgba(17, 24, 39, 0.96), rgba(15, 23, 32, 0.96));
            border: 1px solid var(--copom-border);
            border-left: 3px solid var(--copom-blue);
            border-radius: 8px;
            padding: 16px 18px;
            margin: 10px 0 18px 0;
        }
        .copom-guide h3 {
            margin: 0 0 8px 0;
        }
        .copom-guide p {
            color: var(--copom-muted);
            margin: 0 0 10px 0;
        }
        .copom-help {
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }
        .copom-help-icon {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 16px;
            height: 16px;
            border: 1px solid var(--copom-blue);
            border-radius: 50%;
            color: var(--copom-blue);
            font-size: 10px;
            font-weight: 700;
            cursor: help;
        }
        .copom-methodology {
            border: 1px solid var(--copom-border);
            border-radius: 8px;
            padding: 14px 16px;
            background: rgba(56, 189, 248, 0.06);
        }
        .copom-metric-card {
            background: linear-gradient(180deg, var(--copom-panel), var(--copom-panel-2));
            border: 1px solid var(--copom-border);
            border-radius: 8px;
            padding: 14px 16px;
            min-height: 118px;
        }
        .copom-metric-label {
            color: var(--copom-muted);
            font-size: 0.78rem;
            margin-bottom: 8px;
        }
        .copom-metric-value {
            color: var(--copom-text);
            font-size: 1.72rem;
            font-weight: 700;
            line-height: 1.1;
        }
        .copom-metric-delta {
            color: var(--copom-muted);
            font-size: 0.86rem;
            margin-top: 6px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_legacy_v1_dashboard(
    scores_view: pd.DataFrame,
    topic_scores: pd.DataFrame,
    evidence: pd.DataFrame,
    focus: pd.DataFrame,
    latest: pd.Series,
    selected_meeting: str,
) -> None:
    st.subheader("Classic view")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Index", _fmt(latest["copom_tone_index"]), _fmt(latest.get("classification", "")))
    c2.metric("Tone change", _fmt(latest.get("delta_tone")))
    c3.metric("Post-meeting Selic", _fmt(latest.get("selic_pos")))
    c4.metric("Selic change", _fmt(latest.get("delta_selic")))

    st.subheader("Time series")
    chart_scores = scores_view.copy()
    chart_scores["classification"] = chart_scores["classification"].map(_value_label)
    fig = px.line(
        chart_scores,
        x="data_referencia",
        y="copom_tone_index",
        markers=True,
        hover_data=["nro_reuniao", "classification", "tone_comunicado", "tone_ata", "delta_selic"],
        labels={**COLUMN_LABELS, "data_referencia": "Date", "copom_tone_index": "Tone index"},
    )
    fig.add_hline(y=50, line_dash="dash", line_color="gray")
    fig.add_hrect(y0=45, y1=55, line_width=0, fillcolor="gray", opacity=0.08)
    st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns([1.1, 1])
    with left:
        st.subheader("Topic breakdown")
        topic_view = topic_scores[topic_scores["meeting_id"].isin(scores_view["meeting_id"])]
        if topic_view.empty:
            st.info("No classified topics.")
        else:
            heatmap = topic_view.pivot_table(index="topic", columns="meeting_id", values="topic_tone", aggfunc="mean")
            fig = px.imshow(
                heatmap,
                aspect="auto",
                color_continuous_scale="RdBu_r",
                zmin=-1,
                zmax=1,
                labels={"color": "Tone"},
            )
            st.plotly_chart(fig, use_container_width=True)
    with right:
        st.subheader("Focus revisions")
        focus_view = focus[focus["meeting_id"] == selected_meeting].copy()
        if focus_view.empty:
            st.info("Focus data are unavailable for this meeting.")
        else:
            display_cols = [
                "variable",
                "reference_year",
                "focus_pre_value",
                "focus_post_comunicado_value",
                "focus_post_ata_value",
                "delta_post_comunicado",
                "delta_post_ata",
            ]
            st.dataframe(_display_frame(focus_view[display_cols]), use_container_width=True, hide_index=True)

    st.subheader("Textual evidence")
    ev = evidence[evidence["meeting_id"] == selected_meeting].copy()
    if ev.empty:
        st.info("No classified evidence for this meeting.")
    else:
        ev["score"] = ev["stance_score"].map(lambda value: f"{value:.2f}")
        st.dataframe(
            _display_frame(ev[["evidence_type", "document_type", "topic", "score", "confidence", "text", "rationale"]]),
            use_container_width=True,
            hide_index=True,
        )

    with st.expander("Indicator table"):
        st.dataframe(_display_frame(scores_view), use_container_width=True, hide_index=True)


def _fmt(value: object) -> str:
    if pd.isna(value):
        return "n/a"
    if isinstance(value, str):
        return str(_value_label(value))
    return f"{float(value):.2f}"


def _status_label(value: object) -> str:
    if pd.isna(value):
        return "n/a"
    text = str(value)
    return _value_label(text)


def _html_escape(value: object) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#x27;")
    )


def help_label(term: str, text: str | None = None) -> str:
    label = text or term
    tooltip = GLOSSARY.get(term, "")
    if not tooltip:
        return _html_escape(label)
    return (
        "<span class='copom-help'>"
        f"{_html_escape(label)}"
        f"<span class='copom-help-icon' title='{_html_escape(tooltip)}'>i</span>"
        "</span>"
    )


def render_help_heading(term: str, level: int = 3, text: str | None = None) -> None:
    st.markdown(f"<h{level}>{help_label(term, text)}</h{level}>", unsafe_allow_html=True)


def render_metric_card(container: Any, term: str, value: object, delta: object | None = None, label: str | None = None) -> None:
    delta_html = f"<div class='copom-metric-delta'>{_html_escape(delta)}</div>" if delta not in (None, "") else ""
    container.markdown(
        "\n".join(
            [
                "<div class='copom-metric-card'>",
                f"<div class='copom-metric-label'>{help_label(term, label or term)}</div>",
                f"<div class='copom-metric-value'>{_html_escape(value)}</div>",
                delta_html,
                "</div>",
            ]
        ),
        unsafe_allow_html=True,
    )


def render_page_intro(page_key: str) -> None:
    page = PAGE_COPY[page_key]
    st.markdown(
        "\n".join(
            [
                "<div class='copom-guide'>",
                "<h3>What this page explains</h3>",
                f"<p>{_html_escape(page.question)}</p>",
                "<strong>How to read</strong>",
                "<ul>",
                *[f"<li>{_html_escape(item)}</li>" for item in page.how_to_read],
                "</ul>",
                "<strong>Limitations</strong>",
                "<ul>",
                *[f"<li>{_html_escape(item)}</li>" for item in page.limitations],
                "</ul>",
                "</div>",
            ]
        ),
        unsafe_allow_html=True,
    )
    with st.expander("Page glossary"):
        for term in page.glossary_terms:
            st.markdown(f"**{term}**: {GLOSSARY[term]}")


def render_methodology_overview() -> None:
    with st.expander("Methodology in plain language", expanded=False):
        st.markdown(f"<div class='copom-methodology'>{_html_escape(METHODOLOGY_OVERVIEW)}</div>", unsafe_allow_html=True)
        st.markdown(
            "- The texts come from official Copom statements and minutes.\n"
            "- The index summarizes the direction of the language, not the policy decision.\n"
            "- Text changes, Focus expectations and markets provide additional context.\n"
            "- The app does not forecast Selic or establish causality in event windows."
        )


def _value_label(value: object) -> object:
    if pd.isna(value):
        return value
    text = str(value)
    return VALUE_LABELS.get(text, text.replace("_", " "))


def _display_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    display = frame.copy()
    for column in display.select_dtypes(include="object").columns:
        if column in {"text", "current_text", "previous_text", "rationale", "source_hash", "sha256"}:
            continue
        if column == "citation":
            display[column] = display[column].map(english_citation)
        else:
            display[column] = display[column].map(_value_label)
    return display.rename(columns={column: COLUMN_LABELS.get(column, column) for column in display.columns})


def _show_technical_details() -> bool:
    return bool(st.session_state.get("show_technical_details", False))


def friendly_report_name(path: Path) -> str:
    name = path.stem
    if "acceptance" in name:
        return "Acceptance report"
    if "release" in name:
        return "Release summary"
    if "macro_market" in name:
        return "Macro and market report"
    if "semantic" in name:
        return "Evidence search report"
    if "benchmark" in name:
        return "Methodology benchmark"
    if "audit" in name:
        return "Methodology audit"
    if "copom_watch" in name:
        return "Meeting report"
    return name.replace("_", " ").capitalize()


def _read_optional(con: duckdb.DuckDBPyConnection, table: str, query: str) -> pd.DataFrame:
    exists = con.execute(
        "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
        [table],
    ).fetchone()[0]
    if not exists:
        return pd.DataFrame()
    return con.execute(query).df()


def _render_v22_product_dashboard(
    latest: pd.Series,
    v2_scores: pd.DataFrame,
    scores: pd.DataFrame,
    topic_scores: pd.DataFrame,
    evidence_v1: pd.DataFrame,
    focus_v1: pd.DataFrame,
    subindices: pd.DataFrame,
    evidence: pd.DataFrame,
    redline: pd.DataFrame,
    model_audit: pd.DataFrame,
    model_audit_details: pd.DataFrame,
    focus_event_features: pd.DataFrame,
    market_event_windows: pd.DataFrame,
    decision_expectations: pd.DataFrame,
    v21_event_panel: pd.DataFrame,
    semantic_chunks: pd.DataFrame,
    app_manifests: pd.DataFrame,
    legacy_latest: pd.Series,
) -> None:
    render_methodology_overview()
    tabs = st.tabs([label for label, _ in APP_TABS])
    with tabs[0]:
        render_page_intro("latest")
        _render_v2_cockpit(latest, subindices, evidence, redline)
        _render_v22_status_cards(v21_event_panel, app_manifests)
    with tabs[1]:
        render_page_intro("timeline")
        _render_v2_timeline(v2_scores)
    with tabs[2]:
        render_page_intro("decomposition")
        _render_v2_topic_decomposition(v2_scores, subindices)
    with tabs[3]:
        render_page_intro("text_changes")
        _render_v2_redline_explorer(latest, redline)
    with tabs[4]:
        render_page_intro("evidence")
        _render_v2_evidence_explorer(latest, evidence)
    with tabs[5]:
        render_page_intro("focus")
        _render_v21_focus_monitor(latest, focus_event_features, v21_event_panel)
    with tabs[6]:
        render_page_intro("market")
        _render_v21_market_reaction(latest, market_event_windows, decision_expectations, v21_event_panel)
    with tabs[7]:
        render_page_intro("audit")
        _render_v2_audit(model_audit, model_audit_details)
    with tabs[8]:
        render_page_intro("ask")
        _render_v2_ask_copom_watch(semantic_chunks)
    with tabs[9]:
        render_page_intro("reports")
        _render_reports_panel(app_manifests)
    with tabs[10]:
        render_page_intro("legacy")
        if legacy_latest.empty:
            st.info("Classic view is unavailable in the current data package.")
        else:
            _render_legacy_v1_dashboard(scores, topic_scores, evidence_v1, focus_v1, legacy_latest, str(latest["meeting_id"]))


def _render_v22_status_cards(v21_event_panel: pd.DataFrame, app_manifests: pd.DataFrame) -> None:
    st.markdown("**Product status**")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Text index", "frozen")
    c2.metric("Expectations and markets", "frozen")
    c3.metric("Events analyzed", f"{len(v21_event_panel):,}" if not v21_event_panel.empty else "n/a")
    c4.metric("Manifests", str(len(app_manifests)) if not app_manifests.empty else "n/a")


def _render_v2_timeline(v2_scores: pd.DataFrame) -> None:
    if v2_scores.empty:
        st.info("Time series unavailable.")
        return
    y_col = "copom_tone_index_v2" if "copom_tone_index_v2" in v2_scores else "tone_raw"
    hover = [col for col in ["nro_reuniao", "classification_v2", "communication_surprise_naive", "directional_intensity"] if col in v2_scores]
    chart_scores = v2_scores.copy()
    if "classification_v2" in chart_scores:
        chart_scores["classification_v2"] = chart_scores["classification_v2"].map(_value_label)
    fig = px.line(
        chart_scores.sort_values("data_referencia"),
        x="data_referencia",
        y=y_col,
        markers=True,
        hover_data=hover,
        labels={**COLUMN_LABELS, "data_referencia": "Date", y_col: "Tone index"},
    )
    fig.add_hline(y=0 if y_col == "tone_raw" else 50, line_dash="dash", line_color="gray")
    st.plotly_chart(fig, use_container_width=True)
    timeline_cols = [
        col
        for col in [
            "nro_reuniao",
            "data_referencia",
            y_col,
            "classification_v2",
            "communication_surprise_naive",
            "directional_intensity",
            "calibration_status",
        ]
        if col in v2_scores
    ]
    st.dataframe(
        _display_frame(v2_scores.sort_values("data_referencia", ascending=False).head(80)[timeline_cols]),
        use_container_width=True,
        hide_index=True,
    )


def _render_v2_topic_decomposition(v2_scores: pd.DataFrame, subindices: pd.DataFrame) -> None:
    if subindices.empty:
        st.info("Subindices unavailable.")
        return
    c1, c2 = st.columns([1, 2])
    labels = sorted(subindices.get("label", pd.Series(dtype=str)).dropna().astype(str).unique())
    label_options = {str(_value_label(label)): label for label in labels}
    selected_display = c1.multiselect(
        "Subindices",
        list(label_options.keys()),
        default=list(label_options.keys()),
        help=GLOSSARY["Subindices"],
    )
    selected = [label_options[label] for label in selected_display]
    filtered = subindices[subindices["label"].isin(selected)].copy() if selected else subindices.copy()
    if "label" in filtered:
        filtered["label"] = filtered["label"].map(_value_label)
    if "data_referencia" not in filtered and "meeting_id" in filtered and "meeting_id" in v2_scores:
        filtered = filtered.merge(v2_scores[["meeting_id", "data_referencia"]], on="meeting_id", how="left")
    with c2:
        if not filtered.empty and {"data_referencia", "tone_raw", "label"}.issubset(filtered.columns):
            fig = px.line(filtered.sort_values("data_referencia"), x="data_referencia", y="tone_raw", color="label", labels={**COLUMN_LABELS, "tone_raw": "Raw tone"})
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            st.plotly_chart(fig, use_container_width=True)
    if not _show_technical_details():
        display_cols = [col for col in ["data_referencia", "label", "tone_raw", "sentence_count"] if col in filtered]
        filtered = filtered[display_cols]
    st.dataframe(_display_frame(filtered), use_container_width=True, hide_index=True)


def _render_v21_focus_monitor(latest: pd.Series, focus_event_features: pd.DataFrame, v21_event_panel: pd.DataFrame) -> None:
    _render_v21_optional_panel(latest, focus_event_features, pd.DataFrame(), pd.DataFrame(), v21_event_panel)


def _render_v21_market_reaction(
    latest: pd.Series,
    market_event_windows: pd.DataFrame,
    decision_expectations: pd.DataFrame,
    v21_event_panel: pd.DataFrame,
) -> None:
    _render_v21_optional_panel(latest, pd.DataFrame(), market_event_windows, decision_expectations, v21_event_panel)


def _render_v2_ask_copom_watch(semantic_chunks: pd.DataFrame) -> None:
    if semantic_chunks.empty:
        st.info("Local semantic index unavailable. Run `copom-watch semantic build-index --method tfidf`.")
        return
    render_help_heading("Evidence search", level=3, text="Questions with official evidence")
    st.markdown(
        "Search language, topics and historical cases in official Copom documents. "
        "Answers must remain grounded in the retrieved citations. "
        "Questions that ask for a Selic forecast receive an explanation of the app's scope."
    )
    st.caption("Example questions")
    example_cols = st.columns(2)
    for index, example in enumerate(ASK_EXAMPLES):
        if example_cols[index % 2].button(example, key=f"ask_example_{index}"):
            st.session_state["ask_query"] = example
    with st.form("v22_ask_form"):
        query = st.text_input(
            "Ask COPOM Watch",
            value=st.session_state.get("ask_query", ""),
            placeholder="E.g. When did Copom discuss unanchored expectations?",
            help=GLOSSARY["Evidence search"],
        )
        top_n = st.number_input("Citations", min_value=3, max_value=12, value=8, step=1, help=GLOSSARY["Key sentences"])
        submitted = st.form_submit_button("Answer with citations")
    if not submitted:
        return
    answer, citations = ask_semantic_chunks(query, semantic_chunks, top_n=int(top_n), method="tfidf")
    if not answer:
        st.info("No citations found for this question.")
        return
    st.markdown(answer.replace("\n", "\n\n"))
    st.dataframe(_display_frame(citations), use_container_width=True, hide_index=True)


def _render_reports_panel(app_manifests: pd.DataFrame) -> None:
    st.markdown("**Release manifests and reports**")
    if not app_manifests.empty:
        cols = [col for col in ["manifest_name", "sha256", "loaded_at"] if col in app_manifests]
        manifest_view = app_manifests[cols].copy()
        if "manifest_name" in manifest_view:
            manifest_view["manifest_name"] = manifest_view["manifest_name"].map(lambda value: friendly_report_name(Path(str(value))))
        st.dataframe(_display_frame(manifest_view), use_container_width=True, hide_index=True)
    report_dir = ROOT / "reports" / "v2"
    reports = sorted(report_dir.glob("*.html")) if report_dir.exists() else []
    if reports:
        report_rows = pd.DataFrame({"Report": [friendly_report_name(path) for path in reports]})
        st.dataframe(report_rows, use_container_width=True, hide_index=True)
    else:
        st.info("Local HTML reports are not available in this environment.")


def _render_v2_dashboard(
    latest: pd.Series,
    subindices: pd.DataFrame,
    evidence: pd.DataFrame,
    redline: pd.DataFrame,
    model_audit: pd.DataFrame,
    model_audit_details: pd.DataFrame,
    focus_event_features: pd.DataFrame,
    market_event_windows: pd.DataFrame,
    decision_expectations: pd.DataFrame,
    v21_event_panel: pd.DataFrame,
    semantic_chunks: pd.DataFrame,
) -> None:
    st.subheader("Latest meeting dashboard")
    overview_tab, evidence_tab, redline_tab, v21_tab, search_tab, audit_tab = st.tabs(
        ["Dashboard", "Evidence", "Text change", "Macro and markets", "Search", "Audit"]
    )
    with overview_tab:
        _render_v2_cockpit(latest, subindices, evidence, redline)
    with evidence_tab:
        _render_v2_evidence_explorer(latest, evidence)
    with redline_tab:
        _render_v2_redline_explorer(latest, redline)
    with v21_tab:
        _render_v21_optional_panel(latest, focus_event_features, market_event_windows, decision_expectations, v21_event_panel)
    with search_tab:
        _render_v2_semantic_search(semantic_chunks)
    with audit_tab:
        _render_v2_audit(model_audit, model_audit_details)


def _render_v2_cockpit(
    latest: pd.Series,
    subindices: pd.DataFrame,
    evidence: pd.DataFrame,
    redline: pd.DataFrame,
) -> None:
    c1, c2, c3, c4, c5 = st.columns(5)
    render_metric_card(c1, "Raw tone", _fmt(latest.get("tone_raw")))
    render_metric_card(c2, "Tone index", _fmt(latest.get("copom_tone_index_v2")), _fmt(latest.get("classification_v2", "")))
    render_metric_card(c3, "Textual surprise", _fmt(latest.get("communication_surprise_naive")))
    render_metric_card(c4, "Intensity", _fmt(latest.get("directional_intensity")))
    render_metric_card(c5, "Calibration", str(latest.get("calibration_status", "n/a")))

    meeting_id = latest["meeting_id"]
    bullets = _v2_bullets(latest, subindices[subindices["meeting_id"] == meeting_id])
    for bullet in bullets:
        st.write(f"- {bullet}")

    left, right = st.columns([1, 1])
    with left:
        render_help_heading("Subindices", level=3)
        sub = subindices[subindices["meeting_id"] == meeting_id].copy()
        if sub.empty:
            st.info("Subindices unavailable.")
        else:
            st.dataframe(_display_frame(sub[["label", "tone_raw", "sentence_count"]]), use_container_width=True, hide_index=True)
    with right:
        render_help_heading("Text change", level=3, text="What changed")
        changes = redline[(redline["meeting_id"] == meeting_id) & (redline["change_type"].isin(["added", "tone_changed"]))].copy()
        if changes.empty:
            st.info("Text changes are unavailable for this meeting.")
        else:
            st.dataframe(_display_frame(changes[["document_type", "change_type", "tone_delta", "current_text"]].head(8)), use_container_width=True, hide_index=True)

    ev = evidence[evidence["meeting_id"] == meeting_id].copy()
    if not ev.empty:
        with st.expander("Most relevant key sentences"):
            evidence_cols = [col for col in ["evidence_type", "citation", "document_type", "primary_topic", "tone_level", "text"] if col in ev]
            st.dataframe(_display_frame(ev[evidence_cols]), use_container_width=True, hide_index=True)


def _render_v2_evidence_explorer(latest: pd.Series, evidence: pd.DataFrame) -> None:
    meeting_id = latest["meeting_id"]
    ev = evidence[evidence["meeting_id"] == meeting_id].copy()
    if ev.empty:
        st.info("No evidence for this meeting.")
        return
    c1, c2, c3 = st.columns(3)
    evidence_types = sorted(ev.get("evidence_type", pd.Series(dtype=str)).dropna().astype(str).unique())
    document_types = sorted(ev.get("document_type", pd.Series(dtype=str)).dropna().astype(str).unique())
    topics = sorted(ev.get("primary_topic", pd.Series(dtype=str)).dropna().astype(str).unique())
    selected_evidence = c1.multiselect("Type", evidence_types, default=evidence_types, key="v2_evidence_type_filter", format_func=_value_label)
    selected_documents = c2.multiselect("Document", document_types, default=document_types, key="v2_evidence_document_filter", format_func=_value_label)
    selected_topics = c3.multiselect("Topic", topics, default=topics, key="v2_evidence_topic_filter", format_func=_value_label)
    filtered = filter_v2_evidence(ev, meeting_id, selected_evidence, selected_documents, selected_topics)
    cols = [
        col
        for col in ["evidence_type", "citation", "document_type", "primary_topic", "stance", "tone_level", "confidence", "text", "rationale"]
        if col in filtered
    ]
    st.dataframe(_display_frame(filtered[cols]), use_container_width=True, hide_index=True)


def _render_v2_redline_explorer(latest: pd.Series, redline: pd.DataFrame) -> None:
    meeting_id = latest["meeting_id"]
    changes = redline[redline["meeting_id"] == meeting_id].copy()
    if changes.empty:
        st.info("No text changes for this meeting.")
        return
    c1, c2 = st.columns(2)
    change_types = sorted(changes.get("change_type", pd.Series(dtype=str)).dropna().astype(str).unique())
    document_types = sorted(changes.get("document_type", pd.Series(dtype=str)).dropna().astype(str).unique())
    default_changes = [item for item in ["added", "tone_changed", "rewritten", "removed"] if item in change_types] or change_types
    selected_changes = c1.multiselect("Change", change_types, default=default_changes, key="v2_redline_change_filter", format_func=_value_label)
    selected_documents = c2.multiselect("Document", document_types, default=document_types, key="v2_redline_document_filter", format_func=_value_label)
    filtered = filter_v2_redline(changes, meeting_id, selected_changes, selected_documents)
    cols = [
        col
        for col in ["document_type", "change_type", "similarity", "tone_delta", "current_text", "previous_text"]
        if col in filtered
    ]
    st.dataframe(_display_frame(filtered[cols]), use_container_width=True, hide_index=True)


def _render_v2_audit(model_audit: pd.DataFrame, model_audit_details: pd.DataFrame) -> None:
    if model_audit.empty:
        st.info("Audit not yet generated.")
        return
    metrics = model_audit.copy()
    if "value" in metrics:
        metrics["value"] = metrics["value"].map(lambda value: _fmt(value) if pd.notna(value) else "n/a")
    st.dataframe(_display_frame(metrics), use_container_width=True, hide_index=True)
    if model_audit_details.empty:
        st.info("Audit details are unavailable because no human labels have been accepted yet.")
        return
    detail_cols = [
        col
        for col in [
            "sentence_id",
            "annotator_id",
            "stance_label",
            "predicted_stance",
            "stance_correct",
            "topic_label",
            "predicted_topic",
            "topic_correct",
        ]
        if col in model_audit_details
    ]
    st.dataframe(_display_frame(model_audit_details[detail_cols]), use_container_width=True, hide_index=True)


def _render_v2_semantic_search(semantic_chunks: pd.DataFrame) -> None:
    if semantic_chunks.empty:
        st.info("Local semantic index unavailable. Run `copom-watch semantic build-index`.")
        return
    with st.form("v2_semantic_search_form"):
        c1, c2 = st.columns([3, 1])
        query = c1.text_input(
            "Local semantic search",
            value="",
            placeholder="E.g. unanchored expectations",
            key="v2_semantic_query",
        )
        top_n = c2.number_input("Results", min_value=3, max_value=25, value=10, step=1, key="v2_semantic_top_n")
        submitted = st.form_submit_button("Search")
    if not submitted:
        st.info("Enter a question and select Search to find historical sentences with citations.")
        return
    if not query.strip():
        st.info("Enter a question to search historical sentences with citations.")
        return
    results = semantic_search_for_dashboard(query, semantic_chunks, top_n=int(top_n))
    if results.empty:
        st.info("No results found for this question.")
        return
    display_cols = [
        col
        for col in ["rank", "score", "citation", "nro_reuniao", "document_type", "sentence_id", "text"]
        if col in results
    ]
    st.dataframe(_display_frame(results[display_cols]), use_container_width=True, hide_index=True)


def _render_v21_optional_panel(
    latest: pd.Series,
    focus_event_features: pd.DataFrame,
    market_event_windows: pd.DataFrame,
    decision_expectations: pd.DataFrame,
    v21_event_panel: pd.DataFrame,
) -> None:
    meeting_id = str(latest["meeting_id"])
    panel_row = (
        v21_event_panel[v21_event_panel["meeting_id"].astype(str) == meeting_id].iloc[0]
        if not v21_event_panel.empty and "meeting_id" in v21_event_panel and not v21_event_panel[v21_event_panel["meeting_id"].astype(str) == meeting_id].empty
        else pd.Series(dtype=object)
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Decision surprise", _fmt(panel_row.get("decision_surprise_bps") if not panel_row.empty else pd.NA))
    c2.metric("Focus status", _status_label(panel_row.get("focus_status", "not_available") if not panel_row.empty else "not_built"))
    c3.metric("Market status", _status_label(panel_row.get("market_status", "no_market_data") if not panel_row.empty else "not_built"))
    c4.metric("Valid market windows", _fmt(panel_row.get("market_ok_windows") if not panel_row.empty else pd.NA))

    st.caption("Descriptive analysis of decisions, Focus expectations and market moves in event windows; no causal inference.")
    left, right = st.columns(2)
    with left:
        st.markdown("**Focus monitor**")
        focus_view = (
            focus_event_features[focus_event_features["meeting_id"].astype(str) == meeting_id].copy()
            if not focus_event_features.empty and "meeting_id" in focus_event_features
            else pd.DataFrame()
        )
        if focus_view.empty:
            st.info("Focus data are unavailable. Refresh the Focus data before using this section.")
        else:
            cols = [
                col
                for col in [
                    "event_type",
                    "indicator",
                    "horizon",
                    "statistic",
                    "pre_value",
                    "post_1_value",
                    "post_2_value",
                    "delta_post_1",
                    "delta_post_2",
                    "missing_reason",
                ]
                if col in focus_view
            ]
            st.dataframe(_display_frame(focus_view[cols].head(80)), use_container_width=True, hide_index=True)
    with right:
        st.markdown("**Market reaction**")
        market_view = (
            market_event_windows[market_event_windows["meeting_id"].astype(str) == meeting_id].copy()
            if not market_event_windows.empty and "meeting_id" in market_event_windows
            else pd.DataFrame()
        )
        if market_view.empty:
            st.info("Optional market data are unavailable. Import a CSV and run `copom-watch market event-study`.")
        else:
            cols = [
                col
                for col in ["document_type", "asset", "vertex", "window", "market_reaction", "status", "known_at_timestamp"]
                if col in market_view
            ]
            st.dataframe(_display_frame(market_view[cols].head(80)), use_container_width=True, hide_index=True)
    if not decision_expectations.empty:
        with st.expander("Imported decision expectations"):
            view = (
                decision_expectations[decision_expectations["meeting_id"].astype(str) == meeting_id].copy()
                if "meeting_id" in decision_expectations
                else decision_expectations.copy()
            )
            st.dataframe(_display_frame(view), use_container_width=True, hide_index=True)


def filter_v2_evidence(
    evidence: pd.DataFrame,
    meeting_id: str,
    evidence_types: list[str] | None = None,
    document_types: list[str] | None = None,
    topics: list[str] | None = None,
) -> pd.DataFrame:
    filtered = evidence[evidence["meeting_id"] == meeting_id].copy() if "meeting_id" in evidence else pd.DataFrame()
    if filtered.empty:
        return filtered
    if evidence_types and "evidence_type" in filtered:
        filtered = filtered[filtered["evidence_type"].isin(evidence_types)]
    if document_types and "document_type" in filtered:
        filtered = filtered[filtered["document_type"].isin(document_types)]
    if topics and "primary_topic" in filtered:
        filtered = filtered[filtered["primary_topic"].isin(topics)]
    sort_cols = [col for col in ["evidence_type", "document_type", "tone_level"] if col in filtered]
    ascending = [True, True, False][: len(sort_cols)]
    return filtered.sort_values(sort_cols, ascending=ascending) if sort_cols else filtered


def semantic_search_for_dashboard(query: str, semantic_chunks: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    results = search_semantic_chunks(query, semantic_chunks, top_n=top_n)
    if results.empty:
        return results
    required = ["citation", "sentence_id", "text"]
    missing = [column for column in required if column not in results]
    if missing:
        return pd.DataFrame()
    return results


def filter_v2_redline(
    redline: pd.DataFrame,
    meeting_id: str,
    change_types: list[str] | None = None,
    document_types: list[str] | None = None,
) -> pd.DataFrame:
    filtered = redline[redline["meeting_id"] == meeting_id].copy() if "meeting_id" in redline else pd.DataFrame()
    if filtered.empty:
        return filtered
    if change_types and "change_type" in filtered:
        filtered = filtered[filtered["change_type"].isin(change_types)]
    if document_types and "document_type" in filtered:
        filtered = filtered[filtered["document_type"].isin(document_types)]
    sort_cols = [col for col in ["document_type", "change_type", "tone_delta"] if col in filtered]
    ascending = [True, True, False][: len(sort_cols)]
    return filtered.sort_values(sort_cols, ascending=ascending) if sort_cols else filtered


def _v2_bullets(latest: pd.Series, subindices: pd.DataFrame) -> list[str]:
    bullets = []
    surprise = latest.get("communication_surprise_naive")
    if pd.notna(surprise):
        direction = "more hawkish" if float(surprise) > 0 else "more dovish" if float(surprise) < 0 else "unchanged"
        bullets.append(f"The simple textual change was {direction} compared with the previous meeting.")
    if not subindices.empty and subindices["tone_raw"].notna().any():
        strongest = subindices.loc[subindices["tone_raw"].abs().idxmax()]
        bullets.append(f"The largest topic contribution came from {_value_label(strongest['label'])}.")
    bullets.append("The analysis separates tone level from new wording and preserves model, taxonomy and calibration versions.")
    return bullets


if __name__ == "__main__":
    main()

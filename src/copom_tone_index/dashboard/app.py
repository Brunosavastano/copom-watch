from __future__ import annotations

import os
from pathlib import Path

import duckdb
import pandas as pd
import plotly.express as px
import streamlit as st


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DB = ROOT / "data" / "copom_tone.duckdb"


@st.cache_data(show_spinner=False)
def load_data(database: str) -> dict[str, pd.DataFrame]:
    db_path = Path(database)
    if not db_path.exists():
        return {}
    with duckdb.connect(str(db_path), read_only=True) as con:
        tables = {
            "scores": con.execute("SELECT * FROM copom_scores ORDER BY data_referencia").df(),
            "meetings": con.execute("SELECT * FROM copom_meetings ORDER BY data_referencia").df(),
            "topic_scores": con.execute("SELECT * FROM copom_topic_scores").df(),
            "evidence": con.execute("SELECT * FROM evidence_sentences").df(),
            "focus": con.execute("SELECT * FROM focus_revisions").df(),
        }
    for key in ["scores", "meetings"]:
        if key in tables and "data_referencia" in tables[key]:
            tables[key]["data_referencia"] = pd.to_datetime(tables[key]["data_referencia"])
    return tables


def main() -> None:
    st.set_page_config(page_title="COPOM Tone Index", layout="wide")
    st.title("COPOM Tone Index")
    st.caption("Leitura quantitativa e auditavel do tom hawkish/dovish em comunicados e atas do COPOM.")

    database = os.getenv("COPOM_TONE_DB", str(DEFAULT_DB))
    data = load_data(database)
    if not data:
        st.warning("Base DuckDB nao encontrada. Rode `copom-watch run-pipeline --use-llm never` antes de abrir o dashboard.")
        st.stop()

    scores = data["scores"]
    meetings = data["meetings"]
    topic_scores = data["topic_scores"]
    evidence = data["evidence"]
    focus = data["focus"]

    with st.sidebar:
        st.header("Controles")
        operational_only = st.toggle("Apenas janela operacional", value=True)
        if operational_only and "in_operational_window" in scores:
            scores_view = scores[scores["in_operational_window"]].copy()
        else:
            scores_view = scores.copy()
        meeting_labels = scores_view.sort_values("data_referencia", ascending=False).apply(
            lambda row: f"{int(row['nro_reuniao'])} - {pd.Timestamp(row['data_referencia']).date()}",
            axis=1,
        )
        selected_label = st.selectbox("Reuniao", meeting_labels.tolist())
        selected_meeting = scores_view.loc[meeting_labels[meeting_labels == selected_label].index[0], "meeting_id"]

    latest = scores_view[scores_view["meeting_id"] == selected_meeting].iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Indice", _fmt(latest["copom_tone_index"]), latest.get("classification", ""))
    c2.metric("Delta tone", _fmt(latest.get("delta_tone")))
    c3.metric("Selic pos", _fmt(latest.get("selic_pos")))
    c4.metric("Delta Selic", _fmt(latest.get("delta_selic")))

    st.subheader("Serie temporal")
    fig = px.line(
        scores_view,
        x="data_referencia",
        y="copom_tone_index",
        markers=True,
        hover_data=["nro_reuniao", "classification", "tone_comunicado", "tone_ata", "delta_selic"],
        labels={"data_referencia": "Data", "copom_tone_index": "COPOM Tone Index"},
    )
    fig.add_hline(y=50, line_dash="dash", line_color="gray")
    fig.add_hrect(y0=45, y1=55, line_width=0, fillcolor="gray", opacity=0.08)
    st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns([1.1, 1])
    with left:
        st.subheader("Decomposicao por topico")
        topic_view = topic_scores[topic_scores["meeting_id"].isin(scores_view["meeting_id"])]
        if topic_view.empty:
            st.info("Sem topicos classificados.")
        else:
            heatmap = topic_view.pivot_table(index="topic", columns="meeting_id", values="topic_tone", aggfunc="mean")
            fig = px.imshow(
                heatmap,
                aspect="auto",
                color_continuous_scale="RdBu_r",
                zmin=-1,
                zmax=1,
                labels={"color": "Tom"},
            )
            st.plotly_chart(fig, use_container_width=True)
    with right:
        st.subheader("Revisoes Focus")
        focus_view = focus[focus["meeting_id"] == selected_meeting].copy()
        if focus_view.empty:
            st.info("Focus indisponivel para esta reuniao.")
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
            st.dataframe(focus_view[display_cols], use_container_width=True, hide_index=True)

    st.subheader("Evidencias textuais")
    ev = evidence[evidence["meeting_id"] == selected_meeting].copy()
    if ev.empty:
        st.info("Nenhuma evidencia classificada para esta reuniao.")
    else:
        ev["score"] = ev["stance_score"].map(lambda value: f"{value:.2f}")
        st.dataframe(
            ev[["evidence_type", "document_type", "topic", "score", "confidence", "text", "rationale"]],
            use_container_width=True,
            hide_index=True,
        )

    with st.expander("Tabela de scores"):
        st.dataframe(scores_view, use_container_width=True, hide_index=True)


def _fmt(value: object) -> str:
    if pd.isna(value):
        return "n.d."
    if isinstance(value, str):
        return value
    return f"{float(value):.2f}"


if __name__ == "__main__":
    main()

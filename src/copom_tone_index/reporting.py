from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px


def generate_meeting_notes(
    meetings: pd.DataFrame,
    scores: pd.DataFrame,
    focus_revisions: pd.DataFrame,
    evidence: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    merged = meetings.merge(scores, on=["meeting_id", "nro_reuniao", "data_referencia"], how="left")
    for _, row in merged.sort_values("data_referencia", ascending=False).iterrows():
        focus = focus_revisions[focus_revisions["meeting_id"] == row["meeting_id"]]
        ev = evidence[evidence["meeting_id"] == row["meeting_id"]]
        path = output_dir / f"copom_{int(row['nro_reuniao'])}_tone_note.md"
        path.write_text(_meeting_note(row, focus, ev), encoding="utf-8")


def _meeting_note(row: pd.Series, focus: pd.DataFrame, evidence: pd.DataFrame) -> str:
    hawkish = evidence[evidence["evidence_type"] == "hawkish"].head(5)
    dovish = evidence[evidence["evidence_type"] == "dovish"].head(5)
    lines = [
        f"# COPOM Tone Note - Reuniao {int(row['nro_reuniao'])}",
        "",
        "## 1. Decisao",
        f"- Data de referencia: {_fmt_date(row.get('data_referencia'))}",
        f"- Selic: {_fmt_num(row.get('selic_pre'))} -> {_fmt_num(row.get('selic_pos'))}",
        f"- Delta: {_fmt_num(row.get('delta_selic'))} p.p.",
        "",
        "## 2. Tom",
        f"- Indice: {_fmt_num(row.get('copom_tone_index'))}",
        f"- Classificacao: {row.get('classification', 'unavailable')}",
        f"- Tone raw: {_fmt_num(row.get('tone_raw'))}",
        f"- Mudanca vs reuniao anterior: {_fmt_num(row.get('delta_tone'))}",
        "",
        "## 3. Evidencias hawkish",
    ]
    lines.extend(_evidence_lines(hawkish))
    lines.extend(["", "## 4. Evidencias dovish"])
    lines.extend(_evidence_lines(dovish))
    lines.extend(["", "## 5. Reacao Focus"])
    if focus.empty:
        lines.append("- Focus indisponivel para esta reuniao.")
    else:
        for _, item in focus.sort_values(["variable", "reference_year"]).iterrows():
            lines.append(
                "- "
                f"{item['variable']} {int(item['reference_year'])}: "
                f"pre={_fmt_num(item['focus_pre_value'])}, "
                f"pos-comunicado={_fmt_num(item['focus_post_comunicado_value'])}, "
                f"pos-ata={_fmt_num(item['focus_post_ata_value'])}"
            )
    lines.extend(
        [
            "",
            "## 6. Interpretacao",
            (
                "O score separa a comunicacao textual da decisao mecanica de Selic. "
                "A leitura deve ser interpretada como monitoramento quantitativo, nao como afirmacao causal."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def _evidence_lines(frame: pd.DataFrame) -> list[str]:
    if frame.empty:
        return ["- Nenhuma sentenca com score material nesta direcao."]
    lines = []
    for _, row in frame.iterrows():
        text = str(row["text"]).replace("\n", " ")
        lines.append(f"- `{row['topic']}` score={row['stance_score']:.2f}: {text}")
    return lines


def write_figures(scores: pd.DataFrame, topic_scores: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if not scores.empty:
        fig = px.line(
            scores.sort_values("data_referencia"),
            x="data_referencia",
            y="copom_tone_index",
            markers=True,
            title="COPOM Tone Index",
            labels={"data_referencia": "Data da reuniao", "copom_tone_index": "Indice"},
        )
        fig.add_hline(y=50, line_dash="dash", line_color="gray")
        fig.write_html(output_dir / "copom_tone_index.html", include_plotlyjs="cdn")
    if not topic_scores.empty:
        heatmap = topic_scores.pivot_table(index="topic", columns="meeting_id", values="topic_tone", aggfunc="mean")
        fig = px.imshow(
            heatmap,
            aspect="auto",
            color_continuous_scale="RdBu_r",
            title="Tom por topico e reuniao",
            labels={"color": "Tom"},
        )
        fig.write_html(output_dir / "topic_heatmap.html", include_plotlyjs="cdn")


def _fmt_num(value: object) -> str:
    if pd.isna(value):
        return "n.d."
    return f"{float(value):.2f}"


def _fmt_date(value: object) -> str:
    if pd.isna(value):
        return "n.d."
    return pd.Timestamp(value).strftime("%Y-%m-%d")

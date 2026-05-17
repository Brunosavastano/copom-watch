from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm

LOGGER = logging.getLogger(__name__)


def topic_weights(topics: dict[str, Any]) -> dict[str, float]:
    return {topic: float(details.get("weight", 1.0)) for topic, details in topics.items()}


def aggregate_scores(
    meetings: pd.DataFrame,
    documents: pd.DataFrame,
    sentences: pd.DataFrame,
    topics: dict[str, Any],
    communication_weight: float = 0.60,
    minutes_weight: float = 0.40,
    min_observations_for_surprise: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    weights = topic_weights(topics)
    sentences = sentences.copy()
    sentences["topic_weight"] = sentences["topic"].map(weights).fillna(1.0)
    sentences["weighted_score"] = sentences["stance_score"] * sentences["topic_weight"]

    document_scores = (
        sentences.groupby(["document_id", "meeting_id", "document_type"], as_index=False)
        .agg(weighted_sum=("weighted_score", "sum"), weight_sum=("topic_weight", "sum"), sentence_count=("sentence_id", "count"))
    )
    document_scores["document_tone"] = np.where(
        document_scores["weight_sum"] > 0,
        document_scores["weighted_sum"] / document_scores["weight_sum"],
        np.nan,
    )
    document_scores = document_scores.merge(
        documents[["document_id", "publication_date", "title", "model_version", "prompt_version"]],
        on="document_id",
        how="left",
    )

    topic_scores = (
        sentences.groupby(["meeting_id", "topic"], as_index=False)
        .agg(weighted_sum=("weighted_score", "sum"), weight_sum=("topic_weight", "sum"), sentence_count=("sentence_id", "count"))
    )
    topic_scores["topic_tone"] = np.where(topic_scores["weight_sum"] > 0, topic_scores["weighted_sum"] / topic_scores["weight_sum"], np.nan)

    score_rows: list[dict[str, Any]] = []
    for _, meeting in meetings.sort_values("data_referencia").iterrows():
        meeting_docs = document_scores[document_scores["meeting_id"] == meeting["meeting_id"]]
        comunicado = _doc_tone(meeting_docs, "comunicado")
        ata = _doc_tone(meeting_docs, "ata")
        if pd.notna(comunicado) and pd.notna(ata):
            tone_raw = communication_weight * comunicado + minutes_weight * ata
        elif pd.notna(comunicado):
            tone_raw = comunicado
        else:
            tone_raw = ata
        score_rows.append(
            {
                "meeting_id": meeting["meeting_id"],
                "nro_reuniao": meeting["nro_reuniao"],
                "data_referencia": meeting["data_referencia"],
                "tone_comunicado": comunicado,
                "tone_ata": ata,
                "tone_raw": tone_raw,
                "tone_nowcast": comunicado,
                "tone_final": tone_raw,
            }
        )
    scores = pd.DataFrame(score_rows)
    scores = _normalize_tone(scores)
    scores["delta_tone"] = scores["tone_raw"].diff()
    scores["classification"] = scores["copom_tone_index"].map(classify_index)
    scores["communication_surprise"] = estimate_communication_surprise(
        scores.merge(meetings[["meeting_id", "selic_pre", "delta_selic"]], on="meeting_id", how="left"),
        min_observations=min_observations_for_surprise,
    )
    return document_scores, topic_scores, scores


def _doc_tone(document_scores: pd.DataFrame, document_type: str) -> float:
    subset = document_scores[document_scores["document_type"] == document_type]
    if subset.empty:
        return np.nan
    return float(subset.iloc[0]["document_tone"])


def _normalize_tone(scores: pd.DataFrame) -> pd.DataFrame:
    scores = scores.copy()
    valid = scores["tone_raw"].dropna()
    mean = valid.mean() if not valid.empty else 0.0
    std = valid.std(ddof=0) if len(valid) > 1 else 0.0
    if std == 0 or pd.isna(std):
        LOGGER.warning("Tone historical standard deviation is zero or unavailable; ToneZ set to 0.")
        scores["tone_z"] = 0.0
    else:
        scores["tone_z"] = (scores["tone_raw"] - mean) / std
    scores["copom_tone_index"] = 50 + 10 * scores["tone_z"]
    return scores


def classify_index(value: float) -> str:
    if pd.isna(value):
        return "unavailable"
    if value > 60:
        return "claramente hawkish"
    if value >= 55:
        return "moderadamente hawkish"
    if value >= 45:
        return "neutro / balanceado"
    if value >= 40:
        return "moderadamente dovish"
    return "claramente dovish"


def estimate_communication_surprise(scores_with_controls: pd.DataFrame, min_observations: int = 8) -> pd.Series:
    data = scores_with_controls[["tone_raw", "delta_selic", "selic_pre"]].dropna()
    output = pd.Series(np.nan, index=scores_with_controls.index, dtype=float)
    if len(data) < min_observations:
        LOGGER.info("Not enough observations for communication surprise residuals.")
        return output
    x = sm.add_constant(data[["delta_selic", "selic_pre"]])
    y = data["tone_raw"]
    try:
        model = sm.OLS(y, x).fit()
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Communication surprise regression failed: %s", exc)
        return output
    residuals = y - model.predict(x)
    output.loc[data.index] = residuals
    return output


def evidence_sentences(sentences: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    hawkish = (
        sentences[sentences["stance_score"] > 0]
        .sort_values(["meeting_id", "stance_score", "confidence"], ascending=[True, False, False])
        .groupby("meeting_id")
        .head(top_n)
        .assign(evidence_type="hawkish")
    )
    dovish = (
        sentences[sentences["stance_score"] < 0]
        .sort_values(["meeting_id", "stance_score", "confidence"], ascending=[True, True, False])
        .groupby("meeting_id")
        .head(top_n)
        .assign(evidence_type="dovish")
    )
    return pd.concat([hawkish, dovish], ignore_index=True)


def topic_distribution_json(topic_scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for meeting_id, group in topic_scores.groupby("meeting_id"):
        payload = {
            row["topic"]: {"tone": row["topic_tone"], "sentences": int(row["sentence_count"])}
            for _, row in group.iterrows()
        }
        rows.append({"meeting_id": meeting_id, "topic_distribution": json.dumps(payload, ensure_ascii=False)})
    return pd.DataFrame(rows)

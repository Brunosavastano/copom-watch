from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field, ValidationError

from copom_tone_index.text import contains_any, strip_accents

LOGGER = logging.getLogger(__name__)


class SentenceClassification(BaseModel):
    sentence_id: str
    topic: str
    stance: str
    stance_score: float = Field(ge=-1.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str
    evidence_terms: list[str] = Field(default_factory=list)
    model_version: str
    prompt_version: str


def classify_sentences_baseline(
    sentences: pd.DataFrame,
    topics: dict[str, Any],
    lexicon: dict[str, Any],
    prompt_version: str,
    model_version: str = "lexicon-baseline-v1",
    neutral_band: float = 0.15,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, sentence in sentences.iterrows():
        topic = classify_topic(sentence["text"], topics)
        hawkish_score, hawkish_terms = _score_terms(sentence["text"], lexicon.get("hawkish", []))
        dovish_score, dovish_terms = _score_terms(sentence["text"], lexicon.get("dovish", []))
        total = hawkish_score + dovish_score
        score = 0.0 if total == 0 else (hawkish_score - dovish_score) / total
        score = max(-1.0, min(1.0, score))
        if abs(score) < neutral_band:
            stance = "neutral"
            score = 0.0
        elif score > 0:
            stance = "hawkish"
        else:
            stance = "dovish"
        matched_terms = hawkish_terms + dovish_terms
        confidence = 0.2 if not matched_terms else min(0.95, 0.35 + 0.12 * len(matched_terms) + 0.25 * abs(score))
        rationale = _rationale(stance, topic, hawkish_terms, dovish_terms)
        rows.append(
            SentenceClassification(
                sentence_id=sentence["sentence_id"],
                topic=topic,
                stance=stance,
                stance_score=round(float(score), 4),
                confidence=round(float(confidence), 4),
                rationale=rationale,
                evidence_terms=matched_terms[:8],
                model_version=model_version,
                prompt_version=prompt_version,
            ).model_dump()
        )
    classified = pd.DataFrame(rows)
    return sentences.merge(classified, on="sentence_id", how="left")


def classify_topic(text: str, topics: dict[str, Any]) -> str:
    counts = {
        topic: contains_any(text, details.get("keywords", []))
        for topic, details in topics.items()
        if topic != "institutional"
    }
    best_topic = max(counts, key=counts.get) if counts else "institutional"
    if counts.get(best_topic, 0) == 0:
        return "institutional"
    return best_topic


def _score_terms(text: str, terms: list[dict[str, Any]]) -> tuple[float, list[str]]:
    lowered = strip_accents(text.lower())
    score = 0.0
    matched: list[str] = []
    for item in terms:
        term = item["term"]
        if strip_accents(term.lower()) in lowered:
            score += float(item.get("weight", 1.0))
            matched.append(term)
    return score, matched


def _rationale(stance: str, topic: str, hawkish_terms: list[str], dovish_terms: list[str]) -> str:
    if stance == "hawkish":
        return f"Termos associados a viés restritivo no tópico {topic}: {', '.join(hawkish_terms[:4])}."
    if stance == "dovish":
        return f"Termos associados a viés benigno/flexibilização no tópico {topic}: {', '.join(dovish_terms[:4])}."
    return f"Sentença classificada como neutra ou pouco informativa para viés de política monetária no tópico {topic}."


def maybe_classify_with_llm(
    classified_baseline: pd.DataFrame,
    topics: dict[str, Any],
    prompt_version: str,
    cache_dir: Path,
    use_llm: str = "auto",
    model: str = "claude-3-5-sonnet-latest",
) -> pd.DataFrame:
    """Optional LLM layer. It falls back to baseline unless explicitly available and requested."""
    if use_llm == "never":
        return classified_baseline
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        if use_llm == "always":
            LOGGER.warning("ANTHROPIC_API_KEY is not set; keeping lexicon baseline.")
        return classified_baseline
    try:
        import anthropic  # type: ignore
    except ImportError:
        LOGGER.warning("anthropic package is not installed; keeping lexicon baseline.")
        return classified_baseline

    cache_dir.mkdir(parents=True, exist_ok=True)
    client = anthropic.Anthropic(api_key=api_key)
    updates: list[pd.DataFrame] = []
    topic_names = list(topics)
    for document_id, group in classified_baseline.groupby("document_id"):
        cache_key = _hash_payload(group[["sentence_id", "text"]].to_dict("records"), prompt_version, model)
        cache_path = cache_dir / f"llm_{document_id}_{cache_key}.json"
        if cache_path.exists():
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
        else:
            prompt = _llm_prompt(group, topic_names)
            try:
                response = client.messages.create(
                    model=model,
                    max_tokens=6000,
                    temperature=0,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = response.content[0].text
                payload = json.loads(text)
                cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception as exc:  # noqa: BLE001 - LLM enrichments must not break offline reproducibility.
                LOGGER.warning("LLM classification failed for %s: %s", document_id, exc)
                continue
        try:
            validated = [
                SentenceClassification(
                    **item,
                    model_version=model,
                    prompt_version=prompt_version,
                ).model_dump()
                for item in payload
            ]
        except (TypeError, ValidationError) as exc:
            LOGGER.warning("Invalid LLM payload for %s: %s", document_id, exc)
            continue
        llm_df = pd.DataFrame(validated)
        updates.append(llm_df)
    if not updates:
        return classified_baseline
    columns = [
        "sentence_id",
        "topic",
        "stance",
        "stance_score",
        "confidence",
        "rationale",
        "evidence_terms",
        "model_version",
        "prompt_version",
    ]
    llm_updates = pd.concat(updates, ignore_index=True)[columns]
    updated = classified_baseline.merge(llm_updates, on="sentence_id", how="left", suffixes=("", "_llm"))
    for column in columns:
        if column == "sentence_id":
            continue
        llm_column = f"{column}_llm"
        if llm_column not in updated:
            continue
        updated[column] = updated[llm_column].combine_first(updated[column])
        updated = updated.drop(columns=[llm_column])
    return updated


def _hash_payload(records: list[dict[str, Any]], prompt_version: str, model: str) -> str:
    payload = json.dumps({"records": records, "prompt": prompt_version, "model": model}, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _llm_prompt(group: pd.DataFrame, topic_names: list[str]) -> str:
    sentence_payload = group[["sentence_id", "text"]].to_dict("records")
    return (
        "Você é um economista especializado em política monetária brasileira.\n"
        "Classifique cada sentença do COPOM por tópico e tom hawkish/dovish. "
        "Não use sentimento genérico. Retorne apenas JSON válido.\n"
        f"Tópicos permitidos: {topic_names}.\n"
        "Campos obrigatórios: sentence_id, topic, stance, stance_score, confidence, rationale, evidence_terms.\n"
        "stance deve ser hawkish, dovish ou neutral. stance_score varia de -1 a 1.\n"
        f"Sentenças: {json.dumps(sentence_payload, ensure_ascii=False)}"
    )

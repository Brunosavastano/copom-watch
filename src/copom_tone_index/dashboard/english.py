"""English presentation and deterministic bilingual retrieval; no API calls."""
import re

import pandas as pd

from copom_tone_index.semantic import (
    is_out_of_scope_forecast_query,
    search_semantic_chunks,
)

# Query expansion bridges the English UI to the original Portuguese corpus.
# It changes retrieval terms only, never evidence text or analytical scores.
SEARCH_TERMS = {
    "unanchored expectations": "expectativas desancoradas",
    "hawkish": "restritiva restritivo aperto",
    "dovish": "expansionista afrouxamento",
    "inflation": "inflacao",
    "expectations": "expectativas",
    "fiscal risk": "risco fiscal",
    "economic activity": "atividade economica",
    "rate-cutting": "corte reducao juros",
    "rate cuts": "corte reducao juros",
    "rate hikes": "elevacao aumento juros",
    "interest rate": "taxa juros",
    "labor market": "mercado trabalho",
    "exchange rate": "cambio",
    "uncertainty": "incerteza",
    "growth": "crescimento",
    "slowdown": "desaceleracao",
    "inflation target": "meta inflacao",
    "services": "servicos",
    "food": "alimentos",
    "credit": "credito",
    "external": "externo",
    "global": "global internacional",
}


def retrieval_query(query: str) -> str:
    expanded = query.lower()
    pattern = "|".join(re.escape(key) for key in sorted(SEARCH_TERMS, key=len, reverse=True))
    return re.sub(r"\b(?:" + pattern + r")\b", lambda m: SEARCH_TERMS[m.group(0)], expanded)


def english_citation(value: object) -> str:
    text = str(value)
    for old, new in (("Reunião ", "Meeting "), ("Reuniao ", "Meeting "),
                     (", comunicado,", ", statement,"), (", ata,", ", minutes,"),
                     ("sentença ", "sentence "), ("sentenca ", "sentence ")):
        text = text.replace(old, new)
    return text


def search_english_chunks(query: str, chunks: pd.DataFrame, top_n: int = 10,
                          method: str | None = None) -> pd.DataFrame:
    results = search_semantic_chunks(retrieval_query(query), chunks, top_n=top_n, method=method)
    if not results.empty:
        results = results.copy()
        results["query"] = query
        results["citation"] = results["citation"].map(english_citation)
    return results


def ask_english_chunks(query: str, chunks: pd.DataFrame, top_n: int = 8,
                       method: str | None = "tfidf") -> tuple[str, pd.DataFrame]:
    asks_forecast = is_out_of_scope_forecast_query(query) or (
        "selic" in query.lower()
        and any(term in query.lower() for term in ("next", "will", "predict", "future"))
    )
    if asks_forecast:
        return ("COPOM Watch does not forecast the next Selic decision or recommend a policy action. "
                "It can retrieve historical cases and official evidence about communication, "
                "expectations and market reactions."), pd.DataFrame()
    matches = search_english_chunks(query, chunks, top_n=top_n, method=method)
    if matches.empty:
        return "", matches
    lines = ["Retrieved official evidence (original Portuguese):"]
    for _, row in matches.head(4).iterrows():
        lines.append(f"- {row['citation']}: {str(row['text']).strip()}")
    lines.append("No inference is made beyond these citations.")
    return "\n".join(lines), matches

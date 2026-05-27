from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from copom_tone_index.config import get_paths
from copom_tone_index.storage import export_tables, write_tables
from copom_tone_index.text import strip_accents
from copom_tone_index.v2 import read_optional_table, stable_text_hash, utc_now_naive

FORECAST_SCOPE_MESSAGE = (
    "O COPOM Watch não prevê a próxima Selic nem produz recomendação de decisão. "
    "Ele pode buscar casos históricos semelhantes e mostrar evidências textuais oficiais "
    "sobre comunicação, expectativas e reação de mercado."
)


@dataclass(frozen=True)
class SemanticCommandResult:
    database: Path
    output_path: Path | None
    rows: int
    status: str


@dataclass(frozen=True)
class SemanticSearchResult:
    database: Path
    output_path: Path | None
    rows: int
    status: str
    query: str
    top_matches: list[dict[str, object]]


@dataclass(frozen=True)
class SemanticAskResult:
    database: Path
    output_path: Path | None
    report_path: Path | None
    rows: int
    status: str
    query: str
    answer: str
    citations: list[dict[str, object]]


def build_semantic_index_command(method: str = "tfidf") -> SemanticCommandResult:
    paths = get_paths()
    sentences = read_optional_table(paths.database, "v2_sentence_scores", pd.DataFrame())
    chunks = build_semantic_chunks(sentences, method=method)
    write_tables(paths.database, {"semantic_chunks": chunks})
    export_tables(paths.database, paths.processed, ["semantic_chunks"])
    status = "completed" if not chunks.empty else "no_v2_sentences"
    return SemanticCommandResult(paths.database, paths.processed / "semantic_chunks.csv", len(chunks), status)


def semantic_search_command(query: str, top_n: int = 10, method: str | None = None) -> SemanticSearchResult:
    paths = get_paths()
    chunks = read_optional_table(paths.database, "semantic_chunks", pd.DataFrame())
    if chunks.empty:
        build_semantic_index_command(method=method or "tfidf")
        chunks = read_optional_table(paths.database, "semantic_chunks", pd.DataFrame())
    matches = search_semantic_chunks(query, chunks, top_n=top_n, method=method)
    output_path = paths.processed / "semantic_search_results.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matches.to_csv(output_path, index=False)
    if chunks.empty:
        status = "no_semantic_index"
    elif matches.empty:
        status = "no_matches"
    else:
        status = "completed"
    return SemanticSearchResult(
        database=paths.database,
        output_path=output_path,
        rows=len(matches),
        status=status,
        query=query,
        top_matches=matches.head(top_n).to_dict("records"),
    )


def semantic_ask_command(query: str, top_n: int = 8, method: str = "tfidf") -> SemanticAskResult:
    paths = get_paths()
    chunks = read_optional_table(paths.database, "semantic_chunks", pd.DataFrame())
    if chunks.empty:
        build_semantic_index_command(method=method)
        chunks = read_optional_table(paths.database, "semantic_chunks", pd.DataFrame())
    answer, matches = ask_semantic_chunks(query, chunks, top_n=top_n, method=method)
    output_path = paths.processed / "semantic_ask_results.csv"
    report_path = paths.reports.parent / "v2" / "semantic_ask_report.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    matches.to_csv(output_path, index=False)
    report_path.write_text(render_semantic_ask_report_html(query, answer, matches), encoding="utf-8")
    status = "completed" if answer and not matches.empty else "no_matches" if not chunks.empty else "no_semantic_index"
    return SemanticAskResult(
        database=paths.database,
        output_path=output_path,
        report_path=report_path,
        rows=len(matches),
        status=status,
        query=query,
        answer=answer,
        citations=matches.head(top_n).to_dict("records"),
    )


def build_semantic_chunks(sentences: pd.DataFrame, method: str = "tfidf") -> pd.DataFrame:
    columns = [
        "chunk_id",
        "sentence_id",
        "meeting_id",
        "nro_reuniao",
        "document_type",
        "text",
        "citation",
        "token_signature",
        "retrieval_method",
        "source_hash",
        "created_at",
    ]
    if sentences.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for _, row in sentences.iterrows():
        text = str(row.get("text", ""))
        token_signature = semantic_token_signature(text)
        sentence_id = str(row.get("sentence_id", ""))
        rows.append(
            {
                "chunk_id": stable_text_hash(f"{sentence_id}:{text}")[:24],
                "sentence_id": sentence_id,
                "meeting_id": row.get("meeting_id", ""),
                "nro_reuniao": row.get("nro_reuniao", ""),
                "document_type": row.get("document_type", ""),
                "text": text,
                "citation": f"Reunião {row.get('nro_reuniao', '')}, {row.get('document_type', '')}, sentença {sentence_id}",
                "token_signature": token_signature,
                "retrieval_method": method,
                "source_hash": row.get("source_hash", ""),
                "created_at": utc_now_naive(),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def semantic_token_signature(text: str, max_terms: int = 32) -> str:
    lowered = strip_accents(text.lower())
    tokens = re.findall(r"[a-z0-9]{4,}", lowered)
    counts: dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:max_terms]
    return "|".join(token for token, _ in ranked)


def search_semantic_chunks(query: str, chunks: pd.DataFrame, top_n: int = 10, method: str | None = None) -> pd.DataFrame:
    columns = [
        "rank",
        "score",
        "query",
        "citation",
        "meeting_id",
        "nro_reuniao",
        "document_type",
        "sentence_id",
        "text",
        "token_signature",
        "retrieval_method",
        "source_hash",
    ]
    if chunks.empty or not query.strip():
        return pd.DataFrame(columns=columns)
    selected_method = method or inferred_retrieval_method(chunks)
    if selected_method == "tfidf":
        return search_semantic_chunks_tfidf(query, chunks, top_n=top_n, columns=columns)
    query_signature = semantic_token_signature(query, max_terms=48)
    query_terms = token_set(query_signature)
    normalized_query = strip_accents(query.lower()).strip()
    if not query_terms:
        return pd.DataFrame(columns=columns)

    rows = []
    for _, chunk in chunks.iterrows():
        chunk_signature = str(chunk.get("token_signature", ""))
        chunk_terms = token_set(chunk_signature)
        text = str(chunk.get("text", ""))
        score = semantic_match_score(query_terms, normalized_query, chunk_terms, text)
        if score <= 0:
            continue
        rows.append(
            {
                "score": round(score, 6),
                "query": query,
                "citation": chunk.get("citation", ""),
                "meeting_id": chunk.get("meeting_id", ""),
                "nro_reuniao": chunk.get("nro_reuniao", ""),
                "document_type": chunk.get("document_type", ""),
                "sentence_id": chunk.get("sentence_id", ""),
                "text": text,
                "token_signature": chunk_signature,
                "retrieval_method": chunk.get("retrieval_method", "token"),
                "source_hash": chunk.get("source_hash", ""),
            }
        )
    if not rows:
        return pd.DataFrame(columns=columns)
    results = pd.DataFrame(rows).sort_values(
        ["score", "nro_reuniao", "document_type", "sentence_id"],
        ascending=[False, False, True, True],
        kind="mergesort",
    )
    results = results.head(max(1, int(top_n))).reset_index(drop=True)
    results.insert(0, "rank", range(1, len(results) + 1))
    return results[columns]


def search_semantic_chunks_tfidf(query: str, chunks: pd.DataFrame, top_n: int, columns: list[str]) -> pd.DataFrame:
    frame = chunks.copy()
    frame["text"] = frame.get("text", pd.Series(dtype=str)).fillna("").astype(str)
    valid = frame["text"].str.strip() != ""
    frame = frame[valid].reset_index(drop=True)
    if frame.empty:
        return pd.DataFrame(columns=columns)
    corpus = frame["text"].map(lambda text: strip_accents(text.lower())).tolist()
    normalized_query = strip_accents(query.lower()).strip()
    try:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        matrix = vectorizer.fit_transform(corpus)
        query_vector = vectorizer.transform([normalized_query])
        scores = cosine_similarity(query_vector, matrix).ravel()
    except ValueError:
        return pd.DataFrame(columns=columns)
    rows = []
    for idx, score in enumerate(scores):
        if score <= 0:
            continue
        chunk = frame.iloc[idx]
        rows.append(
            {
                "score": round(float(score), 6),
                "query": query,
                "citation": chunk.get("citation", ""),
                "meeting_id": chunk.get("meeting_id", ""),
                "nro_reuniao": chunk.get("nro_reuniao", ""),
                "document_type": chunk.get("document_type", ""),
                "sentence_id": chunk.get("sentence_id", ""),
                "text": chunk.get("text", ""),
                "token_signature": chunk.get("token_signature", ""),
                "retrieval_method": "tfidf",
                "source_hash": chunk.get("source_hash", ""),
            }
        )
    if not rows:
        return pd.DataFrame(columns=columns)
    results = pd.DataFrame(rows).sort_values(
        ["score", "nro_reuniao", "document_type", "sentence_id"],
        ascending=[False, False, True, True],
        kind="mergesort",
    )
    results = results.head(max(1, int(top_n))).reset_index(drop=True)
    results.insert(0, "rank", range(1, len(results) + 1))
    return results[columns]


def ask_semantic_chunks(query: str, chunks: pd.DataFrame, top_n: int = 8, method: str | None = "tfidf") -> tuple[str, pd.DataFrame]:
    if is_out_of_scope_forecast_query(query):
        empty = pd.DataFrame(
            columns=[
                "rank",
                "score",
                "query",
                "citation",
                "meeting_id",
                "nro_reuniao",
                "document_type",
                "sentence_id",
                "text",
                "token_signature",
                "retrieval_method",
                "source_hash",
            ]
        )
        return FORECAST_SCOPE_MESSAGE, empty
    matches = search_semantic_chunks(query, chunks, top_n=top_n, method=method)
    if matches.empty:
        return "", matches
    answer_lines = ["Síntese baseada exclusivamente nas citações recuperadas:"]
    for _, row in matches.head(4).iterrows():
        answer_lines.append(f"- {row.get('citation', '')}: {str(row.get('text', '')).strip()}")
    answer_lines.append("Não há inferência fora dessas citações.")
    return "\n".join(answer_lines), matches


def is_out_of_scope_forecast_query(query: str) -> bool:
    normalized = strip_accents(query.lower())
    if "selic" not in normalized:
        return False
    forecast_terms = [
        "prever",
        "previsao",
        "preve",
        "forecast",
        "proxima",
        "proximo",
        "qual sera",
        "quanto sera",
        "vai cortar",
        "vai subir",
        "vai cair",
        "decisao futura",
    ]
    return any(term in normalized for term in forecast_terms)


def render_semantic_ask_report_html(query: str, answer: str, matches: pd.DataFrame) -> str:
    citations = matches.to_html(index=False, escape=True) if not matches.empty else "<p>Nenhuma citação recuperada.</p>"
    safe_answer = "<br>".join(escape_html(line) for line in answer.splitlines()) if answer else "Sem resposta: nenhuma citação recuperada."
    return "\n".join(
        [
            "<!doctype html><html><head><meta charset='utf-8'>",
            "<title>Pergunte ao COPOM Watch</title>",
            "<style>body{font-family:Arial,sans-serif;margin:32px;line-height:1.45;color:#111827}table{border-collapse:collapse;width:100%;margin:12px 0}td,th{border:1px solid #ddd;padding:6px}th{background:#f3f4f6}.answer{background:#f9fafb;border:1px solid #e5e7eb;padding:12px;border-radius:6px}</style>",
            "</head><body>",
            "<h1>Pergunte ao COPOM Watch</h1>",
            f"<p><strong>Consulta:</strong> {escape_html(query)}</p>",
            f"<div class='answer'>{safe_answer}</div>",
            "<h2>Citações</h2>",
            citations,
            "</body></html>",
        ]
    )


def inferred_retrieval_method(chunks: pd.DataFrame) -> str:
    if "retrieval_method" not in chunks or chunks["retrieval_method"].dropna().empty:
        return "token"
    methods = set(chunks["retrieval_method"].dropna().astype(str).str.lower())
    return "tfidf" if "tfidf" in methods else "token"


def semantic_match_score(query_terms: set[str], normalized_query: str, chunk_terms: set[str], text: str) -> float:
    if not query_terms or not chunk_terms:
        return 0.0
    overlap = query_terms & chunk_terms
    if not overlap:
        normalized_text = strip_accents(text.lower())
        return 0.2 if normalized_query and normalized_query in normalized_text else 0.0
    recall = len(overlap) / len(query_terms)
    precision = len(overlap) / len(chunk_terms)
    jaccard = len(overlap) / len(query_terms | chunk_terms)
    normalized_text = strip_accents(text.lower())
    phrase_bonus = 0.25 if normalized_query and normalized_query in normalized_text else 0.0
    return (0.65 * recall) + (0.2 * precision) + (0.15 * jaccard) + phrase_bonus


def token_set(signature: str) -> set[str]:
    return {token for token in str(signature).split("|") if token}


def escape_html(value: object) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#x27;")
    )

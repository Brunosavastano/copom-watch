from __future__ import annotations

import html
import re
import unicodedata
from typing import Iterable

import pandas as pd
from bs4 import BeautifulSoup


INSTITUTIONAL_PATTERNS = [
    r"^votaram por essa decisão",
    r"^votaram por essa decisao",
    r"^votaram por esta decisão",
    r"^votaram por esta decisao",
    r"^membros do copom",
    r"^chefes de departamento",
    r"^demais participantes",
    r"^horário de início",
    r"^local:",
    r"^data:",
    r"^notas de rodapé",
    r"^tabela \d*",
    r"^projeções de inflação no cenário",
    r"^variação do ipca acumulada",
]


def normalize_whitespace(text: str) -> str:
    text = html.unescape(text or "")
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def strip_accents(text: str) -> str:
    return "".join(char for char in unicodedata.normalize("NFKD", text) if not unicodedata.combining(char))


def clean_copom_html(raw_html: str) -> str:
    soup = BeautifulSoup(raw_html or "", "html.parser")
    for selector in ["script", "style", "table", "sup", "#ata_info", ".notasrodape", ".olho-background"]:
        for tag in soup.select(selector):
            tag.decompose()

    text_blocks: list[str] = []
    for tag in soup.find_all(["h3", "h4", "p", "li"]):
        text = normalize_whitespace(tag.get_text(" ", strip=True))
        if not text:
            continue
        lowered = strip_accents(text.lower())
        if any(re.search(pattern, lowered) for pattern in INSTITUTIONAL_PATTERNS):
            continue
        text = re.sub(r"^\d+\.\s*", "", text)
        text_blocks.append(text)
    if not text_blocks:
        text_blocks = [normalize_whitespace(soup.get_text(" ", strip=True))]
    return "\n\n".join(dict.fromkeys(text_blocks))


def split_sentences(text: str) -> list[str]:
    text = normalize_whitespace(text)
    if not text:
        return []
    placeholders = {
        "p.p.": "p<<DOT>>p.",
        "a.a.": "a<<DOT>>a.",
        "EUA.": "EUA<<DOT>>",
        "US$.": "US$<<DOT>>",
    }
    for original, replacement in placeholders.items():
        text = text.replace(original, replacement)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÂÊÔÃÕÇ])", text)
    sentences = []
    for part in parts:
        for original, replacement in placeholders.items():
            part = part.replace(replacement, original)
        cleaned = normalize_whitespace(part)
        if len(cleaned) >= 20:
            sentences.append(cleaned)
    return sentences


def build_sentence_frame(documents: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, doc in documents.iterrows():
        clean_text = clean_copom_html(doc.get("raw_text", ""))
        sentences = split_sentences(clean_text)
        for order, sentence in enumerate(sentences, start=1):
            rows.append(
                {
                    "sentence_id": f"{doc['document_id']}_{order:03d}",
                    "document_id": doc["document_id"],
                    "meeting_id": doc["meeting_id"],
                    "nro_reuniao": doc["nro_reuniao"],
                    "document_type": doc["document_type"],
                    "sentence_order": order,
                    "text": sentence,
                }
            )
    return pd.DataFrame(rows)


def attach_clean_text(documents: pd.DataFrame) -> pd.DataFrame:
    documents = documents.copy()
    documents["clean_text"] = documents["raw_text"].fillna("").map(clean_copom_html)
    return documents


def contains_any(text: str, keywords: Iterable[str]) -> int:
    lowered = strip_accents(text.lower())
    return sum(1 for keyword in keywords if strip_accents(keyword.lower()) in lowered)

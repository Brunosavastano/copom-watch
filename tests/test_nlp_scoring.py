import pandas as pd

from copom_tone_index.nlp import classify_sentences_baseline, classify_topic
from copom_tone_index.scoring import aggregate_scores, classify_index


TOPICS = {
    "inflation_expectations": {"weight": 1.3, "keywords": ["expectativas", "desancoradas"]},
    "activity_growth": {"weight": 0.95, "keywords": ["atividade", "desaceleração"]},
    "institutional": {"weight": 0.2, "keywords": ["votaram"]},
}

LEXICON = {
    "hawkish": [{"term": "expectativas desancoradas", "weight": 2.0}],
    "dovish": [{"term": "desaceleração", "weight": 1.0}],
}


def test_classify_topic_uses_taxonomy_keywords() -> None:
    assert classify_topic("As expectativas seguem desancoradas.", TOPICS) == "inflation_expectations"
    assert classify_topic("Texto operacional sem conteúdo macro.", TOPICS) == "institutional"


def test_classify_sentences_baseline_bounds() -> None:
    sentences = pd.DataFrame(
        [
            {
                "sentence_id": "s1",
                "document_id": "d1",
                "meeting_id": "m1",
                "nro_reuniao": 1,
                "document_type": "comunicado",
                "sentence_order": 1,
                "text": "As expectativas desancoradas exigem cautela.",
            },
            {
                "sentence_id": "s2",
                "document_id": "d1",
                "meeting_id": "m1",
                "nro_reuniao": 1,
                "document_type": "comunicado",
                "sentence_order": 2,
                "text": "A atividade mostra desaceleração.",
            },
        ]
    )
    result = classify_sentences_baseline(sentences, TOPICS, LEXICON, "p1")
    assert result["stance_score"].between(-1, 1).all()
    assert result["confidence"].between(0, 1).all()
    assert result.loc[result["sentence_id"] == "s1", "stance"].iloc[0] == "hawkish"
    assert result.loc[result["sentence_id"] == "s2", "stance"].iloc[0] == "dovish"


def test_aggregate_scores_generates_index() -> None:
    meetings = pd.DataFrame(
        [
            {
                "meeting_id": "m1",
                "nro_reuniao": 1,
                "data_referencia": pd.Timestamp("2024-01-01"),
                "selic_pre": 10.0,
                "selic_pos": 10.5,
                "delta_selic": 0.5,
            },
            {
                "meeting_id": "m2",
                "nro_reuniao": 2,
                "data_referencia": pd.Timestamp("2024-02-01"),
                "selic_pre": 10.5,
                "selic_pos": 10.25,
                "delta_selic": -0.25,
            },
        ]
    )
    documents = pd.DataFrame(
        [
            {
                "document_id": "m1_comunicado",
                "meeting_id": "m1",
                "document_type": "comunicado",
                "publication_date": pd.Timestamp("2024-01-01"),
                "title": "x",
                "model_version": "baseline",
                "prompt_version": "p1",
            },
            {
                "document_id": "m2_comunicado",
                "meeting_id": "m2",
                "document_type": "comunicado",
                "publication_date": pd.Timestamp("2024-02-01"),
                "title": "x",
                "model_version": "baseline",
                "prompt_version": "p1",
            },
        ]
    )
    sentences = pd.DataFrame(
        [
            {
                "sentence_id": "s1",
                "document_id": "m1_comunicado",
                "meeting_id": "m1",
                "document_type": "comunicado",
                "topic": "inflation_expectations",
                "stance_score": 1.0,
                "confidence": 0.9,
            },
            {
                "sentence_id": "s2",
                "document_id": "m2_comunicado",
                "meeting_id": "m2",
                "document_type": "comunicado",
                "topic": "activity_growth",
                "stance_score": -1.0,
                "confidence": 0.9,
            },
        ]
    )
    _, _, scores = aggregate_scores(meetings, documents, sentences, TOPICS)
    assert scores["copom_tone_index"].notna().all()
    assert scores.loc[scores["meeting_id"] == "m1", "copom_tone_index"].iloc[0] > 50
    assert scores.loc[scores["meeting_id"] == "m2", "copom_tone_index"].iloc[0] < 50


def test_classify_index_thresholds() -> None:
    assert classify_index(61) == "claramente hawkish"
    assert classify_index(50) == "neutro / balanceado"
    assert classify_index(39) == "claramente dovish"

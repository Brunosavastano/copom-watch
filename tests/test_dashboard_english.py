import pandas as pd
from pandas.testing import assert_frame_equal

from copom_tone_index.dashboard.app import _display_frame
from copom_tone_index.dashboard.english import ask_english_chunks
from copom_tone_index.semantic import build_semantic_chunks


def test_english_display_keeps_quoted_evidence_and_raw_data_intact():
    original = pd.DataFrame([{
        "document_type": "ata", "classification_v2": "neutro / balanceado",
        "tone_level": 0.6, "text": "As expectativas seguem desancoradas_hoje.",
        "citation": "Reunião 270, ata, sentença s1",
    }])
    before = original.copy(deep=True)
    displayed = _display_frame(original)
    assert displayed.iloc[0]["Text"] == original.iloc[0]["text"]
    assert displayed.iloc[0]["Document"] == "minutes"
    assert displayed.iloc[0]["Classification"] == "neutral / balanced"
    assert displayed.iloc[0]["Citation"] == "Meeting 270, minutes, sentence s1"
    assert displayed.iloc[0]["Tone"] == 0.6
    assert_frame_equal(original, before)


def test_english_question_retrieves_portuguese_evidence_without_api():
    chunks = build_semantic_chunks(pd.DataFrame([{
        "sentence_id": "s1", "meeting_id": "copom_270", "nro_reuniao": 270,
        "document_type": "ata", "text": "As expectativas seguem desancoradas.",
        "source_hash": "source",
    }]))
    answer, citations = ask_english_chunks("When did Copom discuss unanchored expectations?", chunks)
    assert len(citations) == 1
    assert "Meeting 270" in answer
    assert "As expectativas seguem desancoradas." in answer
    assert "original Portuguese" in answer
    refused, empty = ask_english_chunks("What will the next Selic rate be?", chunks)
    assert "does not forecast" in refused
    assert empty.empty

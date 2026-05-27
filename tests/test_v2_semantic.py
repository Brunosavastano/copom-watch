import pandas as pd

from copom_tone_index.config import ProjectPaths
from copom_tone_index.semantic import ask_semantic_chunks, build_semantic_chunks, semantic_ask_command, search_semantic_chunks, semantic_search_command
from copom_tone_index.storage import write_tables


def _semantic_sentence_scores() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sentence_id": "s1",
                "meeting_id": "copom_250",
                "nro_reuniao": 250,
                "document_type": "comunicado",
                "text": "As expectativas de inflacao seguem desancoradas e exigem cautela adicional.",
                "source_hash": "hash1",
            },
            {
                "sentence_id": "s2",
                "meeting_id": "copom_251",
                "nro_reuniao": 251,
                "document_type": "ata",
                "text": "A atividade economica mostra desaceleracao e maior ociosidade.",
                "source_hash": "hash2",
            },
        ]
    )


def test_semantic_search_returns_cited_matches() -> None:
    chunks = build_semantic_chunks(_semantic_sentence_scores(), method="tfidf")

    results = search_semantic_chunks("expectativas desancoradas", chunks, top_n=5)

    assert len(results) == 1
    assert results["sentence_id"].iloc[0] == "s1"
    assert "Reunião 250" in results["citation"].iloc[0]
    assert "expectativas" in results["text"].iloc[0]
    assert results["retrieval_method"].iloc[0] == "tfidf"


def test_semantic_search_command_writes_csv(tmp_path, monkeypatch) -> None:
    database = tmp_path / "copom_tone.duckdb"
    chunks = build_semantic_chunks(_semantic_sentence_scores())
    write_tables(database, {"semantic_chunks": chunks})
    paths = ProjectPaths(
        database=database,
        raw=tmp_path / "raw",
        processed=tmp_path / "processed",
        figures=tmp_path / "figures",
        reports=tmp_path / "reports" / "meeting_notes",
    )
    monkeypatch.setattr("copom_tone_index.semantic.get_paths", lambda: paths)

    result = semantic_search_command("atividade ociosidade", top_n=3)
    output = pd.read_csv(result.output_path)

    assert result.status == "completed"
    assert result.rows == 1
    assert result.output_path.exists()
    assert output["sentence_id"].iloc[0] == "s2"


def test_semantic_ask_requires_citations() -> None:
    chunks = build_semantic_chunks(_semantic_sentence_scores(), method="tfidf")

    answer, citations = ask_semantic_chunks("expectativas desancoradas", chunks, top_n=3)

    assert answer
    assert "Reunião 250" in answer
    assert not citations.empty
    assert citations["citation"].notna().all()


def test_semantic_ask_no_match_has_no_answer() -> None:
    chunks = build_semantic_chunks(_semantic_sentence_scores(), method="tfidf")

    answer, citations = ask_semantic_chunks("termo inexistente zzz", chunks, top_n=3)

    assert answer == ""
    assert citations.empty


def test_semantic_ask_refuses_selic_forecast() -> None:
    chunks = build_semantic_chunks(_semantic_sentence_scores(), method="tfidf")

    answer, citations = ask_semantic_chunks("O app consegue prever a próxima Selic?", chunks, top_n=3)

    assert "não prevê a próxima Selic" in answer
    assert "casos históricos semelhantes" in answer
    assert citations.empty


def test_semantic_ask_command_writes_outputs(tmp_path, monkeypatch) -> None:
    database = tmp_path / "copom_tone.duckdb"
    chunks = build_semantic_chunks(_semantic_sentence_scores(), method="tfidf")
    write_tables(database, {"semantic_chunks": chunks})
    paths = ProjectPaths(
        database=database,
        raw=tmp_path / "raw",
        processed=tmp_path / "processed",
        figures=tmp_path / "figures",
        reports=tmp_path / "reports" / "meeting_notes",
    )
    monkeypatch.setattr("copom_tone_index.semantic.get_paths", lambda: paths)

    result = semantic_ask_command("atividade ociosidade", top_n=3)

    assert result.status == "completed"
    assert result.answer
    assert result.output_path.exists()
    assert result.report_path.exists()

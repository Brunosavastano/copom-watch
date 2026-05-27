import pandas as pd

from copom_tone_index.dashboard.app import APP_TABS, filter_v2_evidence, filter_v2_redline, help_label, load_data, semantic_search_for_dashboard
from copom_tone_index.dashboard.ui_copy import GLOSSARY, PAGE_COPY
from copom_tone_index.semantic import build_semantic_chunks
from copom_tone_index.storage import write_tables


def test_filter_v2_evidence_by_meeting_type_document_and_topic() -> None:
    evidence = pd.DataFrame(
        [
            {
                "meeting_id": "copom_1",
                "evidence_type": "hawkish",
                "document_type": "comunicado",
                "primary_topic": "inflation_expectations",
                "tone_level": 1.0,
            },
            {
                "meeting_id": "copom_1",
                "evidence_type": "dovish",
                "document_type": "ata",
                "primary_topic": "activity_growth",
                "tone_level": -0.7,
            },
            {
                "meeting_id": "copom_2",
                "evidence_type": "hawkish",
                "document_type": "comunicado",
                "primary_topic": "fiscal_risk",
                "tone_level": 0.9,
            },
        ]
    )

    filtered = filter_v2_evidence(
        evidence,
        "copom_1",
        evidence_types=["hawkish"],
        document_types=["comunicado"],
        topics=["inflation_expectations"],
    )

    assert len(filtered) == 1
    assert filtered["primary_topic"].iloc[0] == "inflation_expectations"


def test_filter_v2_redline_by_meeting_change_and_document() -> None:
    redline = pd.DataFrame(
        [
            {
                "meeting_id": "copom_1",
                "document_type": "comunicado",
                "change_type": "added",
                "tone_delta": 0.4,
            },
            {
                "meeting_id": "copom_1",
                "document_type": "ata",
                "change_type": "removed",
                "tone_delta": -0.2,
            },
            {
                "meeting_id": "copom_2",
                "document_type": "comunicado",
                "change_type": "added",
                "tone_delta": 0.1,
            },
        ]
    )

    filtered = filter_v2_redline(redline, "copom_1", change_types=["added"], document_types=["comunicado"])

    assert len(filtered) == 1
    assert filtered["change_type"].iloc[0] == "added"
    assert filtered["document_type"].iloc[0] == "comunicado"


def test_semantic_search_for_dashboard_returns_citations() -> None:
    sentence_scores = pd.DataFrame(
        [
            {
                "sentence_id": "s1",
                "meeting_id": "copom_270",
                "nro_reuniao": 270,
                "document_type": "ata",
                "text": "As expectativas de inflacao seguem desancoradas.",
                "source_hash": "hash1",
            },
            {
                "sentence_id": "s2",
                "meeting_id": "copom_271",
                "nro_reuniao": 271,
                "document_type": "comunicado",
                "text": "A atividade economica mostra desaceleracao.",
                "source_hash": "hash2",
            },
        ]
    )
    chunks = build_semantic_chunks(sentence_scores)

    results = semantic_search_for_dashboard("expectativas desancoradas", chunks, top_n=5)

    assert len(results) == 1
    assert results["sentence_id"].iloc[0] == "s1"
    assert "Reunião 270" in results["citation"].iloc[0]


def test_dashboard_load_data_reads_public_package(tmp_path) -> None:
    database = tmp_path / "copom_watch_public.duckdb"
    write_tables(
        database,
        {
            "v2_meeting_scores": pd.DataFrame(
                [
                    {
                        "meeting_id": "copom_270",
                        "nro_reuniao": 270,
                        "data_referencia": "2024-01-31",
                        "copom_tone_index_v2": 55.0,
                    }
                ]
            ),
            "semantic_chunks": build_semantic_chunks(
                pd.DataFrame(
                    [
                        {
                            "sentence_id": "s1",
                            "meeting_id": "copom_270",
                            "nro_reuniao": 270,
                            "document_type": "ata",
                            "text": "Expectativas desancoradas.",
                            "source_hash": "hash1",
                        }
                    ]
                )
            ),
        },
    )

    load_data.clear()
    data = load_data(str(database))

    assert not data["v2_scores"].empty
    assert not data["semantic_chunks"].empty
    assert data["scores"].empty


def test_dashboard_pages_have_guided_explanations() -> None:
    tab_keys = {key for _, key in APP_TABS}

    assert tab_keys.issubset(PAGE_COPY)
    for _, key in APP_TABS:
        page = PAGE_COPY[key]
        assert page.question
        assert page.how_to_read
        assert page.limitations
        assert page.glossary_terms


def test_dashboard_core_terms_have_tooltips() -> None:
    required_terms = [
        "Tom bruto",
        "Índice de tom",
        "Surpresa textual",
        "Intensidade",
        "Calibração",
        "Mudança textual",
        "Subíndices",
        "Focus",
        "Reação de mercado",
        "Frases-chave",
        "Auditoria",
        "Busca com evidências",
        "Período analisado",
    ]

    for term in required_terms:
        assert term in GLOSSARY
        assert len(GLOSSARY[term]) > 40
        assert "copom-help-icon" in help_label(term)


def test_dashboard_public_labels_are_product_oriented() -> None:
    labels = [label for label, _ in APP_TABS]

    assert "Evolução do tom" in labels
    assert "Perguntas com evidências" in labels
    assert "Linha do tempo" not in labels
    assert "Busca com evidências" not in labels
    assert all("V2" not in label for label in labels)

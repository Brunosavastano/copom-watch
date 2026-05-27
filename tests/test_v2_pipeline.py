from pathlib import Path

import pandas as pd

from copom_tone_index.config import ProjectPaths
from copom_tone_index.market import (
    build_market_event_windows,
    build_public_market_coverage,
    derive_public_decision_expectations,
    normalize_decision_expectations,
    normalize_market_observations,
    parse_anbima_intraday_html,
)
from copom_tone_index.storage import write_tables
from copom_tone_index.v2 import (
    aggregate_v2_scores,
    build_v2_calibration,
    build_v2_evidence,
    build_v2_redline,
    build_v2_sentences,
    build_v2_subindices,
    best_previous_match,
    coalesce_llm_with_baseline_for_test,
    ensure_v2_document_source_urls,
    existing_v2_backfill_is_sufficient,
    prepare_v2_backfill,
    redline_is_current,
    score_v2_sentences,
    stable_text_hash,
    validate_v2_schema,
)
from copom_tone_index.v21 import build_v21_event_panel, v21_health_command


TOPICS = {
    "inflation_current": {"weight": 1.1, "keywords": ["inflacao", "nucleos", "ipca"]},
    "inflation_expectations": {"weight": 1.3, "keywords": ["expectativas", "desancoradas"]},
    "activity_growth": {"weight": 0.9, "keywords": ["atividade", "demanda"]},
    "labor_market": {"weight": 0.9, "keywords": ["emprego", "massa salarial", "rendimento medio"]},
    "fiscal_risk": {"weight": 1.2, "keywords": ["fiscal"]},
    "risk_balance": {"weight": 1.2, "keywords": ["riscos", "cenario inflacionario"]},
    "uncertainty": {"weight": 1.0, "keywords": ["incerteza", "volatilidade"]},
    "external_environment": {"weight": 1.0, "keywords": ["estados unidos", "eua", "china", "europa"]},
    "fx_commodities": {"weight": 1.0, "keywords": ["petroleo", "commodities", "cambio"]},
    "credit_conditions": {"weight": 0.85, "keywords": ["credito", "inadimplencia"]},
    "policy_decision": {"weight": 1.5, "keywords": ["selic", "decidiu", "decisao"]},
    "forward_guidance": {"weight": 1.4, "keywords": ["proximos passos", "ciclo", "magnitude"]},
    "institutional": {"weight": 0.2, "keywords": ["votaram"]},
}

LEXICON = {
    "hawkish": [
        {"term": "desancoradas", "weight": 2.0},
        {"term": "cautela adicional", "weight": 1.0},
        {"term": "premio de risco", "weight": 1.0},
    ],
    "dovish": [{"term": "desaceleracao", "weight": 1.0}, {"term": "ociosidade", "weight": 1.0}, {"term": "arrefecimento", "weight": 1.0}],
}

V2_SETTINGS = {
    "project": {
        "default_model_version": "test-baseline",
        "prompt_version": "test-prompt",
        "taxonomy_version": "test-taxonomy",
        "lexicon_version": "test-lexicon",
    },
    "scoring": {
        "neutral_band": 0.15,
        "low_information_threshold": 0.5,
        "communication_weight": 0.6,
        "minutes_weight": 0.4,
        "default_calibration": "calibration_v1_2006_2019",
        "min_calibration_observations": 2,
    },
    "calibrations": {
        "calibration_v1_2006_2019": {"start": "2006-01-01", "end": "2019-12-31", "official": True}
    },
    "subindices": {
        "inflation_pressure": {"label": "Inflation Pressure Index", "topics": ["inflation_current"]},
        "expectations_anchoring": {"label": "Expectations Anchoring Index", "topics": ["inflation_expectations"]},
        "activity_slack": {"label": "Activity Slack Index", "topics": ["activity_growth"]},
    },
    "reaction_function": {
        "version": "test-rf",
        "label": "Text-Implied Reaction Function Index",
        "weights": {"inflation_pressure": 1.0, "expectations_anchoring": 1.0, "activity_slack": -0.7},
    },
    "redline": {"maintained_similarity": 0.9, "related_similarity": 0.55, "tone_change_threshold": 0.15},
}


def _documents() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "document_id": "copom_1_comunicado",
                "meeting_id": "copom_1",
                "nro_reuniao": 1,
                "document_type": "comunicado",
                "publication_date": pd.Timestamp("2010-01-01"),
                "title": "Comunicado 1",
                "url": "",
                "raw_text": "As expectativas desancoradas exigem cautela adicional.",
                "source": "fixture",
            },
            {
                "document_id": "copom_2_comunicado",
                "meeting_id": "copom_2",
                "nro_reuniao": 2,
                "document_type": "comunicado",
                "publication_date": pd.Timestamp("2010-02-01"),
                "title": "Comunicado 2",
                "url": "",
                "raw_text": "A inflacao e a atividade mostram desaceleracao e ociosidade.",
                "source": "fixture",
            },
        ]
    )


def _meetings() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"meeting_id": "copom_1", "nro_reuniao": 1, "data_referencia": pd.Timestamp("2010-01-01")},
            {"meeting_id": "copom_2", "nro_reuniao": 2, "data_referencia": pd.Timestamp("2010-02-01")},
        ]
    )


def _score_sentence(text: str) -> pd.Series:
    frame = pd.DataFrame(
        [
            {
                "sentence_id": "fixture_sentence",
                "document_id": "fixture_document",
                "meeting_id": "copom_1",
                "nro_reuniao": 1,
                "document_type": "ata",
                "sentence_order": 1,
                "text": text,
                "sentence_hash": stable_text_hash(text),
                "source_hash": stable_text_hash(text),
                "run_id": "run1",
            }
        ]
    )
    return score_v2_sentences(frame, TOPICS, LEXICON, V2_SETTINGS, "run1").iloc[0]


def test_pipeline_idempotency_for_v2_sentence_keys() -> None:
    meetings, documents = prepare_v2_backfill(_meetings(), _documents(), "run1", {"backfill": {"min_minutes_meeting": 1, "min_statement_meeting": 1}})
    first = build_v2_sentences(documents, "run1")
    second = build_v2_sentences(documents, "run2")
    assert first["sentence_id"].tolist() == second["sentence_id"].tolist()
    assert not first["sentence_id"].duplicated().any()
    assert len(meetings) == 2


def test_source_hash_change_detection() -> None:
    assert stable_text_hash("texto original") != stable_text_hash("texto alterado")


def test_v2_source_url_falls_back_to_official_detail_endpoint() -> None:
    documents = _documents()
    documents["url"] = ""
    enriched = ensure_v2_document_source_urls(documents)

    assert "source_url" in enriched.columns
    assert enriched["source_url"].str.contains("bcb.gov.br/api/servico/sitebcb/copom").all()
    assert enriched.loc[enriched["document_type"] == "comunicado", "source_url"].str.contains("comunicados_detalhes").all()


def test_score_reproducibility() -> None:
    _, documents = prepare_v2_backfill(_meetings(), _documents(), "run1", {"backfill": {"min_minutes_meeting": 1, "min_statement_meeting": 1}})
    sentences = build_v2_sentences(documents, "run1")
    first = score_v2_sentences(sentences, TOPICS, LEXICON, V2_SETTINGS, "run1")
    second = score_v2_sentences(sentences, TOPICS, LEXICON, V2_SETTINGS, "run1")
    assert first[["sentence_id", "tone_level", "primary_topic", "stance"]].equals(
        second[["sentence_id", "tone_level", "primary_topic", "stance"]]
    )


def test_empty_or_low_information_document() -> None:
    docs = _documents().iloc[[0]].copy()
    docs["raw_text"] = "Votaram por essa decisao os membros do comite."
    meetings, docs = prepare_v2_backfill(_meetings().iloc[[0]], docs, "run1", {"backfill": {"min_minutes_meeting": 1, "min_statement_meeting": 1}})
    sentences = build_v2_sentences(docs, "run1")
    scores = score_v2_sentences(sentences, TOPICS, LEXICON, V2_SETTINGS, "run1")
    document_scores, meeting_scores = aggregate_v2_scores(meetings, docs, scores, V2_SETTINGS, "run1")
    assert document_scores["score_status"].iloc[0] == "low_information"
    assert meeting_scores["score_status"].iloc[0] == "low_information"


def test_v2_institutional_header_filter_neutralizes_summary_blocks() -> None:
    scored = _score_sentence(
        "Sumario: Data: 10/01/2000. Local: Brasilia. Presentes: Membros do Copom e Chefes de Departamento. "
        "Credito, inadimplencia e condicoes financeiras foram listados no expediente."
    )

    assert scored["primary_topic"] == "institutional"
    assert scored["stance"] == "neutral"
    assert bool(scored["is_informative"]) is False
    assert scored["tone_level"] == 0


def test_v2_incc_sentence_is_inflation_current_and_informative() -> None:
    scored = _score_sentence("O INCC apresentou arrefecimento, com menor variacao dos custos da construcao civil.")

    assert scored["primary_topic"] == "inflation_current"
    assert scored["stance"] == "dovish"
    assert bool(scored["is_informative"]) is True


def test_v2_oil_upside_rule_is_hawkish() -> None:
    scored = _score_sentence("Os precos do petroleo apresentaram alta expressiva, com pressao altista adicional.")

    assert scored["primary_topic"] == "fx_commodities"
    assert scored["stance"] == "hawkish"
    assert scored["stance_score"] > 0


def test_v2_risk_premium_down_rule_is_dovish() -> None:
    scored = _score_sentence("O premio de risco-Brasil apresentou queda em ambiente favoravel.")

    assert scored["stance"] == "dovish"
    assert scored["stance_score"] < 0


def test_v2_expectations_upside_is_hawkish_and_expectations_topic() -> None:
    scored = _score_sentence("A mediana das expectativas para o IPCA continuou em tendencia de elevacao.")

    assert scored["primary_topic"] == "inflation_expectations"
    assert scored["stance"] == "hawkish"
    assert bool(scored["is_informative"]) is True


def test_v2_expectations_projection_beats_policy_decision_topic() -> None:
    scored = _score_sentence(
        "No cenario de referencia, a projecao de inflacao para 2006 permanece acima da meta, mesmo com a taxa Selic constante."
    )

    assert scored["primary_topic"] == "inflation_expectations"
    assert scored["stance"] == "hawkish"


def test_v2_expectations_downside_is_dovish() -> None:
    scored = _score_sentence("As expectativas se reduziram ligeiramente, ficando mais proximas da meta.")

    assert scored["primary_topic"] == "inflation_expectations"
    assert scored["stance"] == "dovish"


def test_v2_activity_strength_is_hawkish() -> None:
    scored = _score_sentence("Os dados indicaram continuidade do forte crescimento do nivel de atividade.")

    assert scored["primary_topic"] == "activity_growth"
    assert scored["stance"] == "hawkish"


def test_v2_domestic_activity_beats_external_and_guidance_terms() -> None:
    scored = _score_sentence(
        "As importacoes de bens de capital e as vendas internas de autoveiculos indicam continuidade da expansao da demanda domestica."
    )

    assert scored["primary_topic"] == "activity_growth"
    assert scored["stance"] == "hawkish"


def test_v2_labor_market_beats_fx_real_keyword() -> None:
    scored = _score_sentence("A massa salarial real cresceu, com rendimento medio e emprego em expansao.")

    assert scored["primary_topic"] == "labor_market"
    assert scored["stance"] == "hawkish"


def test_v2_projection_model_sentence_is_low_information() -> None:
    scored = _score_sentence(
        "Esse modelo considera componentes sazonais, variacoes cambiais e projecao para o spread da taxa Selic."
    )

    assert scored["primary_topic"] == "institutional"
    assert bool(scored["is_informative"]) is False


def test_v2_section_heading_is_low_information() -> None:
    scored = _score_sentence("Credito e inadimplencia 48.")

    assert scored["primary_topic"] == "institutional"
    assert bool(scored["is_informative"]) is False


def test_v2_actual_selic_decision_overrides_forward_guidance() -> None:
    scored = _score_sentence(
        "O Copom decidiu reduzir a taxa Selic para 17,25% ao ano e definir os proximos passos na estrategia de politica monetaria."
    )

    assert scored["primary_topic"] == "policy_decision"
    assert scored["stance"] == "dovish"


def test_v2_monitoring_next_steps_is_forward_guidance() -> None:
    scored = _score_sentence(
        "O Comite ira monitorar atentamente a evolucao do cenario macroeconomico ate sua proxima reuniao, para entao definir os proximos passos."
    )

    assert scored["primary_topic"] == "forward_guidance"


def test_v2_future_adjustment_after_rate_cuts_is_dovish_guidance() -> None:
    scored = _score_sentence(
        "Considerando que os efeitos das reducoes de juros ainda nao se refletiram integralmente, a decisao contribuira para aumentar a magnitude do ajuste a ser implementado."
    )

    assert scored["primary_topic"] == "forward_guidance"
    assert scored["stance"] == "dovish"


def test_v2_reduction_of_inflation_pressures_is_dovish() -> None:
    scored = _score_sentence("A trajetoria dos indices de precos evidencia reducao das pressoes inflacionarias externas.")

    assert scored["stance"] == "dovish"
    assert scored["stance_score"] < 0


def test_v2_benign_scenario_risk_reversal_is_hawkish() -> None:
    scored = _score_sentence("Esses fatores podem aumentar os riscos para o cenario benigno de inflacao.")

    assert scored["stance"] == "hawkish"
    assert scored["stance_score"] > 0


def test_v2_taxonomy_boundary_flag_policy_vs_guidance() -> None:
    scored = _score_sentence(
        "O Copom decidiu manter a taxa Selic, mas avaliara os proximos passos da estrategia de politica monetaria."
    )

    assert scored["primary_topic"] == "policy_decision"
    assert scored["taxonomy_boundary_flag"] == "policy_decision_vs_forward_guidance"
    assert scored["rule_engine_version"] == "taxonomy-rules-v2.0.4"


def test_v2_production_drop_is_activity_dovish() -> None:
    scored = _score_sentence("A producao industrial registrou queda no mes, com desaceleracao da atividade fabril.")

    assert scored["primary_topic"] == "activity_growth"
    assert scored["stance"] == "dovish"


def test_v2_resilient_demand_is_activity_hawkish() -> None:
    scored = _score_sentence("A demanda domestica segue resiliente, com forte crescimento do comercio varejista.")

    assert scored["primary_topic"] == "activity_growth"
    assert scored["stance"] == "hawkish"


def test_v2_decreasing_benign_scenario_risks_are_dovish() -> None:
    scored = _score_sentence("Sao decrescentes os riscos para a consolidacao de um cenario inflacionario benigno.")

    assert scored["primary_topic"] == "risk_balance"
    assert scored["stance"] == "dovish"


def test_v2_global_producer_inflation_is_external_topic() -> None:
    scored = _score_sentence("A inflacao ao produtor norte-americana foi pressionada pelo preco do petroleo.")

    assert scored["primary_topic"] == "external_environment"
    assert scored["primary_topic"] != "inflation_current"


def test_v2_global_labor_indicator_stays_external() -> None:
    scored = _score_sentence("A taxa de desemprego na Alemanha subiu, reforcando preocupacoes com a economia europeia.")

    assert scored["primary_topic"] == "external_environment"


def test_v2_next_steps_without_decision_is_forward_guidance_not_boundary() -> None:
    scored = _score_sentence("O Comite avaliara os proximos passos da estrategia de politica monetaria.")

    assert scored["primary_topic"] == "forward_guidance"
    assert scored["taxonomy_boundary_flag"] == ""


def test_v2_expectations_above_target_guard_beats_policy_easing_word() -> None:
    scored = _score_sentence(
        "As projecoes de inflacao ficam acima das metas, pois pressupõem reducao da taxa de juros basica."
    )

    assert scored["primary_topic"] == "inflation_expectations"
    assert scored["stance"] == "hawkish"


def test_v2_forward_guidance_convergence_statement_can_be_neutral() -> None:
    scored = _score_sentence("A estrategia visa assegurar a convergencia da inflacao para a trajetoria de metas.")

    assert scored["primary_topic"] == "forward_guidance"
    assert scored["stance"] == "neutral"


def test_v2_falling_unemployment_is_labor_hawkish() -> None:
    scored = _score_sentence("A taxa de desemprego recuou e atingiu o menor nivel da serie historica.")

    assert scored["primary_topic"] == "labor_market"
    assert scored["stance"] == "hawkish"


def test_v2_inflation_reacceleration_guard_is_hawkish() -> None:
    scored = _score_sentence("A inflacao havia desacelerado, mas voltou a subir no mes corrente.")

    assert scored["primary_topic"] == "inflation_current"
    assert scored["stance"] == "hawkish"


def test_v2_market_uncertainty_easing_is_dovish() -> None:
    scored = _score_sentence("A volatilidade permaneceu elevada, mas houve melhora em relacao ao cenario anterior.")

    assert scored["primary_topic"] == "uncertainty"
    assert scored["stance"] == "dovish"


def test_v2_easing_of_commodity_upside_pressure_is_dovish() -> None:
    scored = _score_sentence("Houve arrefecimento nas pressoes altistas, especialmente para as commodities agricolas.")

    assert scored["primary_topic"] == "fx_commodities"
    assert scored["stance"] == "dovish"


def test_conflicting_multilabel_topics_enter_subindices() -> None:
    _, documents = prepare_v2_backfill(_meetings(), _documents(), "run1", {"backfill": {"min_minutes_meeting": 1, "min_statement_meeting": 1}})
    sentences = build_v2_sentences(documents, "run1")
    scores = score_v2_sentences(sentences, TOPICS, LEXICON, V2_SETTINGS, "run1")
    _, meeting_scores = aggregate_v2_scores(_meetings(), documents, scores, V2_SETTINGS, "run1")
    calibration = build_v2_calibration(meeting_scores, V2_SETTINGS, "run1")
    subindices = build_v2_subindices(scores, meeting_scores, V2_SETTINGS, "run1")
    meeting_2 = subindices[subindices["meeting_id"] == "copom_2"]
    assert {"inflation_pressure", "activity_slack"}.issubset(set(meeting_2["subindex"]))
    assert calibration["status"].iloc[0] == "ok"


def test_v2_evidence_has_sentence_citation() -> None:
    _, documents = prepare_v2_backfill(_meetings(), _documents(), "run1", {"backfill": {"min_minutes_meeting": 1, "min_statement_meeting": 1}})
    sentences = build_v2_sentences(documents, "run1")
    scores = score_v2_sentences(sentences, TOPICS, LEXICON, V2_SETTINGS, "run1")
    evidence = build_v2_evidence(scores, pd.DataFrame(), top_n=3)

    assert not evidence.empty
    assert "citation" in evidence.columns
    assert evidence["citation"].str.contains("Reuniao").all()
    assert evidence["citation"].str.contains("sentenca").all()


def test_redline_detects_tone_changed_sentence() -> None:
    _, documents = prepare_v2_backfill(_meetings(), _documents(), "run1", {"backfill": {"min_minutes_meeting": 1, "min_statement_meeting": 1}})
    sentences = build_v2_sentences(documents, "run1")
    scores = score_v2_sentences(sentences, TOPICS, LEXICON, V2_SETTINGS, "run1")
    redline = build_v2_redline(documents, scores, V2_SETTINGS, "run1")
    assert (redline["change_type"].isin(["added", "tone_changed", "rewritten", "removed", "maintained"])).all()
    assert not redline.empty


def test_redline_match_prefers_token_candidate() -> None:
    previous = pd.DataFrame(
        [
            {"sentence_id": "s1", "text": "A atividade mostra desaceleracao e ociosidade."},
            {"sentence_id": "s2", "text": "As expectativas de inflacao seguem desancoradas e exigem cautela."},
        ]
    )

    row, similarity = best_previous_match("As expectativas seguem desancoradas e exigem cautela adicional.", previous, set())

    assert row is not None
    assert row["sentence_id"] == "s2"
    assert similarity > 0.5


def test_redline_is_current_when_expected_pairs_are_covered() -> None:
    _, documents = prepare_v2_backfill(_meetings(), _documents(), "run1", {"backfill": {"min_minutes_meeting": 1, "min_statement_meeting": 1}})
    sentences = build_v2_sentences(documents, "run1")
    scores = score_v2_sentences(sentences, TOPICS, LEXICON, V2_SETTINGS, "run1")
    redline = build_v2_redline(documents, scores, V2_SETTINGS, "run1")

    assert redline_is_current(redline, documents, scores, min_coverage=1.0)


def test_no_lookahead_market() -> None:
    events = pd.DataFrame(
        [
            {
                "meeting_id": "copom_1",
                "document_type": "comunicado",
                "known_at_timestamp": pd.Timestamp("2024-01-10 18:30"),
            }
        ]
    )
    observations = normalize_market_observations(
        pd.DataFrame(
            [
                {"asset": "DI", "vertex": "1y", "timestamp": "2024-01-10 17:00", "value": 10.0},
                {"asset": "DI", "vertex": "1y", "timestamp": "2024-01-11 10:00", "value": 10.2},
            ]
        )
    )
    windows = build_market_event_windows(events, observations)
    ok = windows[windows["status"] == "ok"].iloc[0]
    assert ok["pre_timestamp"] < events["known_at_timestamp"].iloc[0]
    assert ok["post_timestamp"] > events["known_at_timestamp"].iloc[0]
    assert round(float(ok["market_reaction"]), 6) == 0.2
    assert {"close_to_next_close", "close_to_second_close"}.issubset(set(windows["window"]))


def test_market_intraday_window_and_metadata() -> None:
    events = pd.DataFrame(
        [
            {
                "meeting_id": "copom_1",
                "document_type": "comunicado",
                "release_date": pd.Timestamp("2024-01-10"),
                "market_close_convention": "after_close",
            }
        ]
    )
    observations = normalize_market_observations(
        pd.DataFrame(
            [
                {"asset": "DI", "asset_class": "rates", "vertex": "1y", "timestamp": "2024-01-10 17:00", "value": 10.0},
                {"asset": "DI", "asset_class": "rates", "vertex": "1y", "timestamp": "2024-01-10 19:00", "value": 10.1},
                {"asset": "DI", "asset_class": "rates", "vertex": "1y", "timestamp": "2024-01-11 17:00", "value": 10.2},
            ]
        )
    )

    windows = build_market_event_windows(events, observations)
    intraday = windows[(windows["window"] == "intraday_before_after") & (windows["status"] == "ok")].iloc[0]

    assert intraday["asset_class"] == "rates"
    assert round(float(intraday["market_reaction"]), 6) == 0.1
    assert intraday["known_at_timestamp"] == pd.Timestamp("2024-01-10 18:30")


def test_decision_expectations_and_v21_event_panel() -> None:
    meetings = pd.DataFrame(
        [
            {
                "meeting_id": "copom_1",
                "nro_reuniao": 1,
                "data_referencia": pd.Timestamp("2024-01-10"),
                "delta_selic": -0.25,
            }
        ]
    )
    scores = pd.DataFrame([{"meeting_id": "copom_1", "communication_surprise_naive": 0.4}])
    expectations = normalize_decision_expectations(
        pd.DataFrame(
            [
                {
                    "meeting_id": "copom_1",
                    "as_of_timestamp": "2024-01-09 17:00",
                    "expected_selic_change_bps": -50,
                    "source": "fixture",
                }
            ]
        )
    )
    focus = pd.DataFrame(
        [
            {
                "meeting_id": "copom_1",
                "delta_post_1": 0.1,
                "delta_post_2": pd.NA,
            }
        ]
    )
    market = pd.DataFrame(
        [
            {
                "meeting_id": "copom_1",
                "status": "ok",
                "market_reaction": 0.2,
            }
        ]
    )

    panel = build_v21_event_panel(meetings, scores, focus, market, expectations)
    row = panel.iloc[0]

    assert row["selic_decision_bps"] == -25.0
    assert row["decision_surprise_bps"] == 25.0
    assert row["decision_surprise_official_bps"] == 25.0
    assert pd.isna(row["decision_surprise_proxy_bps"])
    assert row["decision_surprise_status"] == "official"
    assert row["communication_surprise_naive"] == 0.4
    assert row["focus_status"] == "ok"


def test_public_decision_proxy_does_not_fill_official_surprise() -> None:
    meetings = pd.DataFrame(
        [
            {
                "meeting_id": "copom_1",
                "nro_reuniao": 1,
                "data_referencia": pd.Timestamp("2024-01-10"),
                "selic_pre": 11.75,
                "delta_selic": -0.25,
            }
        ]
    )
    focus_features = pd.DataFrame(
        [
            {
                "meeting_id": "copom_1",
                "indicator": "Selic",
                "horizon": "current_year",
                "statistic": "median",
                "pre_date": "2024-01-08",
                "pre_value": 11.5,
            }
        ]
    )

    expectations, audit = derive_public_decision_expectations(meetings, focus_features, pd.DataFrame())
    panel = build_v21_event_panel(meetings, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), expectations)
    row = panel.iloc[0]

    assert bool(expectations["is_proxy"].iloc[0])
    assert "b3:opcao_copom" in set(audit["source"])
    assert pd.isna(row["decision_surprise_official_bps"])
    assert row["decision_surprise_proxy_bps"] == 0.0
    assert row["decision_surprise_status"] == "proxy"


def test_anbima_intraday_fixture_generates_public_market_observations() -> None:
    html = """
    <table>
      <tr><th>Dias</th><th>D-1</th><th>D0</th></tr>
      <tr><td>63</td><td>10,00</td><td>10,10</td></tr>
      <tr><td>126</td><td>10,20</td><td>10,30</td></tr>
      <tr><td>252</td><td>10,40</td><td>10,50</td></tr>
      <tr><td>504</td><td>10,60</td><td>10,70</td></tr>
      <tr><td>1260</td><td>10,80</td><td>10,90</td></tr>
    </table>
    <table>
      <tr><th>Dias</th><th>D-1</th><th>D0</th></tr>
      <tr><td>63</td><td>5,00</td><td>5,10</td></tr>
      <tr><td>252</td><td>5,40</td><td>5,50</td></tr>
      <tr><td>1260</td><td>5,80</td><td>5,90</td></tr>
    </table>
    """

    observations = parse_anbima_intraday_html(html, pd.Timestamp("2024-01-10"))

    assert {"ANBIMA_ETTJ_PRE", "ANBIMA_ETTJ_REAL"}.issubset(set(observations["asset"]))
    assert {"3m", "6m", "1y", "2y", "5y"}.issubset(set(observations["vertex"]))
    assert observations["data_access_tier"].eq("PUBLIC_API").all()


def test_public_market_coverage_separates_official_and_proxy_expectations() -> None:
    observations = normalize_market_observations(
        pd.DataFrame([{"asset": "USD_BRL_PTAX", "vertex": "spot", "timestamp": "2024-01-10", "value": 5.0}])
    )
    expectations = normalize_decision_expectations(
        pd.DataFrame(
            [
                {"meeting_id": "copom_1", "as_of_timestamp": "2024-01-09", "expected_selic_change_bps": -25, "is_proxy": False},
                {"meeting_id": "copom_2", "as_of_timestamp": "2024-02-09", "expected_selic_change_bps": -50, "is_proxy": True},
            ]
        )
    )

    coverage = build_public_market_coverage(observations, pd.DataFrame(), expectations, pd.DataFrame(), pd.DataFrame())
    metrics = {(row["section"], row["metric"]): row["value"] for _, row in coverage.iterrows()}

    assert metrics[("decision_expectations", "official_rows")] == 1
    assert metrics[("decision_expectations", "proxy_rows")] == 1


def test_v21_health_outputs_and_b3_warning(tmp_path, monkeypatch) -> None:
    database = tmp_path / "copom_tone.duckdb"
    tables = _v21_fixture_tables()
    write_tables(database, tables)
    paths = ProjectPaths(
        database=database,
        raw=tmp_path / "raw",
        processed=tmp_path / "outputs" / "processed",
        figures=tmp_path / "outputs" / "figures",
        reports=tmp_path / "reports" / "meeting_notes",
    )
    monkeypatch.setattr("copom_tone_index.v21.get_paths", lambda: paths)
    monkeypatch.setattr("copom_tone_index.v2_health.build_v2_acceptance_report", lambda database: {"status": "warning", "warnings": [], "errors": []})

    result = v21_health_command()

    assert result.json_path.exists()
    assert result.html_path.exists()
    assert (tmp_path / "outputs" / "v2" / "v21_acceptance_by_meeting.csv").exists()
    assert (tmp_path / "outputs" / "v2" / "v21_acceptance_by_source.csv").exists()
    assert result.status == "warning"
    html = result.html_path.read_text(encoding="utf-8")
    assert "COPOM Watch V2.1 Acceptance Report" in html
    assert result.errors == 0


def test_v21_health_fails_without_event_panel() -> None:
    report, _, _ = build_v21_acceptance_report_from_tables({**_v21_fixture_tables(), "v21_event_panel": pd.DataFrame()})

    assert report["status"] == "fail"
    assert any("event panel" in error.lower() for error in report["errors"])


def test_v21_health_detects_duplicate_and_lookahead() -> None:
    tables = _v21_fixture_tables()
    tables["market_event_windows"] = pd.concat(
        [
            tables["market_event_windows"],
            tables["market_event_windows"].assign(pre_timestamp=pd.Timestamp("2024-01-10 20:00")).iloc[[0]],
        ],
        ignore_index=True,
    )
    report, _, _ = build_v21_acceptance_report_from_tables(tables)

    assert report["status"] == "fail"
    assert report["idempotency"]["market_event_windows"] > 0
    assert report["lookahead"]["market_lookahead_violations"] > 0


def build_v21_acceptance_report_from_tables(tables: dict[str, pd.DataFrame]) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    database = Path("v21_fixture_unused.duckdb")
    from copom_tone_index.v21 import (
        build_v21_acceptance_by_meeting,
        build_v21_acceptance_by_source,
        collect_v21_findings,
        v21_decision_summary,
        v21_event_panel_summary,
        v21_focus_summary,
        v21_idempotency_summary,
        v21_lookahead_summary,
        v21_market_summary,
    )

    by_meeting = build_v21_acceptance_by_meeting(tables)
    by_source = build_v21_acceptance_by_source(tables)
    focus = v21_focus_summary(tables["focus_vintages"], tables["focus_event_features"])
    market = v21_market_summary(tables["market_observations"], tables["market_event_windows"], tables["public_market_source_audit"])
    decision = v21_decision_summary(tables["decision_expectations"], tables["decision_expectation_source_audit"], tables["v21_event_panel"])
    panel = v21_event_panel_summary(tables["v21_event_panel"])
    idempotency = v21_idempotency_summary(tables)
    lookahead = v21_lookahead_summary(tables["focus_event_features"], tables["market_event_windows"], tables["decision_expectations"], tables["v21_event_panel"])
    health = {"status": "warning", "warnings": 1, "errors": 0}
    warnings: list[str] = []
    errors: list[str] = []
    collect_v21_findings(warnings, errors, focus, market, decision, panel, idempotency, lookahead, health)
    status = "fail" if errors else "warning" if warnings else "pass"
    return (
        {
            "status": status,
            "database_path": str(database),
            "focus": focus,
            "market": market,
            "decision_expectations": decision,
            "event_panel": panel,
            "idempotency": idempotency,
            "lookahead": lookahead,
            "warnings": warnings,
            "errors": errors,
        },
        by_meeting,
        by_source,
    )


def _v21_fixture_tables() -> dict[str, pd.DataFrame]:
    focus_vintages = pd.DataFrame(
        [
            {
                "focus_release_date": "2024-01-08",
                "focus_reference_date": "2024-12-31",
                "indicator": "Selic",
                "reference_year": 2024,
                "horizon": "current_year",
                "statistic": "median",
                "value": 11.5,
                "source": "focus_odata",
                "source_date": pd.NaT,
                "collected_at": "2024-01-08",
                "query_signature": "q1",
                "data_access_tier": "PUBLIC_API",
            }
        ]
    )
    focus_features = pd.DataFrame(
        [
            {
                "meeting_id": "copom_1",
                "nro_reuniao": 1,
                "event_type": "comunicado",
                "event_date": pd.Timestamp("2024-01-10"),
                "known_at_timestamp": pd.Timestamp("2024-01-10 18:30"),
                "indicator": "Selic",
                "reference_year": 2024,
                "horizon": "current_year",
                "statistic": "median",
                "pre_date": pd.Timestamp("2024-01-08"),
                "pre_value": 11.5,
                "post_1_date": pd.Timestamp("2024-01-11"),
                "post_1_value": 11.4,
                "post_2_date": pd.Timestamp("2024-01-12"),
                "post_2_value": 11.3,
                "delta_post_1": -0.1,
                "delta_post_2": -0.2,
                "missing_reason": "",
                "source": "focus_odata",
                "data_access_tier": "PUBLIC_API",
            }
        ]
    )
    market_observations = normalize_market_observations(
        pd.DataFrame(
            [
                {"asset": "USD_BRL_PTAX", "asset_class": "fx", "vertex": "spot", "timestamp": "2024-01-10 17:00", "value": 5.0, "source": "bcb_ptax"},
                {"asset": "USD_BRL_PTAX", "asset_class": "fx", "vertex": "spot", "timestamp": "2024-01-11 17:00", "value": 5.1, "source": "bcb_ptax"},
            ]
        )
    )
    market_windows = build_market_event_windows(
        pd.DataFrame([{"meeting_id": "copom_1", "nro_reuniao": 1, "document_type": "comunicado", "known_at_timestamp": pd.Timestamp("2024-01-10 18:30")}]),
        market_observations,
    )
    expectations = normalize_decision_expectations(
        pd.DataFrame(
            [
                {
                    "meeting_id": "copom_1",
                    "source": "focus_selic_proxy",
                    "as_of_timestamp": "2024-01-09 08:00",
                    "expected_selic_change_bps": -25.0,
                    "is_proxy": True,
                    "proxy_method": "focus_selic_year_end_minus_current_selic",
                }
            ]
        )
    )
    panel = build_v21_event_panel(
        pd.DataFrame([{"meeting_id": "copom_1", "nro_reuniao": 1, "data_referencia": pd.Timestamp("2024-01-10"), "delta_selic": -0.25}]),
        pd.DataFrame(),
        focus_features,
        market_windows,
        expectations,
    )
    return {
        "focus_vintages": focus_vintages,
        "focus_event_features": focus_features,
        "market_observations": market_observations,
        "market_event_windows": market_windows,
        "decision_expectations": expectations,
        "public_market_source_audit": pd.DataFrame([{"source": "ptax:USD_BRL", "status": "ok", "rows": 2, "detail": ""}]),
        "decision_expectation_source_audit": pd.DataFrame(
            [
                {"source": "b3:opcao_copom", "status": "unavailable", "rows": 0, "detail": "No structured public history."},
                {"source": "public:proxy", "status": "ok", "rows": 1, "detail": "Proxy only."},
            ]
        ),
        "public_market_coverage": pd.DataFrame(columns=["section", "metric", "value", "status"]),
        "v21_event_panel": panel,
    }


def test_schema_migrations_nonexistent_database() -> None:
    schema = validate_v2_schema(Path("missing_v2_test.duckdb"))
    assert not schema["exists"].any()


def test_existing_v2_backfill_sufficiency_threshold(tmp_path) -> None:
    database = tmp_path / "copom_tone.duckdb"
    meetings = pd.DataFrame({"meeting_id": [f"copom_{i}" for i in range(1, 251)]})
    documents = pd.DataFrame({"document_id": [f"doc_{i}" for i in range(1, 251)]})
    import duckdb

    with duckdb.connect(str(database)) as con:
        con.register("meetings", meetings)
        con.register("documents", documents)
        con.execute("CREATE TABLE v2_meetings AS SELECT * FROM meetings")
        con.execute("CREATE TABLE v2_documents AS SELECT * FROM documents")

    assert existing_v2_backfill_is_sufficient(database, quantity=400)
    assert existing_v2_backfill_is_sufficient(database, quantity=50)


def test_coalesce_llm_preserves_baseline_when_update_missing() -> None:
    baseline = pd.DataFrame(
        [
            {"sentence_id": "s1", "topic": "inflation_current", "stance": "hawkish", "stance_score": 1.0, "confidence": 0.8, "rationale": "b", "evidence_terms": [], "model_version": "base", "prompt_version": "p1"},
            {"sentence_id": "s2", "topic": "activity_growth", "stance": "dovish", "stance_score": -1.0, "confidence": 0.8, "rationale": "b", "evidence_terms": [], "model_version": "base", "prompt_version": "p1"},
        ]
    )
    llm = pd.DataFrame(
        [
            {"sentence_id": "s1", "topic": "fiscal_risk", "stance": "neutral", "stance_score": 0.0, "confidence": 0.7, "rationale": "l", "evidence_terms": [], "model_version": "llm", "prompt_version": "p2"}
        ]
    )
    merged = coalesce_llm_with_baseline_for_test(baseline, llm)
    assert merged.loc[merged["sentence_id"] == "s1", "topic"].iloc[0] == "fiscal_risk"
    assert merged.loc[merged["sentence_id"] == "s2", "topic"].iloc[0] == "activity_growth"
    assert merged.loc[merged["sentence_id"] == "s2", "stance_score"].iloc[0] == -1.0

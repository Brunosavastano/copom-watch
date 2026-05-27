# COPOM Watch

Aplicacao analitica para transformar comunicados e atas do COPOM em um indicador quantitativo, interpretavel e auditavel de tom de politica monetaria hawkish/dovish.

O projeto segue a SPEC em `SPEC_COPOM_Tone_Index.md` e entrega um MVP local com ingestao de dados publicos do Banco Central, scoring textual, revisoes Focus, dashboard e notas automaticas por reuniao.

A V2 adiciona um pipeline paralelo para medir comunicacao marginal do BCB com historico ampliado, indice estavel, decomposicao por subindice, redline textual, evidencias e versionamento metodologico. Os comandos V1 foram preservados.

## Acesse o app

O deploy publico foi preparado para o Streamlit Community Cloud.

- Repositorio: `Brunosavastano/copom-watch`
- Branch de deploy: `main`
- Arquivo principal: `streamlit_app.py`
- URL sugerida no Streamlit Cloud: `https://copom-watch.streamlit.app`

Execucao local equivalente:

```powershell
pip install -r requirements.txt
streamlit run streamlit_app.py --server.port 8502 --server.address localhost
```

O app publico usa `app_data/copom_watch_public.duckdb`, um pacote DuckDB reduzido com tabelas analiticas processadas, indice textual, subindices, evidencias, mudancas textuais, painel Focus/mercado e indice local de busca com citacoes. Ele nao precisa de chave de API nem coleta online para carregar a interface.

Limitacoes do app publico:

- nao preve a proxima Selic;
- nao faz recomendacao de investimento;
- nao afirma causalidade em reacoes de mercado;
- usa dados publicos/processados e pode exibir ausencia de dados oficiais quando uma fonte publica nao oferece historico estruturado.

## Fontes de dados

- COPOM atas e comunicados: APIs publicas do Banco Central do Brasil.
- Selic: SGS serie 432, meta Selic definida pelo COPOM.
- Focus: API OData de Expectativas de Mercado, variaveis anuais `Selic` e `IPCA`.

As respostas brutas das APIs sao cacheadas em `data/raw/`. Se uma fonte falhar temporariamente e houver cache, o pipeline usa o cache e registra aviso. Se nao houver cache, a falha e exposta.

## Metodologia

1. Baixa comunicados e atas do COPOM.
2. Limpa HTML, tabelas, rodapes e blocos institucionais.
3. Segmenta os textos em sentencas.
4. Classifica cada sentenca por topico macroeconomico.
5. Calcula score textual hawkish/dovish por baseline lexico auditavel.
6. Opcionalmente substitui/enriquece a classificacao por LLM se `ANTHROPIC_API_KEY` estiver configurada.
7. Agrega scores por documento e por reuniao:
   - `ToneNowcast = ToneComunicado`
   - `ToneFinal = 0.60 * ToneComunicado + 0.40 * ToneAta`
8. Normaliza o historico disponivel para obter `COPOMToneIndex = 50 + 10 * ToneZ`.
9. Calcula revisoes Focus pre e pos-comunicado/ata.
10. Gera dashboard, tabelas exportaveis, figuras HTML e notas por reuniao.

O baseline nao e sentimento generico: termos de inflacao persistente, expectativas desancoradas e cautela entram como hawkish; desaceleracao, desinflacao e riscos baixistas entram como dovish.

## Instalacao

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

Para habilitar LLM opcional:

```powershell
pip install -e ".[dev,llm]"
$env:ANTHROPIC_API_KEY="sua-chave"
```

Sem chave de API, o projeto roda integralmente com o baseline lexico.

## Execucao

Pipeline completo:

```powershell
copom-watch run-pipeline --months 24 --use-llm never
```

Atualizando tambem a camada Focus:

```powershell
copom-watch run-pipeline --months 24 --use-llm never --refresh-focus
```

Para uma execucao mais curta de smoke test, limite a quantidade de reunioes baixadas:

```powershell
copom-watch run-pipeline --months 24 --copom-quantity 12 --use-llm never
```

Com LLM opcional quando disponivel:

```powershell
copom-watch run-pipeline --months 24 --use-llm auto
```

Validar outputs ja gerados:

```powershell
copom-watch validate
```

Revisar a acuracia econometrica:

```powershell
copom-watch review-econometrics
```

Comandos especificos para Focus:

```powershell
copom-watch fetch-focus --months 24
copom-watch rebuild-focus-revisions
copom-watch focus-coverage
copom-watch import-focus-snapshot --path caminho\snapshot_focus.csv --source-date 2026-04-20
```

Pipeline V2 metodologico:

```powershell
copom-watch v2 backfill --quantity 400
copom-watch v2 score
copom-watch v2 calibrate
copom-watch v2 redline
copom-watch v2 audit
copom-watch v2 benchmark-baseline
copom-watch v2 review-remaining-errors --limit 177
copom-watch v2 train-supervised
copom-watch v2 freeze-release --version v2.0.4-holdout-stance-hardened
copom-watch v2 import-labels --path outputs\v2\manual_label_sample.csv --label-source human
copom-watch v2 export-label-sample --n 300 --out data\labels\review_sample_001.csv
copom-watch v2 health-check
copom-watch v2 report
```

Ou tudo em uma etapa:

```powershell
copom-watch v2 run-all --quantity 400
```

`run-all` reutiliza um backfill V2 amplo ja existente para evitar nova coleta longa em rodadas locais de aceite. Para forcar nova coleta de documentos, use:

```powershell
copom-watch v2 run-all --quantity 400 --refresh-backfill
```

O redline tambem reutiliza a cobertura atual quando ela ja esta consistente com os documentos e sentencas V2. Para recomputar explicitamente:

```powershell
copom-watch v2 redline --force
```

Mercado e busca semantica sao modulos opcionais:

```powershell
copom-watch market fetch-public --sources bcb-sgs,ptax,anbima --months 400
copom-watch market derive-decision-expectations --method public
copom-watch market public-coverage
copom-watch market import-csv --path caminho\market.csv --source user_csv --data-access-tier USER_CSV
copom-watch market event-study
copom-watch semantic build-index
copom-watch semantic search --query "expectativas desancoradas" --top-n 10
copom-watch semantic ask --query "Quando o Copom falou de expectativas desancoradas?" --top-n 8
```

Dashboard:

```powershell
streamlit run streamlit_app.py --server.port 8502 --server.address localhost
```

Para desenvolvimento interno, tambem e possivel abrir diretamente o modulo do dashboard:

```powershell
streamlit run src/copom_tone_index/dashboard/app.py
```

A interface principal e guiada para research macro: explica o objetivo de cada pagina, inclui glossario contextual, mostra evidencias citadas e delimita claramente o escopo da busca com evidencias.

## Outputs

- `data/copom_tone.duckdb`: base analitica consolidada.
- `outputs/processed/*.csv`: tabelas exportaveis.
- `outputs/processed/*.parquet`: tabelas exportaveis em formato colunar.
- `outputs/figures/copom_tone_index.html`: grafico do indice.
- `outputs/figures/topic_heatmap.html`: heatmap por topico.
- `reports/meeting_notes/*.md`: nota interpretativa por reuniao.
- `reports/validation_report.md`: checagens de integridade.
- `reports/econometric_accuracy_review.md`: diagnostico de acuracia econometrica.
- `outputs/econometrics/`: auditoria de dados, sensibilidade do indice, amostra manual e diagnosticos de regressao.
- `outputs/econometrics/focus_coverage_audit.csv`: cobertura de deltas Focus por variavel e ano.
- `reports/v2/*.html`: relatorios HTML simples da V2.
- `reports/v2/acceptance_report.html`: relatorio de aceite tecnico-metodologico da V2.0.
- `reports/v2/acceptance_report.json`: payload estruturado do health-check V2.0.
- `reports/v2/model_audit.html`: relatorio local da validacao humana, com resumo de metricas e erros priorizados.
- `reports/v2/validation_disagreement_report.html`: relatorio que separa conflitos entre revisores, ambiguidade legitima de taxonomia e erro provavel do baseline.
- `reports/v2/baseline_benchmark_report.html`: benchmark permanente do baseline V2 contra labels humanos por amostra, topico e stance.
- `reports/v2/remaining_error_review.html`: revisao qualitativa dos erros provaveis remanescentes, sem aplicar novas regras automaticamente.
- `reports/v2/supervised_model_report.html`: benchmark experimental de modelo supervisionado leve; nao substitui o indice oficial.
- `reports/v2/v2_0_methodology_report.html`: relatorio metodologico de fechamento da V2.0.
- `reports/v2/release_manifest.json`: manifesto reproduzivel da release V2.0 congelada.
- `reports/v2/releases/<version>/`: copia dos principais artefatos da release, com hashes no manifesto.
- `outputs/v2/baseline_benchmark_by_sample.csv`: metricas por sample 001, sample 002 Claude/GPT, consenso do holdout e consenso total.
- `outputs/v2/baseline_benchmark_by_topic.csv`: metricas do baseline por topico humano.
- `outputs/v2/baseline_benchmark_by_stance.csv`: precision, recall e F1 por stance.
- `outputs/v2/model_audit_error_analysis.csv`: frases revisadas em que baseline e humano divergem, com prioridade e acao sugerida.
- `outputs/v2/model_audit_error_classification.csv`: classificacao dos erros entre `likely_baseline_error`, `legitimate_taxonomy_ambiguity`, `taxonomy_boundary_case` e `baseline_error_with_reviewer_disagreement_context`.
- `outputs/v2/model_audit_error_summary.csv`: resumo agregado das principais confusoes por topico, stance e informatividade.
- `outputs/v2/remaining_error_review.csv`: classificacao dos erros remanescentes entre regra candidata, fronteira, limite contextual, revisao de label e nao-tunar.
- `outputs/v2/supervised_model_audit.csv`: metricas do modelo supervisionado experimental por alvo.
- `outputs/v2/supervised_model_predictions.csv`: predicoes holdout do modelo supervisionado experimental.
- `outputs/v2/reviewer_disagreements.csv`: sentencas com discordancia entre revisores humanos.
- `data/labels/review_sample_001.csv`: amostra estratificada para revisao humana, quando exportada.
- `data/labels/review_sample_001_codebook.md`: codebook da amostra humana, com taxonomia, valores permitidos e instrucoes de importacao.

Tabelas principais:

- `copom_meetings`: calendario, datas, Selic e janela operacional.
- `copom_documents`: comunicados/atas e textos limpos.
- `copom_sentences`: sentencas classificadas com topico, tom, score, confianca e evidencias.
- `copom_scores`: score consolidado por reuniao.
- `focus_revisions`: Focus pre e pos-evento.
- `focus_observations`: observacoes Focus auditaveis com fonte, data, mediana e assinatura de consulta.
- `v2_documents`, `v2_sentences`, `v2_sentence_scores`: documentos e sentencas versionadas com `source_hash`, `source_url` auditavel e `run_id`.
- `v2_meeting_scores`: tom V2 por reuniao, `communication_surprise_naive`, versoes e status de calibracao.
- `v2_subindices`: subindices macro e Text-Implied Reaction Function Index.
- `v2_redline`: frases adicionadas, removidas, mantidas, reescritas ou com mudanca de tom.
- `v2_model_audit`: auditoria de labels/modelo; labels LLM bootstrap nao contam como verdade humana.
- `v2_model_audit_details`: comparacao linha a linha entre labels aceitos e predicoes baseline, quando houver validacao humana.
- `v2_supervised_predictions`, `v2_supervised_model_audit`: benchmark experimental supervisionado; nunca alimenta os scores oficiais.
- `market_observations`, `market_event_windows`: modulo opcional de mercado via fontes publicas ou CSV.
- `public_market_source_audit`, `decision_expectation_source_audit`, `public_market_coverage`: auditoria das fontes publicas de mercado e das expectativas/proxies de decisao.
- `semantic_chunks`: indice local de citacoes para busca semantica.
- `outputs/processed/semantic_search_results.csv`: ultimos resultados de busca semantica local, sempre com reuniao/documento/sentenca citados.
- `outputs/processed/semantic_ask_results.csv`: citacoes usadas pelo modo `Ask COPOM Watch`.
- `app_data/copom_watch_public.duckdb`: pacote DuckDB reduzido para deploy Streamlit.
- `app_data/public_data_manifest.json`: manifesto do pacote publico com hashes e tabelas incluidas.

## Aceite V2.0

O comando abaixo inspeciona a base V2 existente e gera um relatorio local de aceite:

```powershell
copom-watch v2 health-check
```

O health-check avalia cobertura de documentos, sentencas, scores, calibracao fixa, formula do `tone_level`, subindices, redline, evidencias, labels, mercado opcional, semantica opcional, metadados de origem (`source_url`) e duplicidades de idempotencia. O JSON e o HTML tambem incluem `implementation_status`, com status visual macro do roadmap V2 e micro dos componentes tecnicos. Mercado e semantica ausentes geram warning ou info, nao erro.

Para preparar validacao humana:

```powershell
copom-watch v2 export-label-sample --n 300 --out data\labels\review_sample_001.csv
```

O CSV sai com campos humanos vazios (`human_topic`, `human_stance`, `human_is_informative`, `human_notes`, `reviewer_id`, `accepted`) para revisao posterior. O comando tambem gera um codebook `.md` no mesmo diretorio do CSV, com taxonomia, stance permitida, regras de aceite e comando de reimportacao. Nenhum label e gerado por LLM nessa etapa.
Ao reimportar, linhas ainda sem revisao ficam como `pending_review`; somente linhas humanas preenchidas ou `accepted=true` entram na auditoria formal. Labels com `label_source=llm_bootstrap` permanecem excluidos da verdade formal mesmo quando marcados como aceitos.
A auditoria reporta acuracia por stance/topico, F1 macro, matriz de confusao e concordancia entre revisores quando houver dupla revisao. Para evitar sobrepeso de sentencas revisadas por duas pessoas, as metricas de acuracia usam um consenso unico por `sentence_id`; a contagem bruta de labels aceitos e a contagem de sentencas unicas aceitas sao reportadas separadamente.

### Taxonomy Decision Rules V2.0.4

O baseline deterministico usa `rule_engine_version=taxonomy-rules-v2.0.4`. A taxonomia e aplicada como motor de regras em etapas: filtro institucional, deteccao de topicos candidatos, resolucao de prioridade, deteccao direcional de stance, guards de negacao/reversao, marcacao de fronteira de taxonomia e formula do score. A formula principal permanece:

```text
tone_level = stance_score * topic_weight * confidence * information_weight
```

`novelty_score`, `tone_change` e `communication_surprise_naive` continuam separados do nivel principal de tom.

Regras de decisao:

- `policy_decision`: decisao efetiva, voto ou mudanca concreta de Selic, compulsorio ou instrumento monetario. Exemplo positivo: "O Copom decidiu reduzir a taxa Selic...". Exemplo negativo: uma projecao condicionada a Selic constante.
- `forward_guidance`: proximos passos, estrategia futura, condicionalidade e sinalizacao da trajetoria de politica. Exemplo positivo: "ira monitorar... para definir os proximos passos". Exemplo negativo: frase factual de atividade com "continuidade" ou "trajetoria".
- `inflation_current`: inflacao observada, IPCA, nucleos, IPA, INCC, servicos, alimentos, administrados e precos correntes.
- `inflation_expectations`: Focus, mediana das expectativas, expectativas de mercado, projecoes de inflacao, metas futuras e ancoragem.
- `activity_growth`: producao, comercio, vendas, confianca, PIB, industria, demanda domestica, bens de capital e indicadores setoriais domesticos.
- `labor_market`: emprego, desemprego, renda, salarios, massa salarial, rendimento medio e pessoal ocupado.
- `external_environment`: EUA, Europa, China, Fed, comercio exterior, liquidez global, economia mundial e ambiente externo.
- `uncertainty`: incerteza como conceito dominante, nao uma palavra isolada em frase de outro topico.
- `fiscal_risk`: divida, resultado primario, premio fiscal, solvencia, politica fiscal e credibilidade do arcabouco.

Fronteiras legitimas sao marcadas em `taxonomy_boundary_flag`, nao tratadas automaticamente como bug do baseline: `policy_decision_vs_forward_guidance`, `current_inflation_vs_expectations`, `fiscal_risk_vs_external_environment`, `activity_growth_vs_labor_market` e `external_environment_vs_uncertainty`.

Benchmark permanente:

```powershell
copom-watch v2 benchmark-baseline
```

O benchmark separa sample 001, sample 002 Claude, sample 002 GPT, consenso sample 002 e consenso total. Ele aplica gates contra overfitting: nao aceitar melhora em sample 001 com piora no holdout, nao permitir queda relevante em informatividade e nao permitir deterioracao grande de F1 por stance quando houver benchmark anterior.
Antes de sobrescrever os arquivos "latest", o comando preserva uma copia do benchmark V2.0.3 em `outputs/v2/benchmarks/v2.0.3/`, usada como base fixa de comparacao.

Estado local de aceite desta rodada:

- Backfill V2: 258 reunioes, 491 documentos, 36.386 sentencas e 24.486 sentencas informativas.
- Redline V2: 441/441 pares documento-documento cobertos, com 54.814 linhas.
- Labels humanos: 900 labels aceitos em 600 sentencas unicas; a segunda amostra tem dupla revisao Claude/GPT.
- Auditoria baseline V2.0.4 vs consenso humano: stance accuracy 77,3%, topic accuracy 70,0%, informativeness accuracy 94,7% e F1 macro 76,9%.
- Concordancia humana holdout: Claude vs GPT concordaram em 89,0% no stance, 83,7% no topico e 99,0% em informatividade na amostra 002.
- Analise erro vs ambiguidade: dos 272 erros baseline vs consenso, 177 sao erro provavel do baseline, 68 sao ambiguidade legitima por conflito humano, 24 sao fronteiras de taxonomia e 3 sao erros com contexto de discordancia humana.
- Benchmark permanente: `copom-watch v2 benchmark-baseline` gera metricas por sample/holdout/topico/stance e aplica gates contra overfitting. Nesta rodada ficou em `pass`: `likely_baseline_error < 180`, `stance_f1_macro >= 70%`, `topic_accuracy >= 68%` e `informativeness_accuracy >= 91%`.
- Diagnostico de auditoria: `copom-watch v2 audit` gera erros priorizados e resumo agregado para orientar a proxima melhoria deterministica do baseline.
- Health-check: `warning`, com 6 warnings e 0 errors; os warnings refletem principalmente V2.0.2 ainda em validacao, mercado opcional ausente e limitacoes conhecidas de dados.
- Baseline V2.0.4: inclui `rule_engine_version=taxonomy-rules-v2.0.4`, filtros deterministico-auditaveis para cabecalhos/boilerplate, prioridades formais de taxonomia, `taxonomy_boundary_flag`, distincao entre decisao efetiva de Selic/compulsorio e forward guidance, e sinais direcionais de expectativas, atividade, emprego, inflacao, petroleo, commodities, cambio, credito, fiscal, premio de risco e guards adicionais para falsos neutros no holdout. LLM continua opcional e fora da verdade formal.

### Fechamento V2.0

O fechamento da V2.0 consolida tres entregas:

```powershell
copom-watch v2 review-remaining-errors --limit 177
copom-watch v2 train-supervised
copom-watch v2 freeze-release --version v2.0.4-holdout-stance-hardened
```

`review-remaining-errors` classifica os erros provaveis remanescentes para governanca metodologica, mas nao altera o lexico nem o motor de regras. `train-supervised` treina um benchmark experimental com labels humanos aceitos, usando sample 001 como treino inicial e sample 002 como holdout principal. `freeze-release` exige benchmark sem `fail` e health-check com 0 errors, gera manifesto com hashes e copia os artefatos principais para a pasta de release.

O indice oficial da V2.0 permanece o baseline deterministico `taxonomy-rules-v2.0.4`. O modelo supervisionado e apenas uma medicao experimental do potencial de ganho para uma versao futura.

## Testes

```powershell
pytest
python -m compileall src tests
ruff check src tests
mypy src --ignore-missing-imports
```

## Limitacoes conhecidas

- A inferencia econometrica e exploratoria: a janela de 24 meses tem poucas reunioes.
- A API OData do Focus pode retornar indisponibilidade temporaria; o cache mitiga apenas depois da primeira coleta bem-sucedida.
- Quando o OData Focus fica indisponivel, snapshots oficiais podem ser importados como fallback com `import-focus-snapshot`; PDFs devem ser convertidos para CSV/Excel antes da importacao.
- O baseline lexico e transparente, mas nao captura toda nuance contextual. O LLM opcional melhora interpretacao, mas introduz custo, dependencia externa e possivel drift.
- `communication_surprise` so e calculado quando ha observacoes suficientes para regressao simples; caso contrario permanece nulo.
- A revisao econometrica bloqueia regressao com Focus ausente, preditor sem variacao ou amostra insuficiente, em vez de reportar inferencia enganosa.
- Opcoes de COPOM da B3 entram apenas quando houver fonte historica publica estruturada ou CSV auditavel; sem isso, `decision_surprise_official` permanece ausente.
- Na V2, `communication_surprise_naive` mede apenas mudanca textual contra a reuniao anterior, nao surpresa de mercado.
- O indice V2 oficial usa calibracao fixa quando a janela tem observacoes suficientes; caso contrario o status de calibracao fica explicito.
- Mercado e semantica sao opcionais e nunca sao dependencia do pipeline textual principal.

## Proximos passos

- V2.1: ampliar dados publicos de mercado, melhorar proxies auditaveis de expectativa de decisao e rodar event study sem look-ahead, mantendo mercado como modulo opcional.
- V2.2: relatorio mais completo, tela de busca semantica no dashboard e deploy gratuito sem login.

## V2.2 UI, RAG local e deploy

A V2.2 transforma as camadas V2.0.4 e V2.1 em produto navegavel, sem alterar o indice textual oficial. A UI passa a ser V2-first, com abas para Latest COPOM, timeline, decomposicao, redline, evidencias, Focus, mercado, auditoria, Ask COPOM Watch, relatorios e Legacy V1.

RAG local:

```powershell
copom-watch semantic build-index --method tfidf
copom-watch semantic ask --query "expectativas desancoradas" --top-n 8
```

O `ask` e extrativo: ele sintetiza apenas as sentencas recuperadas e sempre mostra citacao com reuniao, documento e sentence_id. Sem citacao, nao ha resposta.

Pacote publico e aceite:

```powershell
copom-watch v2 package-public-data
copom-watch v2 v22-health
copom-watch v2 freeze-v22 --version v2.2-product-rag-deploy
```

`package-public-data` gera `app_data/copom_watch_public.duckdb` com tabelas reduzidas para o app, excluindo caches brutos, labels humanos completos e textos brutos volumosos. `v22-health` valida entrypoint Streamlit, requirements, config, pacote publico, semantic index, manifests V2.0/V2.1 e tabelas minimas. O deploy principal e Streamlit Community Cloud apontando para `streamlit_app.py`; Hugging Face Spaces pode usar o mesmo entrypoint como alternativa.

### Freeze V2.2

O fechamento formal da V2.2 congela a camada de produto e deploy:

```powershell
copom-watch v2 freeze-v22 --version v2.2-product-rag-deploy
```

O comando roda preflight local com `benchmark-baseline`, `health-check`, `v21-health` e `v22-health`, sem coletar novos dados. Ele gera `reports/v2/v22_release_manifest.json`, `reports/v2/v22_release_summary.html` e copia o pacote para `reports/v2/releases/v2.2-product-rag-deploy/`, incluindo `app_data/copom_watch_public.duckdb`, manifests, health reports, `semantic_ask_report.html`, `streamlit_app.py`, `requirements.txt` e `.streamlit/config.toml`.

O comando oficial local da V2.2 e:

```powershell
streamlit run streamlit_app.py --server.port 8502 --server.address localhost
```

Se a porta 8502 ja estiver ocupada por uma sessao antiga rodando `src/copom_tone_index/dashboard/app.py`, encerre essa sessao e reinicie pelo entrypoint acima. `src/copom_tone_index/dashboard/app.py` continua funcional para desenvolvimento, mas `streamlit_app.py` e o entrypoint de release e deploy.

Para Streamlit Community Cloud, publique o repositorio com `requirements.txt`, `.streamlit/config.toml`, `streamlit_app.py` e `app_data/copom_watch_public.duckdb`, e configure o app para iniciar por `streamlit_app.py`. A V2.2 nao altera a metodologia V2.0.4, nao usa LLM obrigatorio e nao transforma o event study em inferencia causal.

## V2.1 Mercado e Focus expandido

A V2.1 adiciona uma camada opcional sobre o core textual V2.0.4. O indice oficial continua sendo o baseline deterministico `taxonomy-rules-v2.0.4`; Focus e mercado entram apenas para diagnostico de expectativas, decisao, comunicacao e reacao em janela de evento.

Comandos principais:

```powershell
copom-watch v2 focus-refresh --months 400
copom-watch v2 focus-audit
copom-watch market fetch-public --sources bcb-sgs,ptax,anbima --months 400
copom-watch market derive-decision-expectations --method public
copom-watch market public-coverage
copom-watch market import-csv --path data\market\market_observations.csv --source user_csv --data-access-tier USER_CSV
copom-watch market import-decision-expectations --path data\market\decision_expectations.csv
copom-watch market event-study
copom-watch v2 build-event-panel
copom-watch v2 health-check
copom-watch v2 v21-health
copom-watch v2 freeze-v21 --version v2.1-public-focus-market-acceptance
```

Outputs V2.1:

- `outputs/v2/focus_vintages.csv`: tabela longa vintage-safe de Focus por data de divulgacao, indicador, horizonte e estatistica.
- `outputs/v2/focus_event_features.csv`: valores Focus pre-evento, primeiro pos-evento e segundo pos-evento, sem forward-fill.
- `outputs/v2/focus_v21_coverage.csv`: cobertura Focus por indicador, horizonte, estatistica e tipo de evento.
- `data/market/generated/market_observations_public.csv`: observacoes publicas geradas automaticamente por BCB SGS, PTAX e ANBIMA quando disponiveis.
- `data/market/generated/decision_expectations_public.csv`: expectativas/proxies publicos de decisao, com proxies marcados explicitamente.
- `outputs/v2/public_market_source_audit.csv`: fontes publicas tentadas, status, linhas e limitacoes.
- `outputs/v2/decision_expectation_source_audit.csv`: auditoria de expectativa oficial ausente/presente e proxies derivados.
- `outputs/v2/public_market_coverage.csv`: cobertura por fonte, ativo, janela e expectativa oficial/proxy.
- `outputs/v2/decision_expectations.csv`: expectativas de decisao importadas ou derivadas; proxies nunca preenchem `decision_surprise_official_bps`.
- `outputs/v2/market_event_study.csv`: janelas de mercado sem look-ahead.
- `outputs/v2/v21_event_panel.csv`: painel consolidado de decisao, comunicacao, Focus e mercado.
- `outputs/v2/v21_acceptance_by_meeting.csv`: cobertura de Focus, mercado e surpresa por reuniao.
- `outputs/v2/v21_acceptance_by_source.csv`: cobertura e status por fonte publica/proxy.
- `reports/v2/v21_release_manifest.json`: manifesto reproduzivel da release V2.1 congelada.
- `reports/v2/v21_release_summary.html`: resumo local da release V2.1 com hashes, warnings e limitacoes.
- `reports/v2/focus_v21_report.html`, `reports/v2/public_market_data_report.html`, `reports/v2/public_market_coverage_report.html`, `reports/v2/market_event_study_report.html`, `reports/v2/v21_macro_market_report.html` e `reports/v2/v21_acceptance_report.html`.

Regras metodologicas:

- `focus_pre_event` usa apenas a ultima observacao Focus antes do evento.
- `focus_post_event_1` e `focus_post_event_2` usam observacoes efetivamente publicadas depois do evento.
- `decision_surprise_official_bps` so e calculada quando houver expectativa oficial/publicamente observavel antes da reuniao.
- `decision_surprise_proxy_bps` pode ser derivada de Focus Selic ou curva curta publica, mas sempre com `is_proxy=true` e `decision_surprise_status=proxy`.
- A fonte B3 Opcao de Copom e auditada como preferencia oficial; se nao houver historico publico estruturado, a ausencia e warning, nao erro.
- Janelas de mercado usam `known_at_timestamp`; se o timing for ambiguo, a linha fica marcada como `ambiguous_event_timing`.
- Ausencia de mercado ou expectativas de decisao gera status explicito, nao erro do core.
- O event study e associativo/descritivo; nao deve ser reportado como inferencia causal.

### Fechamento V2.1

O aceite analitico da V2.1 roda em cima das tabelas ja geradas:

```powershell
copom-watch v2 v21-health
```

O comando gera `reports/v2/v21_acceptance_report.html`, `reports/v2/v21_acceptance_report.json`, `outputs/v2/v21_acceptance_by_meeting.csv` e `outputs/v2/v21_acceptance_by_source.csv`. Ele falha apenas quando ha problema que compromete a camada V2.1, como painel ausente, Focus event features ausentes, duplicatas logicas, look-ahead ou proxy preenchendo surpresa oficial. Ausencia de historico publico estruturado da B3 Opcao de Copom permanece como warning documentado; nesse caso, `decision_surprise_official_bps` fica nulo e apenas `decision_surprise_proxy_bps` pode ser preenchido.

Para congelar formalmente a V2.1, rode:

```powershell
copom-watch v2 freeze-v21 --version v2.1-public-focus-market-acceptance
```

O freeze da V2.1 nao faz nova coleta de Focus, SGS, PTAX, ANBIMA ou B3. Ele executa preflight local com `benchmark-baseline`, `health-check` e `v21-health`, valida a presenca dos artefatos centrais e copia o pacote para `reports/v2/releases/<version>/`. O manifesto registra `rule_engine_version=taxonomy-rules-v2.0.4`, cobertura de Focus, cobertura de mercado, proxies de decisao, ausencia de surpresa oficial B3 quando aplicavel e hashes SHA-256 dos artefatos copiados.

A diferenca para o freeze V2.0.4 e que `freeze-release` congela o indice textual oficial; `freeze-v21` congela a camada analitica adicional de Focus vintage-safe, mercado publico opcional, estudo de evento e surpresas proxy/oficiais. Warnings conhecidos como ausencia de historico publico estruturado da B3 Opcao de Copom, proxies exploratorios e cobertura parcial de mercado sao aceitaveis desde que os health-checks tenham 0 errors. A V2.1 continua sendo descritiva/associativa, nao causal.

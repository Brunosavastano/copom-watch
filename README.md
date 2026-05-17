# COPOM Tone Index

Aplicacao analitica para transformar comunicados e atas do COPOM em um indicador quantitativo, interpretavel e auditavel de tom de politica monetaria hawkish/dovish.

O projeto segue a SPEC em `SPEC_COPOM_Tone_Index.md` e entrega um MVP local com ingestao de dados publicos do Banco Central, scoring textual, revisoes Focus, dashboard e notas automaticas por reuniao.

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

Dashboard:

```powershell
streamlit run src/copom_tone_index/dashboard/app.py
```

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

Tabelas principais:

- `copom_meetings`: calendario, datas, Selic e janela operacional.
- `copom_documents`: comunicados/atas e textos limpos.
- `copom_sentences`: sentencas classificadas com topico, tom, score, confianca e evidencias.
- `copom_scores`: score consolidado por reuniao.
- `focus_revisions`: Focus pre e pos-evento.
- `focus_observations`: observacoes Focus auditaveis com fonte, data, mediana e assinatura de consulta.

## Testes

```powershell
pytest
python -m compileall src tests
```

## Limitacoes conhecidas

- A inferencia econometrica e exploratoria: a janela de 24 meses tem poucas reunioes.
- A API OData do Focus pode retornar indisponibilidade temporaria; o cache mitiga apenas depois da primeira coleta bem-sucedida.
- Quando o OData Focus fica indisponivel, snapshots oficiais podem ser importados como fallback com `import-focus-snapshot`; PDFs devem ser convertidos para CSV/Excel antes da importacao.
- O baseline lexico e transparente, mas nao captura toda nuance contextual. O LLM opcional melhora interpretacao, mas introduz custo, dependencia externa e possivel drift.
- `communication_surprise` so e calculado quando ha observacoes suficientes para regressao simples; caso contrario permanece nulo.
- A revisao econometrica bloqueia regressao com Focus ausente, preditor sem variacao ou amostra insuficiente, em vez de reportar inferencia enganosa.
- Mercado DI, opcoes de COPOM, CDS, cambio intradiario e apresentacao executiva ficam fora do MVP.

## Proximos passos

- Rotular manualmente uma amostra de sentencas e medir concordancia.
- Adicionar DI/curva de juros e estudo de evento de mercado.
- Treinar classificador supervisionado com os rotulos humanos.
- Criar relatorio metodologico mais formal com validacao econometrica e intervalos de incerteza.
- Publicar dashboard em ambiente controlado.

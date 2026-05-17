# SPEC — COPOM Tone Index

## Replicação interpretativa do COPOM Watch: leitura quantitativa de comunicados e atas

**Projeto:** Indicador Proprietário de Tom da Política Monetária do COPOM  
**Versão:** 1.0  
**Objetivo:** Transformar comunicados e atas do COPOM em um indicador quantitativo, interpretável e auditável de tom hawkish/dovish.  
**Aplicação:** Monitoramento e avaliação das decisões do COPOM.  

---

## 1. Visão executiva

O projeto consiste em criar uma ferramenta quantitativa para monitorar a comunicação do COPOM a partir de NLP, análise macroeconômica e validação com dados de mercado e expectativas.

A solução coleta todos os comunicados e atas do COPOM dos últimos 24 meses, classifica o tom da comunicação em uma escala **hawkish–dovish**, identifica os principais tópicos macroeconômicos e cruza o resultado com:

1. decisão da Selic;
2. variação da Selic versus reunião anterior;
3. revisões das expectativas Focus após o comunicado e/ou ata;
4. temas predominantes na comunicação oficial do Banco Central.

O produto final é um indicador proprietário de **tom da política monetária**, acompanhado de dashboard, evidências textuais e nota interpretativa por reunião.

A proposta diferencia-se por combinar:

- macroeconomia monetária;
- análise textual especializada;
- NLP aplicado a bancos centrais;
- avaliação de expectativas;
- interpretação econométrica de comunicação de política monetária.

---

## 2. Problema a resolver

O acompanhamento tradicional do COPOM tende a se concentrar na decisão da Selic e em leituras qualitativas dos comunicados. No entanto, a comunicação contém informação marginal importante, especialmente sobre:

- balanço de riscos;
- persistência da inflação;
- ancoragem das expectativas;
- riscos fiscais;
- cenário externo;
- atividade econômica;
- ritmo esperado de flexibilização ou aperto monetário;
- forward guidance.

O problema central é transformar essa informação textual em uma métrica objetiva e comparável ao longo do tempo.

Pergunta principal:

> O COPOM ficou mais hawkish, mais dovish ou apenas repetiu a função de reação já esperada pelo mercado?

---

## 3. Objetivos

### 3.1 Objetivo geral

Construir um indicador quantitativo e interpretável que mensure o tom da comunicação do COPOM por reunião.

### 3.2 Objetivos específicos

- Coletar comunicados e atas oficiais do COPOM dos últimos 24 meses.
- Associar cada reunião à respectiva decisão de Selic.
- Medir o tom hawkish/dovish dos textos.
- Identificar tópicos macroeconômicos relevantes.
- Medir a variação do tom entre reuniões.
- Cruzar o tom da comunicação com revisões das expectativas Focus.
- Gerar evidências textuais que expliquem o score.
- Produzir dashboard e notas automáticas por reunião.

---

## 4. Hipótese central

A comunicação do COPOM contém informação incremental além da decisão da Selic.

Essa informação aparece em frases sobre persistência inflacionária, expectativas desancoradas, incerteza, balanço de riscos, atividade econômica, cenário externo e forward guidance.

Hipótese empírica:

```text
ΔFocus_{t+h} = α + β1 Tone_t + β2 ΔTone_t + β3 SurpriseSelic_t + γX_t + ε_t
```

Onde:

- `Tone_t` é o tom hawkish/dovish da reunião `t`;
- `ΔTone_t` é a mudança de tom em relação à reunião anterior;
- `SurpriseSelic_t` é a diferença entre decisão efetiva e expectativa pré-COPOM;
- `ΔFocus_{t+h}` é a revisão pós-comunicado das expectativas Focus;
- `X_t` representa controles macroeconômicos e financeiros.

---

## 5. Escopo

### 5.1 Escopo do MVP

O MVP cobre uma janela operacional de **24 meses**, mas deve usar o histórico completo disponível para calibração, benchmarking e normalização.

Isso é importante porque, em 24 meses, haverá aproximadamente 16 reuniões do COPOM, uma amostra pequena para treinamento e validação estatística robusta.

O MVP deve entregar:

1. base de reuniões do COPOM;
2. textos de comunicados e atas;
3. decisão da Selic por reunião;
4. revisões Focus pré e pós-evento;
5. score hawkish/dovish por documento;
6. score consolidado por reunião;
7. decomposição por tópico;
8. evidências textuais;
9. dashboard;
10. nota interpretativa por reunião.

### 5.2 Fora do escopo do MVP

Não entram na primeira versão:

- previsão da próxima decisão do COPOM;
- recomendação de trading;
- modelo estrutural completo de regra de Taylor;
- análise intradiária de mercado;
- discursos individuais de diretores do BCB;
- fontes não oficiais como input principal.

Esses itens podem ser adicionados em versões futuras.

---

## 6. Fontes de dados

### 6.1 Documentos oficiais do COPOM

Fonte primária: Banco Central do Brasil, base de documentos do COPOM.

Recursos relevantes:

```text
GET /api/servico/sitebcb/copom/atas?quantidade=100
GET /api/servico/sitebcb/copom/atas_detalhes?nro_reuniao={nro_reuniao}
GET /api/servico/sitebcb/copom/comunicados?quantidade=100
GET /api/servico/sitebcb/copom/comunicados_detalhes?nro_reuniao={nro_reuniao}
```

Campos esperados:

```text
nro_reuniao
data_referencia
data_publicacao
titulo
url_pdf
texto
```

Fontes:

- https://dadosabertos.bcb.gov.br/dataset/atas-comunicados-copom
- https://www.bcb.gov.br/api/servico/sitebcb/copom/atas_detalhes?nro_reuniao=255

### 6.2 Selic

Fonte primária: SGS do Banco Central do Brasil.

Série recomendada:

```text
432 — Taxa de juros - Meta Selic definida pelo COPOM
```

Fonte:

- https://dadosabertos.bcb.gov.br/dataset/432-taxa-de-juros---meta-selic-definida-pelo-copom

### 6.3 Expectativas Focus

Fonte primária: API OData de Expectativas de Mercado do Banco Central.

Variáveis mínimas:

- Selic fim do ano corrente;
- Selic fim do ano seguinte;
- IPCA ano corrente;
- IPCA ano seguinte;
- IPCA 12 meses, se disponível;
- câmbio fim do ano, opcional;
- PIB, opcional.

Fontes:

- https://dadosabertos.bcb.gov.br/dataset/expectativas-mercado
- https://dadosabertos.bcb.gov.br/dataset/expectativas-mercado/resource/53b9ecd6-f148-488f-b884-4757542ad9f3

### 6.4 Mercado, opcional

Para versão avançada:

- DI 1 ano;
- DI 2 anos;
- inclinação da curva DI;
- opções de COPOM;
- câmbio BRL/USD;
- CDS Brasil;
- Treasuries norte-americanos;
- commodities relevantes.

Fonte potencial:

- https://www.b3.com.br/pt_br/produtos-e-servicos/negociacao/juros/dashboard-publico-opcoes-de-copom/

---

## 7. Unidade de análise

A unidade principal é a **reunião do COPOM**.

Cada reunião deve conter:

```text
meeting_id
nro_reuniao
data_referencia
data_publicacao_comunicado
data_publicacao_ata
titulo_comunicado
titulo_ata
texto_comunicado
texto_ata
selic_pre
selic_pos
delta_selic
focus_pre
focus_post_comunicado
focus_post_ata
delta_focus_selic
delta_focus_ipca
tone_comunicado
tone_ata
tone_total
delta_tone
topic_distribution
evidence_sentences
model_version
prompt_version
```

O comunicado é o texto de reação imediata. A ata é o texto interpretativo mais completo.

Para uso operacional:

```text
ToneNowcast_t = ToneComunicado_t
```

Após a publicação da ata:

```text
ToneFinal_t = 0.60 × ToneComunicado_t + 0.40 × ToneAta_t
```

---

## 8. Definição macroeconômica de hawkish e dovish

O projeto não deve usar sentimento genérico. Em política monetária, uma frase negativa sobre inflação pode ser hawkish, enquanto uma frase negativa sobre atividade pode ser dovish.

### 8.1 Hawkish

Classificar como hawkish quando o texto enfatizar:

- inflação persistente;
- núcleos elevados;
- expectativas desancoradas;
- risco fiscal pressionando expectativas;
- mercado de trabalho aquecido;
- hiato mais apertado;
- cenário externo adverso para inflação;
- câmbio depreciado ou commodities pressionando preços;
- necessidade de cautela;
- manutenção de juros elevados por mais tempo;
- menor probabilidade de flexibilização;
- forward guidance restritivo.

### 8.2 Dovish

Classificar como dovish quando o texto enfatizar:

- desinflação em curso;
- desaceleração de atividade;
- arrefecimento de crédito;
- ociosidade econômica;
- queda de commodities;
- apreciação cambial com efeito benigno;
- riscos baixistas para inflação;
- maior confiança na convergência da inflação;
- maior espaço para cortes;
- forward guidance expansionista ou benigno.

### 8.3 Neutro

Classificar como neutro quando a frase for factual, institucional, operacional ou sem implicação clara para o viés de política monetária.

---

## 9. Taxonomia de tópicos

Cada sentença deve ser classificada em um tópico principal.

Tópicos mínimos:

```text
inflation_current
inflation_expectations
activity_growth
labor_market
credit_conditions
fiscal_risk
external_environment
fx_commodities
risk_balance
forward_guidance
policy_decision
uncertainty
institutional
```

Descrição dos tópicos:

| Tópico | Descrição |
|---|---|
| `inflation_current` | Inflação corrente, núcleos, serviços, bens industriais, alimentos, administrados. |
| `inflation_expectations` | Expectativas de inflação, ancoragem, metas, projeções. |
| `activity_growth` | PIB, demanda, consumo, investimento, desaceleração ou aquecimento. |
| `labor_market` | Emprego, salários, mercado de trabalho, ociosidade. |
| `credit_conditions` | Crédito, inadimplência, concessões, condições financeiras domésticas. |
| `fiscal_risk` | Política fiscal, arcabouço fiscal, dívida, prêmio de risco. |
| `external_environment` | Fed, juros globais, China, crescimento mundial, aversão a risco. |
| `fx_commodities` | Câmbio, commodities, petróleo, alimentos, repasse cambial. |
| `risk_balance` | Balanço de riscos para inflação e atividade. |
| `forward_guidance` | Sinalização sobre próximos passos da política monetária. |
| `policy_decision` | Decisão de Selic e justificativa direta. |
| `uncertainty` | Grau de incerteza, cautela, dependência de dados. |
| `institutional` | Trechos formais, procedimentais ou sem conteúdo macro relevante. |

---

## 10. Metodologia NLP

### 10.1 Pipeline textual

Etapas:

1. baixar comunicado e ata;
2. remover HTML, notas de rodapé irrelevantes, lista de presentes e blocos operacionais;
3. segmentar por parágrafo e sentença;
4. classificar tópico por sentença;
5. classificar tom por sentença;
6. atribuir peso por tópico e seção;
7. agregar no nível do documento;
8. agregar no nível da reunião;
9. gerar evidências textuais e justificativas.

### 10.2 Trilha A — open-source

Objetivo: máxima replicabilidade.

Componentes possíveis:

- `sentence-transformers` para embeddings;
- embeddings multilíngues ou modelos em português;
- BERTopic para tópicos;
- classificador supervisionado leve;
- baseline léxico hawkish/dovish;
- `scikit-learn` para validação;
- `statsmodels` para regressões.

Vantagens:

- baixo custo;
- reprodutibilidade;
- independência de API proprietária;
- transparência metodológica.

Limitações:

- menor capacidade de interpretação contextual;
- maior necessidade de rótulos manuais;
- risco de classificação literalista.

### 10.3 Trilha B — Claude API / LLM

Objetivo: maior sofisticação semântica.

Características:

- classificação sentença a sentença;
- prompt estruturado;
- saída JSON;
- evidência textual obrigatória;
- score de confiança;
- versionamento de prompt e modelo.

Vantagens:

- melhor entendimento contextual;
- boa classificação de nuances de política monetária;
- capacidade de explicar a classificação;
- facilidade para gerar notas interpretativas.

Limitações:

- custo por chamada;
- dependência de API;
- possível drift de modelo;
- necessidade de auditoria e validação.

### 10.4 Estratégia recomendada

A melhor solução combina as duas trilhas:

```text
baseline open-source + classificação contextual via LLM + validação humana
```

O modelo open-source funciona como âncora reproduzível. O LLM adiciona interpretação macroeconômica granular.

---

## 11. Rubrica de classificação por sentença

Cada sentença deve receber saída estruturada:

```json
{
  "sentence_id": "255_comunicado_004",
  "text": "...",
  "topic": "inflation_expectations",
  "stance": "hawkish",
  "stance_score": 0.82,
  "confidence": 0.91,
  "direction": "higher_for_longer",
  "rationale": "A sentença enfatiza persistência inflacionária e desancoragem das expectativas.",
  "evidence_terms": ["persistência", "desancoradas", "cautela"]
}
```

Regras:

- `stance_score` deve variar de `-1` a `+1`;
- valores positivos indicam tom hawkish;
- valores negativos indicam tom dovish;
- valores próximos de zero indicam neutralidade;
- `confidence` deve variar de `0` a `1`;
- toda classificação hawkish ou dovish deve conter justificativa.

---

## 12. Fórmula do índice

### 12.1 Score por sentença

Para cada sentença `i`:

```text
s_i = P(Hawkish_i) - P(Dovish_i)
```

Interpretação:

```text
s_i > 0  → hawkish
s_i < 0  → dovish
s_i ≈ 0  → neutro
```

### 12.2 Score do documento

```text
DocTone_{d,t} = Σ(w_i × s_i) / Σ(w_i)
```

Onde `w_i` é o peso da sentença, derivado do tópico e da seção.

Pesos sugeridos por tópico:

| Tópico | Peso |
|---|---:|
| `policy_decision` | 1.50 |
| `forward_guidance` | 1.40 |
| `inflation_expectations` | 1.30 |
| `risk_balance` | 1.20 |
| `fiscal_risk` | 1.15 |
| `inflation_current` | 1.10 |
| `external_environment` | 1.00 |
| `activity_growth` | 0.95 |
| `labor_market` | 0.90 |
| `credit_conditions` | 0.85 |
| `institutional` | 0.20 |

### 12.3 Score da reunião

```text
ToneRaw_t = 0.60 × ToneComunicado_t + 0.40 × ToneAta_t
```

Antes da publicação da ata:

```text
ToneNowcast_t = ToneComunicado_t
```

### 12.4 Normalização

```text
ToneZ_t = (ToneRaw_t - μ_hist) / σ_hist
```

Escala final:

```text
COPOMToneIndex_t = 50 + 10 × ToneZ_t
```

Interpretação:

| Índice | Interpretação |
|---:|---|
| `> 60` | Claramente hawkish |
| `55–60` | Moderadamente hawkish |
| `45–55` | Neutro / balanceado |
| `40–45` | Moderadamente dovish |
| `< 40` | Claramente dovish |

### 12.5 Mudança de tom

```text
ΔTone_t = Tone_t - Tone_{t-1}
```

A variação do tom é tão importante quanto o nível absoluto, porque mercados tendem a reagir à surpresa de comunicação.

---

## 13. Separação entre decisão e comunicação

Uma queda da Selic não torna automaticamente a comunicação dovish.

Exemplo:

- corte de 25 bps com linguagem dura pode ser hawkish;
- manutenção com sinalização clara de cortes futuros pode ser dovish;
- alta de juros com indicação de fim de ciclo pode ser menos hawkish do que a decisão isolada sugere.

Portanto, o projeto deve separar três métricas.

### 13.1 Decision Score

Baseado apenas na decisão da Selic:

```text
alta da Selic       → hawkish
manutenção          → neutro, condicionado ao ciclo
queda da Selic      → dovish
```

### 13.2 Communication Score

Baseado apenas no texto do comunicado e da ata.

### 13.3 Communication Surprise Score

Resíduo da comunicação após controlar pela decisão:

```text
CommSurprise_t = ToneRaw_t - E[ToneRaw_t | ΔSelic_t, SelicPre_t, FocusPre_t]
```

Esse é o score mais sofisticado, pois responde:

> Dado o que o COPOM decidiu, o texto veio mais duro ou mais benigno do que o esperado?

---

## 14. Medição da revisão Focus pós-comunicado

Definir janelas:

```text
focus_pre:
última observação Focus disponível antes da data da reunião

focus_post_comunicado:
primeira observação Focus disponível após o comunicado

focus_post_ata:
primeira observação Focus disponível após a ata

Δfocus:
focus_post - focus_pre
```

Variáveis principais:

```text
Δfocus_selic_current_year
Δfocus_selic_next_year
Δfocus_ipca_current_year
Δfocus_ipca_next_year
```

Interpretação esperada:

- comunicação mais hawkish tende a elevar ou sustentar expectativas de Selic;
- comunicação mais dovish tende a reduzir expectativas de Selic;
- comunicação hawkish por inflação pode reduzir IPCA esperado se o mercado interpreta maior compromisso anti-inflacionário;
- comunicação hawkish por risco inflacionário pode elevar IPCA esperado se o mercado interpreta piora do cenário.

Esse ponto exige cuidado: o sinal sobre IPCA pode depender da decomposição entre reação de política monetária e diagnóstico de inflação.

---

## 15. Validação econométrica

### 15.1 Validação descritiva

Para cada reunião:

- comparar classificação do modelo com leitura humana;
- verificar se as sentenças explicativas fazem sentido;
- comparar `ΔTone_t` com mudança percebida de linguagem;
- auditar falsos positivos e falsos negativos.

Critério mínimo:

```text
80% de concordância entre leitura humana e classificação do modelo
em uma amostra manual de sentenças rotuladas.
```

### 15.2 Validação contra Focus

Regressão base:

```text
ΔFocusSelic_{t,h} = α + β1 Tone_t + β2 ΔTone_t + β3 ΔSelic_t + ε_t
```

Regressão expandida:

```text
ΔFocusSelic_{t,h} =
α + β1 CommSurprise_t
  + β2 DecisionSurprise_t
  + β3 ΔIPCAExp_t
  + β4 ΔUSRates_t
  + β5 ΔBRL_t
  + ε_t
```

### 15.3 Validação contra mercado, opcional

Adicionar:

- DI 1 ano;
- DI 2 anos;
- inclinação DI 1y–2y;
- câmbio;
- opções de COPOM;
- CDS;
- taxa de juros externa.

Possível regressão:

```text
ΔDI_{t,h} = α + β1 CommSurprise_t + β2 DecisionSurprise_t + γX_t + ε_t
```

### 15.4 Cuidados estatísticos

Com apenas 24 meses, a inferência formal será limitada.

Mitigações:

- usar histórico amplo para calibração;
- tratar os últimos 24 meses como janela operacional;
- reportar intervalos de confiança;
- usar bootstrap ou wild bootstrap;
- evitar claims causais fortes;
- apresentar o índice como ferramenta de monitoramento e interpretação.

---

## 16. Arquitetura técnica

### 16.1 Stack recomendada

```text
Python
pandas ou polars
requests ou httpx
beautifulsoup4
pydantic
duckdb ou postgres
scikit-learn
sentence-transformers
bertopic
statsmodels
plotly
streamlit ou quarto
prefect ou dagster
git / github actions
```

### 16.2 Estrutura do repositório

```text
copom-tone-index/
  README.md
  pyproject.toml
  config/
    settings.yaml
    topic_taxonomy.yaml
    hawkish_dovish_lexicon.yaml
  data/
    raw/
    interim/
    processed/
  notebooks/
    01_data_audit.ipynb
    02_text_features.ipynb
    03_validation.ipynb
  src/
    ingestion/
      fetch_copom.py
      fetch_focus.py
      fetch_selic.py
    preprocessing/
      clean_text.py
      segment_sentences.py
    nlp/
      lexicon_score.py
      llm_classifier.py
      topic_model.py
      aggregate_scores.py
    econometrics/
      event_study.py
      focus_revisions.py
    dashboard/
      app.py
    reporting/
      generate_meeting_note.py
  outputs/
    dashboard/
    reports/
    figures/
  tests/
    test_ingestion.py
    test_text_cleaning.py
    test_score_bounds.py
```

---

## 17. Modelo de dados

### 17.1 Tabela `copom_meetings`

```text
meeting_id
nro_reuniao
data_referencia
data_comunicado
data_ata
selic_pre
selic_pos
delta_selic
created_at
updated_at
```

### 17.2 Tabela `copom_documents`

```text
document_id
meeting_id
document_type
publication_date
title
url
raw_text
clean_text
model_version
prompt_version
```

### 17.3 Tabela `copom_sentences`

```text
sentence_id
document_id
sentence_order
text
topic
stance
stance_score
confidence
rationale
evidence_terms
```

### 17.4 Tabela `copom_scores`

```text
meeting_id
tone_comunicado
tone_ata
tone_raw
tone_z
copom_tone_index
delta_tone
communication_surprise
classification
```

### 17.5 Tabela `focus_revisions`

```text
meeting_id
variable
reference_year
focus_pre_date
focus_pre_value
focus_post_comunicado_date
focus_post_comunicado_value
focus_post_ata_date
focus_post_ata_value
delta_post_comunicado
delta_post_ata
```

---

## 18. Dashboard

O dashboard deve conter cinco blocos principais.

### 18.1 Visão geral

- índice atual;
- classificação do tom;
- variação versus reunião anterior;
- decisão da Selic;
- revisão Focus pós-comunicado;
- revisão Focus pós-ata.

### 18.2 Série temporal

Exibir:

- COPOM Tone Index;
- Selic;
- Focus Selic;
- Focus IPCA;
- `ΔTone_t`.

### 18.3 Decomposição por tópico

Heatmap:

```text
reunião × tópico × tom
```

Exemplo de leitura:

```text
Inflação: hawkish
Atividade: dovish
Fiscal: hawkish
Externo: neutro
Forward guidance: hawkish
```

### 18.4 Evidências textuais

Para cada reunião:

- top 5 frases hawkish;
- top 5 frases dovish;
- tópico associado;
- documento de origem;
- score;
- justificativa.

### 18.5 Pós-evento

Mostrar:

- revisão Focus Selic;
- revisão Focus IPCA;
- eventual alteração da curva DI;
- comentário interpretativo automático.

---

## 19. Output analítico por reunião

Modelo de nota:

```text
COPOM Tone Note — Reunião {nro_reuniao}

1. Decisão
   Selic: {selic_pre} → {selic_pos}
   Delta: {delta_selic}

2. Tom
   Índice: {score}
   Classificação: {hawkish/moderately hawkish/neutral/dovish}
   Mudança vs reunião anterior: {delta_tone}

3. Principais drivers
   - Inflação/expectativas:
   - Atividade:
   - Fiscal:
   - Externo:
   - Forward guidance:

4. Sentenças-chave
   - Hawkish:
   - Dovish:

5. Reação Focus
   - Selic ano corrente:
   - Selic ano seguinte:
   - IPCA ano corrente:
   - IPCA ano seguinte:

6. Interpretação
   O comunicado foi {mais/menos} restritivo do que a decisão isolada sugeriria.
```

---

## 20. Prompt para Claude API

```text
Você é um economista especializado em política monetária brasileira.

Classifique cada sentença do comunicado/ata do COPOM segundo:
1. tópico macroeconômico;
2. tom de política monetária;
3. intensidade do tom;
4. confiança;
5. evidência textual.

Definições:
- Hawkish: indica maior preocupação com inflação, expectativas desancoradas,
  riscos altistas, necessidade de juros altos por mais tempo ou menor espaço
  para flexibilização.
- Dovish: indica menor pressão inflacionária, desaceleração da atividade,
  riscos baixistas, maior confiança na desinflação ou maior espaço para cortes.
- Neutro: sentença factual sem implicação clara para o viés de política monetária.

Importante:
- Não use sentimento genérico.
- Uma frase negativa sobre inflação pode ser hawkish.
- Uma frase negativa sobre atividade pode ser dovish.
- Retorne apenas JSON válido.

Formato:
[
  {
    "sentence_id": "...",
    "topic": "...",
    "stance": "hawkish|dovish|neutral",
    "stance_score": -1.0 to 1.0,
    "confidence": 0.0 to 1.0,
    "rationale": "...",
    "evidence_terms": ["..."]
  }
]
```

---

## 21. Critérios de aceite

O MVP será considerado concluído quando:

```text
1. Baixar automaticamente todos os comunicados e atas dos últimos 24 meses.
2. Associar cada reunião à decisão da Selic.
3. Associar cada reunião às revisões Focus pré e pós-comunicado.
4. Gerar score hawkish/dovish por comunicado, ata e reunião.
5. Gerar decomposição por tópico.
6. Exibir pelo menos 5 sentenças explicativas por reunião.
7. Produzir dashboard funcional.
8. Produzir nota interpretativa por reunião.
9. Manter outputs versionados.
10. Reproduzir resultados com o mesmo código e mesma versão de modelo/prompt.
```

---

## 22. Roadmap

### Semana 1 — Dados e baseline

- coletar documentos do COPOM;
- coletar Selic SGS 432;
- coletar Focus;
- montar tabela de eventos;
- limpar textos;
- criar baseline léxico hawkish/dovish.

### Semana 2 — NLP e scoring

- segmentar sentenças;
- classificar tópicos;
- rodar modelo open-source;
- testar Claude API;
- consolidar score por documento e reunião;
- criar evidências textuais.

### Semana 3 — Validação e dashboard

- calcular revisões Focus;
- rodar regressões simples;
- construir dashboard;
- gerar notas automáticas;
- documentar metodologia.

### Versão avançada — 4 a 6 semanas

- expandir histórico completo;
- adicionar curva DI;
- adicionar B3/opções de COPOM;
- treinar classificador supervisionado;
- criar score residual de surpresa de comunicação;
- publicar relatório metodológico.

---

## 23. Principais riscos

### 23.1 Amostra pequena

Últimos 24 meses são bons para monitoramento, mas ruins para estimação.

Mitigação:

- calibrar no histórico completo;
- publicar janela operacional de 24 meses;
- tratar regressões como validação exploratória.

### 23.2 Sentimento genérico errado

Modelos genéricos podem classificar “inflação persistente” como sentimento negativo sem entender que, em política monetária, isso indica tom hawkish.

Mitigação:

- usar rubrica macroeconômica;
- classificar por tópico antes do tom;
- validar com leitura humana.

### 23.3 Revisão Focus contaminada

As expectativas Focus podem mudar por dados de inflação, fiscal, câmbio ou cenário externo divulgados no mesmo período.

Mitigação:

- usar janelas curtas;
- adicionar controles;
- interpretar os resultados com cautela.

### 23.4 Drift de LLM

Mudanças de modelo ou prompt podem alterar scores.

Mitigação:

- versionar prompt;
- versionar modelo;
- manter temperatura baixa;
- armazenar outputs;
- manter baseline open-source.

### 23.5 Overfitting narrativo

O modelo pode explicar demais poucos eventos.

Mitigação:

- evidência textual obrigatória;
- métricas simples;
- comparação com leitura humana;
- evitar linguagem causal forte.

---

## 24. Entregáveis finais

```text
1. Dashboard Streamlit ou Quarto.
2. Base tratada por reunião do COPOM.
3. COPOM Tone Index.
4. Decomposição por tópico.
5. Notas automáticas por reunião.
6. Relatório metodológico.
7. Repositório GitHub documentado.
8. Notebook de validação econométrica.
9. Dataset exportável em CSV/Parquet.
10. Apresentação executiva de 5 slides.
```

---

## 25. Como posicionar o projeto em entrevista ou case

Formulação sugerida:

> Desenvolvi uma ferramenta de monitoramento quantitativo do COPOM que transforma atas e comunicados em um índice hawkish/dovish interpretável, cruzando NLP, decisão de Selic e revisões do Focus. A ideia é separar a decisão formal da comunicação marginal do Banco Central, identificando quando o texto veio mais duro ou mais benigno do que a decisão isolada sugeriria.

Essa formulação conecta diretamente:

```text
monitoramento de política monetária
+
macroeconomia aplicada
+
NLP
+
data science
+
interpretação de mercado
+
entregável institucional
```

O diferencial real está em não fazer apenas “sentiment analysis”, mas uma **função de reação textual do COPOM**.

---

## 26. Referências e fontes úteis

- Banco Central do Brasil — Atas e Comunicados do COPOM: https://dadosabertos.bcb.gov.br/dataset/atas-comunicados-copom
- Banco Central do Brasil — Exemplo de detalhe de ata: https://www.bcb.gov.br/api/servico/sitebcb/copom/atas_detalhes?nro_reuniao=255
- Banco Central do Brasil — Meta Selic definida pelo COPOM, SGS 432: https://dadosabertos.bcb.gov.br/dataset/432-taxa-de-juros---meta-selic-definida-pelo-copom
- Banco Central do Brasil — Expectativas de Mercado Focus: https://dadosabertos.bcb.gov.br/dataset/expectativas-mercado
- Banco Central do Brasil — API OData Expectativas: https://dadosabertos.bcb.gov.br/dataset/expectativas-mercado/resource/53b9ecd6-f148-488f-b884-4757542ad9f3
- B3 — Dashboard público de opções de COPOM: https://www.b3.com.br/pt_br/produtos-e-servicos/negociacao/juros/dashboard-publico-opcoes-de-copom/
- BIS — Central bank language models: https://www.bis.org/publ/work1215.pdf
- Paper sobre eventos do COPOM, Focus e features textuais: https://arxiv.org/abs/2604.11926

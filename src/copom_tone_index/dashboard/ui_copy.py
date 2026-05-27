from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PageCopy:
    title: str
    question: str
    how_to_read: tuple[str, ...]
    limitations: tuple[str, ...]
    glossary_terms: tuple[str, ...]


APP_POSITIONING = (
    "O COPOM Watch ajuda a entender como a comunicação do Banco Central mudou, "
    "quais temas explicam essa mudança e quais evidências textuais sustentam a leitura."
)

METHODOLOGY_OVERVIEW = (
    "O app lê comunicados e atas oficiais do Copom, separa o texto em sentenças, "
    "classifica cada sentença por tema e direção econômica, e agrega esse material "
    "em indicadores comparáveis entre reuniões. A leitura é descritiva: ela organiza "
    "evidências textuais, expectativas e reações de mercado, mas não prevê a próxima "
    "Selic, não estima causalidade e não substitui julgamento econômico."
)

GLOSSARY = {
    "Tom bruto": (
        "Média direcional das sentenças informativas antes da normalização. "
        "Valores positivos indicam comunicação mais restritiva; valores negativos, mais expansionista."
    ),
    "Índice de tom": (
        "Versão normalizada do tom para comparar reuniões ao longo do tempo. "
        "Ele resume o texto oficial, mas deve ser lido junto com as frases e os temas que explicam o número."
    ),
    "Surpresa textual": (
        "Diferença entre o tom da reunião atual e o tom da reunião anterior. "
        "É uma medida de mudança marginal da comunicação, não uma surpresa de mercado."
    ),
    "Intensidade": (
        "Parcela do documento com conteúdo direcional relevante. "
        "Ajuda a distinguir um texto fortemente direcional de um texto majoritariamente neutro."
    ),
    "Calibração": (
        "Janela fixa usada para transformar o tom em escala comparável. "
        "Isso evita que o histórico mude retroativamente sempre que novas reuniões entram na base."
    ),
    "Subíndices": (
        "Decomposições do tom por temas macroeconômicos, como inflação, expectativas, atividade, fiscal e cenário externo."
    ),
    "Mudança textual": (
        "Comparação entre o documento atual e o documento equivalente da reunião anterior. "
        "Mostra trechos adicionados, removidos, mantidos ou com mudança de intensidade."
    ),
    "Frases-chave": (
        "Sentenças oficiais que mais ajudam a explicar a leitura do índice. "
        "São a ponte entre o número agregado e a evidência textual."
    ),
    "Focus": (
        "Expectativas de mercado coletadas pelo Banco Central. "
        "Aqui são usadas para medir revisões observáveis antes e depois dos eventos do Copom."
    ),
    "Reação de mercado": (
        "Movimentos de ativos em janelas de evento. "
        "A leitura é associativa e não deve ser interpretada como prova causal."
    ),
    "Auditoria": (
        "Resumo das validações, rótulos humanos, métricas de classificação e limitações conhecidas do método."
    ),
    "Busca com evidências": (
        "Busca local em documentos oficiais do Copom. "
        "A resposta só pode usar trechos recuperados e deve citar reunião, documento e sentença."
    ),
    "Período analisado": (
        "A janela operacional recente facilita a navegação no período mais usado no dia a dia. "
        "O histórico completo mostra toda a base disponível."
    ),
}

PAGE_COPY = {
    "latest": PageCopy(
        title="Última reunião",
        question=(
            "Esta página resume a reunião selecionada: direção do tom, mudança em relação à reunião anterior, "
            "temas dominantes e frases que sustentam a leitura."
        ),
        how_to_read=(
            "Comece pelo índice de tom e pela surpresa textual para entender nível e mudança.",
            "Use os subíndices para ver quais temas explicam a leitura agregada.",
            "Leia as frases-chave antes de tirar conclusão econômica forte.",
        ),
        limitations=(
            "A leitura é textual e descritiva; não prevê a próxima Selic.",
            "Números agregados podem esconder frases ambíguas, por isso a evidência textual é parte essencial da interpretação.",
        ),
        glossary_terms=("Tom bruto", "Índice de tom", "Surpresa textual", "Intensidade", "Calibração", "Subíndices", "Frases-chave"),
    ),
    "timeline": PageCopy(
        title="Evolução do tom",
        question="Esta página mostra como o índice de tom evoluiu no histórico de reuniões do Copom.",
        how_to_read=(
            "Compare movimentos persistentes, não apenas uma observação isolada.",
            "Use a tabela para localizar reuniões específicas e mudanças textuais relevantes.",
            "Interprete o nível do índice junto com o ciclo monetário e o contexto macro da época.",
        ),
        limitations=(
            "O índice mede comunicação oficial, não a decisão em si.",
            "Mudanças de estilo na redação do Banco Central podem afetar comparações muito longas.",
        ),
        glossary_terms=("Índice de tom", "Surpresa textual", "Intensidade", "Calibração"),
    ),
    "decomposition": PageCopy(
        title="Decomposição por temas",
        question="Esta página responde quais temas puxaram a comunicação para uma leitura mais restritiva ou expansionista.",
        how_to_read=(
            "Compare os subíndices para separar inflação, expectativas, atividade, fiscal e cenário externo.",
            "Um índice agregado estável pode esconder mudanças importantes de composição.",
            "Use a contagem de sentenças para avaliar se a leitura de um tema é ampla ou concentrada.",
        ),
        limitations=(
            "Uma frase pode conter mais de um tema, mas a visualização resume a classificação principal.",
            "Subíndices com poucas sentenças devem ser lidos com cautela.",
        ),
        glossary_terms=("Subíndices", "Tom bruto", "Índice de tom"),
    ),
    "text_changes": PageCopy(
        title="Mudanças no texto",
        question="Esta página mostra o que mudou em relação ao documento equivalente da reunião anterior.",
        how_to_read=(
            "Trechos adicionados indicam novos pontos de atenção.",
            "Trechos removidos mostram sinais que deixaram de aparecer.",
            "Mudanças de tom indicam frases parecidas com intensidade diferente.",
        ),
        limitations=(
            "Mudança textual não é automaticamente mudança de política monetária.",
            "Reescritas podem refletir estilo editorial, não apenas alteração econômica.",
        ),
        glossary_terms=("Mudança textual", "Surpresa textual"),
    ),
    "evidence": PageCopy(
        title="Frases-chave",
        question="Esta página lista as sentenças oficiais que sustentam a leitura do índice.",
        how_to_read=(
            "Filtre por documento, tema e sinal para auditar a classificação.",
            "Use as citações para voltar ao texto oficial da reunião.",
            "Priorize frases com maior tom e confiança, mas revise o contexto quando houver ambiguidade.",
        ),
        limitations=(
            "A seleção destaca evidências fortes, mas não substitui leitura integral do comunicado ou da ata.",
            "Frases ambíguas podem exigir julgamento humano.",
        ),
        glossary_terms=("Frases-chave", "Índice de tom", "Subíndices"),
    ),
    "focus": PageCopy(
        title="Expectativas Focus",
        question="Esta página mostra revisões de expectativas antes e depois dos eventos do Copom.",
        how_to_read=(
            "Compare valores pré-evento com a primeira e a segunda observação pós-evento.",
            "Deltas nulos ou ausentes não são imputados; eles indicam falta de observação disponível.",
            "Separe IPCA, Selic, PIB e câmbio por horizonte.",
        ),
        limitations=(
            "Focus mede expectativas declaradas, não preços de mercado em tempo real.",
            "A revisão pós-evento pode refletir outros choques ocorridos na mesma janela.",
        ),
        glossary_terms=("Focus",),
    ),
    "market": PageCopy(
        title="Reação de mercado",
        question="Esta página organiza movimentos de ativos em janelas ao redor de comunicado e ata.",
        how_to_read=(
            "Use as janelas de evento para observar associação temporal com a comunicação.",
            "Verifique o status da janela antes de interpretar a reação.",
            "Compare ativos e vértices quando houver cobertura suficiente.",
        ),
        limitations=(
            "A leitura não é causal; outros eventos podem ter movido os preços.",
            "A camada de mercado é opcional e depende da disponibilidade de dados públicos ou importados.",
        ),
        glossary_terms=("Reação de mercado",),
    ),
    "audit": PageCopy(
        title="Auditoria",
        question="Esta página mostra a qualidade metodológica da classificação textual e suas limitações.",
        how_to_read=(
            "Use métricas de acurácia e F1 para avaliar desempenho da classificação.",
            "Leia os detalhes de rótulos humanos quando disponíveis.",
            "Trate alertas como limitações metodológicas, não como falha automática do app.",
        ),
        limitations=(
            "Sem rótulos humanos suficientes, a auditoria formal fica limitada.",
            "Acurácia textual não equivale a validação causal macroeconômica.",
        ),
        glossary_terms=("Auditoria",),
    ),
    "ask": PageCopy(
        title="Perguntas com evidências oficiais",
        question="Esta página permite consultar documentos históricos do Copom com respostas baseadas apenas em citações recuperadas.",
        how_to_read=(
            "Faça perguntas sobre linguagem, temas e casos históricos.",
            "Toda resposta deve citar reunião, documento e sentença.",
            "Use os exemplos para começar com consultas dentro do escopo.",
        ),
        limitations=(
            "A busca não prevê a próxima Selic e não responde com opinião fora dos documentos recuperados.",
            "Quando não há citações suficientes, a resposta deve recusar ou orientar uma busca alternativa.",
        ),
        glossary_terms=("Busca com evidências", "Frases-chave"),
    ),
    "reports": PageCopy(
        title="Relatórios",
        question="Esta página reúne relatórios e manifestos gerados pelo pipeline analítico.",
        how_to_read=(
            "Use os relatórios como material de auditoria, aceite ou documentação metodológica.",
            "Os manifestos registram versões, hashes e status dos dados processados.",
        ),
        limitations=(
            "Relatórios refletem o estado do pipeline no momento da geração.",
            "Arquivos ausentes indicam que a etapa correspondente ainda não foi executada neste ambiente.",
        ),
        glossary_terms=("Auditoria",),
    ),
    "legacy": PageCopy(
        title="Visão clássica",
        question="Esta página mantém a leitura original do projeto para compatibilidade e comparação.",
        how_to_read=(
            "Use esta seção apenas quando precisar comparar com a versão metodológica anterior.",
            "A interface principal deve ser usada para análise corrente.",
        ),
        limitations=(
            "A visão clássica tem menos explicações, menos decomposição e menor cobertura metodológica.",
        ),
        glossary_terms=("Índice de tom",),
    ),
}

ASK_EXAMPLES = (
    "Quando o Copom falou de expectativas desancoradas?",
    "Quais frases indicaram comunicação mais restritiva?",
    "Mostre menções a risco fiscal em reuniões recentes.",
    "Compare a linguagem sobre atividade econômica em ciclos de corte.",
)

OUT_OF_SCOPE_FORECAST_MESSAGE = (
    "O COPOM Watch não prevê a próxima Selic nem produz recomendação de decisão. "
    "Ele pode buscar casos históricos semelhantes e mostrar evidências textuais oficiais "
    "sobre comunicação, expectativas e reação de mercado."
)

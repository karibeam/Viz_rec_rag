# Recomendador de Visualizações via RAG

MVP de um sistema que recomenda gráficos para **não-especialistas**, a partir de uma
pergunta em linguagem do dia a dia. A recomendação sai como **PRINCIPAL + ALTERNATIVA**,
cada uma com spec **Vega-Lite** renderizável e as **fontes** que a sustentam.

Baseado em **RAG, não em regras**: não existe nenhuma tabela do tipo
"dado categórico + numérico → barras". Toda recomendação vem de trechos recuperados
por similaridade de cosseno de uma base de estudos empíricos de percepção gráfica
([Zeng et al., CHI 2023](https://github.com/Zehua-Zeng/graphical-perception-knowledge)).

---

## O problema que este projeto resolve

A base original tem 59 artigos, 465 designs e 240 achados — mas descritos em notação
de artigo acadêmico:

```json
{"data-type": "quantitative", "channel": "positionY", "scale": "linear"}
```

A pergunta de um usuário leigo — *"quero comparar vendas de 5 categorias"* — não tem
**nenhuma** sobreposição semântica com isso. Não importa a qualidade do modelo de
embedding: os dois lados vivem em regiões diferentes do espaço vetorial, e a
similaridade de cosseno fica baixa.

**Traduzir a base para português não resolve** — o problema não é idioma, é forma.

A solução é um passo de **enriquecimento** (`src/enrich.py`), rodado uma única vez
offline: um LLM reescreve cada achado técnico em um card narrativo que inclui o nome
popular do gráfico, a tarefa analítica, e — o detalhe que mais importa —
**3 exemplos de perguntas do jeito que um leigo perguntaria**. Isso planta o
vocabulário do usuário dentro do documento indexado, aproximando os dois vetores.

```
ANTES (indexado cru):          "channel": "positionY", "mark": "area-rect"
DEPOIS (card enriquecido):     "Barras lado a lado permitiram ordenar valores com
                                mais acerto que barras empilhadas. Indicado para:
                                comparar um valor numérico entre várias categorias.
                                Responde a perguntas como: Qual gráfico usar para
                                comparar vendas de vários produtos?"
```

---

## Pipeline

**Fase A — preparação da base (offline, roda 1×)**

```
59 JSONs  →  serialize.py     (JSON → texto legível, determinístico)
          →  enrich.py        (LLM → card narrativo + metadados)   ← resolve a similaridade
          →  build_index.py   (embeddings → Chroma, espaço cosseno)
```

**Fase B — runtime (por pergunta)**

```
pergunta  →  embedding        (mesmo modelo da indexação)
          →  busca por cosseno no Chroma → top-k chunks
          →  recommend.py     (LLM → JSON: principal + alternativa + Vega-Lite)
          →  validate_spec.py (valida e repara a spec)
          →  resposta + fontes citadas + conflito declarado
```

### Por que o HyDE está desligado

O HyDE (gerar uma resposta hipotética e buscar com ela) parecia uma boa ideia e
foi medido. Resultado, nas 18 perguntas de avaliação, 2 execuções cada:

| | HyDE (temp 0.3) | HyDE (temp 0.0) | **sem HyDE** |
|---|---|---|---|
| recomendação principal idêntica | 53% | 50% | **89%** |
| sobreposição de fontes | 0.42 | 0.61 | **0.76** |

Três motivos pelos quais ele atrapalha aqui:

1. **Dilui a pergunta.** O texto hipotético tem ~450 caracteres contra ~50 da
   pergunta: 90% do vetor de busca passa a ser texto gerado, não o que a pessoa
   perguntou.
2. **Pré-decide a resposta.** O parágrafo gerado já diz "a melhor escolha é o
   gráfico de barras" — e é esse texto que consulta a base. A recuperação passa a
   procurar evidência para um palpite prévio, em vez de deixar a base decidir.
   Isso conflita com a premissa do projeto (RAG, não regras): o conhecimento
   próprio do LLM vaza para dentro da etapa de recuperação.
3. **Responde com confiança a perguntas fora de escopo.** Nas perguntas-armadilha,
   sem HyDE o sistema recusou corretamente e de forma estável; com HyDE, "qual
   biblioteca JavaScript usar?" chegou a receber "gráfico de barras".

Desligá-lo ainda **corta pela metade o consumo de cota** (1 chamada por pergunta
em vez de 2). O problema de similaridade que o HyDE tentaria resolver já foi
resolvido no enriquecimento dos cards — que é onde ele deve ser resolvido.

Para reativar e medir no seu próprio conjunto de perguntas:

```bash
.venv/bin/python tests/stability_test.py --runs 2 --hyde
```

Quando ligado, o texto hipotético é cacheado em `data/processed/hyde_cache.json`
para que a recuperação seja reproduzível entre execuções.

Chunking é **semântico**: 1 chunk = 1 achado do artigo (não corte por número de
caracteres). Artigos sem ranking comparativo viram um card de contexto.

---

## Como rodar

### 1. Instalar

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

### 2. Configurar a chave

```bash
cp .env.example .env
```

Edite `.env` e preencha **uma** das opções — os dois provedores são caminhos
alternativos, não complementares (`LLM_PROVIDER` escolhe qual usar):

- **Gemini** (recomendado — free tier cobre este volume): pegue a chave em
  https://aistudio.google.com/apikey e preencha `GOOGLE_API_KEY`.
- **OpenAI**: preencha `OPENAI_API_KEY` e mude `LLM_PROVIDER=openai`.

**Uma única chave cobre chat e embeddings** — não é preciso gerar chaves separadas.

> Os modelos disponíveis mudam com o tempo (`gemini-2.5-flash` e
> `text-embedding-004`, por exemplo, já não aceitam chaves novas). Se aparecer
> um erro `404 NOT_FOUND`, liste o que sua chave enxerga e ajuste o `.env`:
>
> ```bash
> curl -s "https://generativelanguage.googleapis.com/v1beta/models?key=SUA_CHAVE" | grep '"name"'
> ```

### Cotas do free tier — leia antes de escolher o modelo

As cotas variam **muito** entre modelos, e a diferença decide se o projeto é
utilizável. Medido nesta chave:

| modelo | cota (free tier) | serve para |
|---|---|---|
| `gemini-flash-latest` | **20 requisições/dia** | inviável — 10 perguntas/dia no app |
| `gemini-3.5-flash-lite` | centenas/dia | é o padrão do projeto |
| `gemini-embedding-001` | 100/minuto | indexação (o `build_index.py` já faz throttling) |

Cada pergunta do usuário custa **2 chamadas** (HyDE + geração). Se aparecer
`RESOURCE_EXHAUSTED` com `GenerateRequestsPerDayPerProjectPerModel`, é cota
diária estourada — esperar não resolve dentro do dia, troque de modelo.

### 3. Enriquecer a base (uma vez, ~75 chamadas ao LLM)

```bash
.venv/bin/python src/enrich.py --limit 3
```

Rode com `--limit 3` primeiro e **leia** as fichas geradas em
`data/processed/cards.jsonl`. É o passo mais importante do sistema — se os cards
estiverem genéricos, todo o resto degrada. Ajuste o prompt em `src/enrich.py` se
necessário e rode `--force` para regerar.

Satisfeita, rode a base inteira:

```bash
.venv/bin/python src/enrich.py
```

O script **retoma de onde parou** — se estourar quota, é só rodar de novo.
Se atingir limite de requisições por minuto, reduza a concorrência: `--workers 2`.

### 4. Indexar

```bash
.venv/bin/python src/build_index.py
```

### 5. Testar no terminal (mais rápido para depurar)

```bash
.venv/bin/python src/recommend.py "quero comparar vendas de 5 categorias de produto"
```

### 6. Rodar a interface

```bash
.venv/bin/streamlit run app.py
```

---

## Desempate entre estudos que discordam

Os trechos recuperados vêm de estudos independentes, que compararam conjuntos
diferentes de alternativas em contextos diferentes. Para a tarefa "comparar
valores", a base aponta **34 gráficos distintos** como vencedores. Sem critério,
a escolha entre eles seria arbitrária.

O `recommend.py` aplica um critério explícito, **nesta ordem**:

1. **Correspondência de tarefa analítica** — um estudo sobre "encontrar o maior
   valor" não sustenta recomendação sobre "ver correlação", por mais parecido
   que o texto seja.
2. **Significância estatística** (`significancia=sim`) sobre `nao-reportada`.
3. **Evidência experimental** sobre teórica.
4. **Número de alternativas comparadas** — quem testou mais opções.
5. **Similaridade** — deliberadamente o critério **mais fraco**.

A similaridade vem por último de propósito: ela mede parecença textual com a
pergunta, não força da evidência. E na prática as similaridades ficam a
0.007–0.022 umas das outras — tratar essa diferença como decisiva seria escolher
aleatoriamente com passos extras.

Esses sinais são extraídos **deterministicamente do JSON original** (não pelo
LLM) via `serialize.finding_signals()`, e entram no contexto do prompt. A resposta
traz os campos `conflito` e `criterio_desempate`, então a decisão é auditável em
vez de implícita.

```bash
.venv/bin/python src/backfill_signals.py   # preenche os sinais em cards já gerados
```

## Avaliação

`tests/test_questions.md` traz 15 perguntas de não-especialista + 3 perguntas-armadilha
(fora do escopo da base, onde o sistema **deve** admitir a limitação no campo `ressalva`
em vez de inventar), e uma tabela de "sintoma → onde mexer".

O smoke test verifica o encanamento inteiro sem gastar uma chamada de API:

```bash
.venv/bin/python tests/smoke_offline.py
```

### Teste de estabilidade

Roda cada pergunta N vezes e mede se a recomendação se repete. É o dado que
decide se o **placar determinístico** (agregar os vencedores em Python antes de
gerar) é necessário ou se o desempate no prompt basta:

```bash
.venv/bin/python tests/stability_test.py --runs 2
```

Instabilidade tem duas fontes: o HyDE gera um texto novo a cada busca (mudando
o que é recuperado) e a geração tem temperatura > 0. Para isolar a segunda,
rode com `--no-hyde`.

---

## Estrutura

```
├── data/raw/              59 JSONs originais da base
├── data/processed/        cards.jsonl (saída do enriquecimento)
├── data/chroma/           índice vetorial persistido
├── src/
│   ├── config.py          provedor (Gemini/OpenAI) e caminhos
│   ├── serialize.py       JSON → texto legível (determinístico, sem interpretação)
│   ├── enrich.py          LLM: achado técnico → card narrativo
│   ├── build_index.py     embeddings → Chroma
│   ├── retrieve.py        busca por cosseno + HyDE
│   ├── recommend.py       geração da recomendação (também roda como CLI)
│   └── validate_spec.py   validação da spec Vega-Lite
├── app.py                 interface Streamlit
└── tests/
    ├── smoke_offline.py   pipeline completo com modelos falsos
    └── test_questions.md  avaliação manual
```

---

## Onde a divisão "RAG vs. regras" foi mantida

`serialize.py` faz um mapeamento determinístico, mas **apenas de formatação**:
transforma o JSON em texto legível sem decidir nada. Quem interpreta
(`area-rect + positionX nominal` → "gráfico de barras") é o LLM, e quem escolhe o
que recomendar é a recuperação por similaridade sobre a base. Não há em lugar nenhum
uma regra do tipo "se o dado é X, então use o gráfico Y".

---

## Próximos passos (pós-MVP)

- **Re-ranking** (cross-encoder) sobre o top-k, se a precisão ainda faltar
- **Filtro por metadados** — os cards já trazem `tarefas` e `tipos_de_dado` indexados;
  falta expor isso como filtro na busca (`Retriever.search(filtro=...)` já aceita)
- **Avaliação automatizada** com RAGAS (faithfulness / context relevance)
- Aceitar **upload de CSV** para gerar a spec já com os dados reais do usuário

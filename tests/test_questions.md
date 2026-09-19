# Perguntas de avaliação (não-especialista)

Rode cada uma e anote o resultado. O objetivo é medir **duas coisas separadas**:

1. **Recuperação** — os trechos recuperados são relevantes? (olhe a similaridade e o conteúdo)
2. **Geração** — a recomendação faz sentido e está ancorada nos trechos?

Um erro de recuperação e um erro de geração pedem correções diferentes: o primeiro
se ataca na tradução da pergunta (`PROMPT_TRADUCAO`), o segundo no prompt de
recomendação (`PROMPT_RECOMENDACAO`), ambos em `src/recomendar.py`.

```bash
.venv/bin/python src/recomendar.py "sua pergunta aqui"
```

Para comparar a versão atual com a v1 (cards + placar) nestas perguntas:

```bash
.venv/bin/python tests/comparar_v1_v2.py
```

| # | Pergunta | Esperado (aprox.) | Recuperação OK? | Recomendação OK? |
|---|---|---|---|---|
| 1 | Quero comparar as vendas de 5 categorias de produto | barras | | |
| 2 | Como mostrar a evolução da temperatura ao longo de 12 meses? | linhas | | |
| 3 | Preciso ver se há relação entre horas de estudo e nota da prova | dispersão | | |
| 4 | Qual gráfico usar para mostrar a participação de cada região no total? | barras ou pizza | | |
| 5 | Quero identificar valores fora do padrão numa lista de preços | dispersão/boxplot | | |
| 6 | Tenho 3 lojas e 12 meses de faturamento, como mostro tudo junto? | linhas múltiplas / facetas | | |
| 7 | Como comparar duas fatias de um gráfico de pizza? | barras (pizza é pior) | | |
| 8 | Quero mostrar a distribuição das idades dos meus clientes | histograma | | |
| 9 | Preciso destacar qual vendedor teve o maior resultado | barras ordenadas | | |
| 10 | Como mostro a média de vendas por trimestre? | barras | | |
| 11 | Tenho dados de temperatura por hora e por dia da semana | mapa de calor | | |
| 12 | Quero comparar o desempenho de 20 produtos diferentes | barras horizontais | | |
| 13 | Como agrupo clientes parecidos visualmente? | dispersão com cor | | |
| 14 | Qual a melhor forma de mostrar percentuais que somam 100%? | barras empilhadas / barras | | |
| 15 | Preciso ler o valor exato de cada mês no gráfico | barras com rótulo / tabela | | |

## Perguntas-armadilha (fora do escopo da base)

O sistema **deve** recusar estas perguntas: a tradução classifica a tarefa como
`nenhuma` e nenhuma busca é feita.

| # | Pergunta | Comportamento esperado |
|---|---|---|
| 16 | Qual biblioteca JavaScript devo usar para meus gráficos? | ressalva — fora do escopo |
| 17 | Como mostro uma rede de conexões entre pessoas? | ressalva — a base tem pouco sobre grafos |
| 18 | Qual a cor da capa do meu relatório? | ressalva — nada a ver com escolha de gráfico |

> A pergunta "como faço um mapa coroplético?" **não** é armadilha: a base cobre
> mapas e cartogramas (ex. `Nusrat2018evaluating`, `Golebiowska2020rainbow`).
> Uma resposta fundamentada ali é comportamento correto, não alucinação.

## O que ajustar conforme o resultado

| Sintoma | Onde mexer (`src/recomendar.py`) |
|---|---|
| A tarefa identificada está errada | `PROMPT_TRADUCAO` — descrições das 10 tarefas |
| Tarefa certa, mas achados de outras tarefas | `consulta_tecnica` — formato da consulta |
| Pergunta legítima recusada (tarefa `nenhuma`) | `PROMPT_TRADUCAO` — descrição de `nenhuma` |
| Recomendação contradiz o achado citado | `PROMPT_RECOMENDACAO` |
| Spec Vega-Lite inválida com frequência | `PROMPT_RECOMENDACAO` — regras da spec |

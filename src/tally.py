"""Placar deterministico: decide a recomendacao em Python, nao no LLM.

Por que existe
--------------
A recuperacao e deterministica (mesmos k trechos sempre), mas quando os trechos
apontam graficos diferentes o LLM escolhia sozinho -- e a escolha nao se repetia
entre execucoes. Aqui a escolha vira uma contagem explicita: cada trecho vota no
grafico que venceu no seu estudo, com peso derivado da forca daquela evidencia.
O LLM passa a justificar e desenhar a spec, nao a decidir.

Isto NAO e um sistema de regras: nao existe nenhum mapeamento
"tipo de dado -> grafico". Os votos, os pesos e os candidatos vem inteiramente
dos trechos recuperados da base. Trocar a base muda o resultado.

Ordem de prioridade (a mesma do prompt, agora executavel e auditavel):
  1. tarefa analitica compatível com a pergunta
  2. diferenca estatisticamente significativa
  3. evidencia experimental (nao teorica)
  4. numero de alternativas comparadas
  5. similaridade -- deliberadamente o fator mais fraco
"""

import json
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
import serialize as S

# Pesos multiplicativos. Sao parametros de projeto, nao fatos da base --
# estao aqui, explicitos, para poderem ser discutidos e ajustados.
PESO_TAREFA = 1.60          # criterio 1: o mais forte
PESO_SIGNIFICANCIA = 1.30   # criterio 2
PESO_EXPERIMENTAL = 1.15    # criterio 3
PESO_ALTERNATIVAS = 0.02    # criterio 4: +2% por alternativa comparada (limitado)
MAX_ALTERNATIVAS = 10
PENALIDADE_EVITAR = 0.50    # voto contrario de quem ficou em ultimo

# Descricoes em portugues das 10 tarefas canonicas da base. Servem para casar a
# pergunta do usuario com uma tarefa por similaridade -- nao para recomendar.
DESCRICAO_TAREFAS = {
    "aggregate": "estimar uma media, um total ou um valor agregado de um conjunto de dados",
    "characterize-distribution": "entender como os valores se distribuem, a forma da distribuicao",
    "cluster": "identificar grupos ou agrupamentos de itens parecidos entre si",
    "correlate": "ver a relacao ou correlacao entre duas variaveis",
    "determine-range": "determinar o intervalo, o minimo e o maximo dos valores",
    "filter": "filtrar ou selecionar itens que atendem a um criterio",
    "find-anomalies": "encontrar valores fora do padrao, anomalias ou outliers",
    "find-extremum": "encontrar o maior ou o menor valor do conjunto",
    "retrieve-value": "ler o valor exato de um item especifico",
    "sort": "ordenar ou comparar valores entre categorias",
}

CACHE_TAREFAS = config.PROCESSED_DIR / "task_embeddings.json"

# Piso de relevancia: abaixo disto, a pergunta nao corresponde a nenhuma das 10
# tarefas analiticas da base e o sistema recusa em vez de recomendar.
#
# Calibrado nas 18 perguntas de tests/test_questions.md:
#   legitimas  0.6591 .. 0.8116
#   armadilhas 0.6104 .. 0.6620
# Os intervalos SE SOBREPOEM -- nenhum limiar separa os dois grupos sem erro.
# (A similaridade dos proprios trechos separa ainda pior: ~0.61-0.74 em ambos.)
#
# Escolha: 0.655, o maior valor que nao recusa nenhuma pergunta legitima.
# Consequencia assumida: casos limitrofes passam (ex. "como mostro uma rede de
# conexoes?", 0.6620 -- e uma pergunta de visualizacao, so que sobre um tipo de
# grafico que a base cobre mal). Para esses, a defesa e o campo "ressalva" do
# LLM, nao este piso. Preferimos deixar passar a recusar quem merecia resposta.
LIMIAR_TAREFA = 0.655


def normalizar(nome: str) -> str:
    """Agrupa nomes que diferem so por acento, caixa ou pontuacao.

    Junta "gráfico de dispersão" e "grafico de dispersao" (o mesmo grafico,
    escrito de dois jeitos pelo LLM). NAO junta "barras" com "barras
    agrupadas" -- sao graficos diferentes, e fundi-los perderia informacao.
    """
    t = unicodedata.normalize("NFKD", (nome or "").lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = re.sub(r"\bgraficos?\b|\bde\b|\bem\b|\bcom\b", " ", t)
    return re.sub(r"[^a-z0-9]+", " ", t).strip()


def _embeddings_tarefas():
    """Embeddings das descricoes de tarefa, calculados uma vez e cacheados."""
    if CACHE_TAREFAS.exists():
        try:
            return json.loads(CACHE_TAREFAS.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    emb = config.get_embeddings()
    chaves = sorted(DESCRICAO_TAREFAS)
    vetores = emb.embed_documents([DESCRICAO_TAREFAS[k] for k in chaves])
    mapa = dict(zip(chaves, vetores))
    CACHE_TAREFAS.parent.mkdir(parents=True, exist_ok=True)
    CACHE_TAREFAS.write_text(json.dumps(mapa), encoding="utf-8")
    return mapa


def _cosseno(a, b):
    num = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return num / (na * nb) if na and nb else 0.0


def tarefa_da_pergunta(pergunta: str):
    """Qual das 10 tarefas canonicas a pergunta mais se aproxima.

    Por similaridade de embedding com as descricoes -- deterministico, e o mesmo
    mecanismo usado na busca. Retorna (tarefa, similaridade).
    """
    mapa = _embeddings_tarefas()
    vetor = config.get_embeddings().embed_query(pergunta)
    pares = sorted(
        ((t, _cosseno(vetor, v)) for t, v in mapa.items()), key=lambda p: (-p[1], p[0])
    )
    return pares[0]


def peso_do_trecho(metadata, similaridade, tarefa_alvo):
    """Peso do voto de um trecho, com a conta aberta para auditoria."""
    fatores = {"similaridade": round(float(similaridade), 4)}
    peso = float(similaridade)

    if tarefa_alvo and metadata.get("tarefa_canonica") == tarefa_alvo:
        peso *= PESO_TAREFA
        fatores["tarefa_compativel"] = PESO_TAREFA
    if metadata.get("significancia") == "sim":
        peso *= PESO_SIGNIFICANCIA
        fatores["significancia"] = PESO_SIGNIFICANCIA
    if metadata.get("evidencia") == "experimental":
        peso *= PESO_EXPERIMENTAL
        fatores["experimental"] = PESO_EXPERIMENTAL

    n = min(int(metadata.get("n_designs") or 0), MAX_ALTERNATIVAS)
    if n:
        bonus = 1 + n * PESO_ALTERNATIVAS
        peso *= bonus
        fatores["n_alternativas"] = round(bonus, 3)

    return peso, fatores


def montar_placar(pergunta: str, hits, tarefa_alvo=None):
    """Conta os votos dos trechos recuperados e devolve o ranking de graficos.

    hits: [(Document, similaridade)] -- o conjunto ja recuperado, fixo.
    Retorna dict com o ranking e a memoria de calculo.
    """
    if tarefa_alvo is None:
        tarefa_alvo, sim_tarefa = tarefa_da_pergunta(pergunta)
    else:
        _, sim_tarefa = tarefa_da_pergunta(pergunta)

    # Sem piso de relevancia o placar sempre elege um vencedor, mesmo para
    # perguntas que nada tem a ver com escolha de grafico -- medimos o sistema
    # respondendo "grafico de dispersao" para "qual biblioteca JavaScript usar?".
    fora_de_escopo = sim_tarefa < LIMIAR_TAREFA

    candidatos = {}
    votos_por_artigo = {}
    for i, (doc, similaridade) in enumerate(hits, 1):
        m = doc.metadata
        peso, fatores = peso_do_trecho(m, similaridade, tarefa_alvo)

        # Amortecimento por artigo: um mesmo estudo costuma aparecer varias vezes
        # (condicoes diferentes do MESMO experimento -- ex. 4, 6 e 8 categorias).
        # Sem isso ele vota 3x enquanto estudos independentes votam 1x, e um unico
        # artigo decide o placar. O n-esimo achado do mesmo artigo vale 1/n.
        artigo = m.get("source_file", "?")
        ordem = votos_por_artigo.get(artigo, 0) + 1
        votos_por_artigo[artigo] = ordem
        if ordem > 1:
            peso /= ordem
            fatores["repeticao_do_artigo"] = f"1/{ordem}"

        nome = (m.get("recomendado") or "").strip()
        if nome:
            chave = normalizar(nome)
            c = candidatos.setdefault(
                chave, {"nome": nome, "pontos": 0.0, "a_favor": [], "contra": []}
            )
            c["pontos"] += peso
            c["a_favor"].append({"trecho": i, "peso": round(peso, 4), "fatores": fatores})

        evitar = (m.get("evitar") or "").strip()
        if evitar:
            chave = normalizar(evitar)
            c = candidatos.setdefault(
                chave, {"nome": evitar, "pontos": 0.0, "a_favor": [], "contra": []}
            )
            c["pontos"] -= peso * PENALIDADE_EVITAR
            c["contra"].append({"trecho": i, "peso": round(-peso * PENALIDADE_EVITAR, 4)})

    # ordenacao estavel: pontos desc, depois nome -- garante reprodutibilidade
    ranking = sorted(
        ({"chave": k, **v} for k, v in candidatos.items()),
        key=lambda c: (-round(c["pontos"], 6), c["chave"]),
    )
    for pos, c in enumerate(ranking, 1):
        c["posicao"] = pos
        c["pontos"] = round(c["pontos"], 4)

    return {
        "tarefa_identificada": tarefa_alvo,
        "similaridade_tarefa": round(sim_tarefa, 4),
        "fora_de_escopo": fora_de_escopo,
        "limiar_tarefa": LIMIAR_TAREFA,
        "ranking": ranking,
        # Só o 1o colocado vira recomendacao; o resto do ranking permanece
        # exposto para o usuario auditar como a decisao foi tomada.
        "vencedor": ranking[0] if ranking else None,
    }


def formatar_placar(placar) -> str:
    """Bloco de texto do placar para entrar no prompt de geracao."""
    tarefa = placar.get("tarefa_identificada") or "(nao identificada)"
    desc = DESCRICAO_TAREFAS.get(tarefa, "")
    linhas = [
        f"TAREFA ANALITICA IDENTIFICADA NA PERGUNTA: {tarefa}"
        + (f" ({desc})" if desc else ""),
        "",
        "PLACAR CALCULADO A PARTIR DOS TRECHOS (ja decidido, nao recalcule):",
    ]
    for c in placar["ranking"][:6]:
        favor = ", ".join(str(a["trecho"]) for a in c["a_favor"]) or "-"
        contra = ", ".join(str(a["trecho"]) for a in c["contra"]) or "-"
        linhas.append(
            f"  {c['posicao']}. {c['nome']}  ({c['pontos']} pontos"
            f" | a favor: trechos {favor} | contra: trechos {contra})"
        )
    return "\n".join(linhas)


if __name__ == "__main__":
    from retrieve import Retriever

    pergunta = " ".join(sys.argv[1:]) or "quero comparar as vendas de 5 categorias"
    hits = Retriever(k=9, use_hyde=False).search(pergunta)
    placar = montar_placar(pergunta, hits)
    print(f"PERGUNTA: {pergunta}\n")
    print(formatar_placar(placar))
    print("\nmemoria de calculo do 1o colocado:")
    print(json.dumps(placar["ranking"][0], ensure_ascii=False, indent=2))

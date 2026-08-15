"""Recuperacao dos trechos mais relevantes (similaridade de cosseno).

Inclui HyDE opcional: antes de buscar, o LLM escreve uma resposta hipotetica
para a pergunta. O vetor de busca passa a ser o dessa resposta hipotetica, que
tem vocabulario mais proximo dos documentos do que a pergunta crua do usuario.
"""

import json
import sys
import threading

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

import config
from build_index import get_store

HYDE_PROMPT = """Uma pessoa sem experiencia em visualizacao de dados perguntou:

"{pergunta}"

Escreva um paragrafo curto (3-4 frases), em portugues, como se fosse um trecho de
um guia de visualizacao de dados respondendo a essa pergunta: diga qual grafico
usar, com que tipos de dado ele funciona e para qual tarefa analitica ele serve.
Nao use listas nem titulos. Este texto sera usado apenas como chave de busca."""


# O texto hipotetico e cacheado por pergunta: sem isso, cada busca gera um texto
# novo, a chave de busca muda e a mesma pergunta recupera trechos diferentes.
# O cache vai para disco porque a reprodutibilidade precisa valer TAMBEM entre
# execucoes distintas -- e o que permite repetir um experimento e obter o mesmo
# resultado. Apague o arquivo para forcar a regeracao.
CACHE_PATH = config.PROCESSED_DIR / "hyde_cache.json"
_CACHE_HYDE = {}
_cache_lock = threading.Lock()

if CACHE_PATH.exists():
    try:
        _CACHE_HYDE = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        _CACHE_HYDE = {}


def _salvar_cache():
    with _cache_lock:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        CACHE_PATH.write_text(
            json.dumps(_CACHE_HYDE, ensure_ascii=False, indent=1), encoding="utf-8"
        )


class Retriever:
    """Recupera trechos por similaridade de cosseno.

    HyDE vem DESLIGADO por padrao. Medimos: com HyDE a mesma pergunta produz a
    mesma recomendacao em 50-53% das vezes; sem ele, 89%. O texto hipotetico
    dilui a pergunta (vira ~90% da chave de busca), ja nomeia um grafico antes
    de consultar a base -- enviesando a recuperacao rumo ao proprio palpite --
    e responde com confianca ate a perguntas fora de escopo. Ligue apenas se
    medir ganho real no seu conjunto de perguntas.
    """

    def __init__(self, k: int = 6, use_hyde: bool = False):
        self.k = k
        self.use_hyde = use_hyde
        self.store = get_store()
        self._llm = None

    def _hyde(self, pergunta: str) -> str:
        if pergunta in _CACHE_HYDE:
            return _CACHE_HYDE[pergunta]
        if self._llm is None:
            # temperatura 0: o HyDE existe para aproximar vocabulario, nao para
            # variar. Com temperatura alta a recuperacao deixa de ser reproduzivel.
            self._llm = config.get_chat(temperature=0.0)
        resp = config.invoke_with_retry(self._llm, HYDE_PROMPT.format(pergunta=pergunta))
        # A pergunta original permanece na chave de busca para nao perder o
        # contexto especifico do usuario (dominio, numero de categorias, etc).
        texto = f"{pergunta}\n\n{config.text_of(resp).strip()}"
        _CACHE_HYDE[pergunta] = texto
        _salvar_cache()
        return texto

    def search_com_tarefa(self, pergunta: str, tarefa: str, k: int = None):
        """Busca normal + reforco de trechos da tarefa analitica identificada.

        Sem isso, uma pergunta sobre comparar categorias pode recuperar apenas
        achados rotulados com outra tarefa, e o criterio de desempate mais forte
        (compatibilidade de tarefa) nunca chega a ser aplicado -- medimos 0/6 em
        um dos casos. A busca aberta continua valendo: os trechos da tarefa sao
        acrescentados, nao substituem os demais.
        """
        k = k or self.k
        base = self.search(pergunta, k=k)
        if not tarefa:
            return base

        # Vagas RESERVADAS: trechos da tarefa costumam ter similaridade um pouco
        # menor, entao apenas juntar e reordenar por score os descartaria de novo.
        reserva = max(1, k // 3)
        ja_na_tarefa = [p for p in base if p[0].metadata.get("tarefa_canonica") == tarefa]
        if len(ja_na_tarefa) >= reserva:
            return base

        extras = self.search(pergunta, k=reserva, filtro={"tarefa_canonica": tarefa})
        vistos = {d.metadata.get("finding_id") for d, _ in base}
        novos = [(d, s) for d, s in extras if d.metadata.get("finding_id") not in vistos]
        if not novos:
            return base

        faltam = reserva - len(ja_na_tarefa)
        # descarta os piores da busca aberta para abrir espaco, preservando a ordem
        mantidos = base[: k - min(faltam, len(novos))]
        return sorted(mantidos + novos[:faltam], key=lambda p: -p[1])

    def search(self, pergunta: str, k: int = None, filtro: dict = None):
        """Retorna [(Document, similaridade_de_cosseno)], do mais ao menos relevante.

        Usamos a distancia bruta do Chroma e convertemos aqui: na colecao com
        espaco "cosine", distancia = 1 - similaridade. O helper pronto do
        LangChain assume score em [0, 1] e reclama quando a similaridade e
        negativa (vetores em direcoes opostas), o que e legitimo acontecer.
        """
        query = self._hyde(pergunta) if self.use_hyde else pergunta
        hits = self.store.similarity_search_with_score(query, k=k or self.k, filter=filtro)
        return [(doc, 1.0 - float(dist)) for doc, dist in hits]


def format_context(hits) -> str:
    """Monta o bloco de contexto do prompt de geracao.

    Alem do texto do card, expoe os sinais de forca de evidencia -- e o que
    permite ao modelo aplicar um criterio de desempate explicito quando os
    trechos discordam, em vez de escolher por conta propria.
    """
    blocos = []
    for i, (doc, score) in enumerate(hits, 1):
        m = doc.metadata
        ref = f"{m.get('source_file', '?')}#{m.get('finding_id', '?')}"
        sinais = (
            f"evidencia={m.get('evidencia', '?')}"
            f" | metrica={m.get('metrica', '?') or 'n/a'}"
            f" | diferenca estatisticamente significativa={m.get('significancia', '?')}"
            f" | alternativas comparadas={m.get('n_designs', '?')}"
            f" | houve empate no ranking={'sim' if m.get('tem_empate') else 'nao'}"
        )
        blocos.append(
            f"[TRECHO {i}] (fonte: {ref} | similaridade: {score:.3f})\n"
            f"  sinais: {sinais}\n{doc.page_content}"
        )
    return "\n\n".join(blocos)


if __name__ == "__main__":
    pergunta = " ".join(sys.argv[1:]) or "Como comparar vendas de 5 categorias de produto?"
    r = Retriever()
    for doc, score in r.search(pergunta):
        m = doc.metadata
        print(f"{score:.3f}  {m.get('source_file')}#{m.get('finding_id')}")
        print(f"        {m.get('titulo') or doc.page_content[:100]}")

"""ETAPA OFFLINE (roda uma vez): base de artigos -> textos -> embeddings -> Chroma.

Cada ACHADO de um artigo (um resultado experimental: "para a tarefa X, o
grafico A foi melhor que B") vira um documento de texto. O texto e montado
por um template fixo, sem LLM: a base fica intacta, so muda de formato.

    python src/indexar.py
"""

import json
import re
import sys
import time

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

import config
from langchain_chroma import Chroma
from langchain_core.documents import Document

TAREFAS = (
    "aggregate", "characterize-distribution", "cluster", "correlate", "determine-range",
    "filter", "find-anomalies", "find-extremum", "retrieve-value", "sort",
)


# --- pequenos ajudantes para ler o formato da base -------------------------

def tarefa_do_achado(nome: str) -> str:
    """Os achados sao nomeados pela tarefa: 'sort-2' -> 'sort'."""
    nome = re.sub(r"[-_ ]?\d+$", "", nome).replace(" ", "-")
    return nome if nome in TAREFAS else "general"


def achatar(ranking):
    """Todos os ids de design de um ranking, qualquer que seja o aninhamento."""
    if isinstance(ranking, list):
        return [x for item in ranking for x in achatar(item)]
    return [ranking]


def ranking_em_texto(ranking, nivel=0):
    """[['E-1','E-2'], 'E-3'] -> '(E-1 = E-2) > E-3'  (empates unidos por '=')."""
    while nivel == 0 and isinstance(ranking, list) and len(ranking) == 1 and isinstance(ranking[0], list):
        ranking = ranking[0]
    if not isinstance(ranking, list):
        return str(ranking)
    sep = " > " if nivel % 2 == 0 else " = "
    partes = []
    for item in ranking:
        t = ranking_em_texto(item, nivel + 1)
        partes.append(f"({t})" if isinstance(item, list) and len(item) > 1 else t)
    return sep.join(partes)


def design_em_texto(design_id, design):
    """Uma linha por design: marca e como cada tipo de dado foi codificado."""
    camadas = []
    for camada in design.get("layers") or []:
        codificacoes = [
            f"{e.get('data-type')} -> {e.get('channel')}" for e in camada.get("encodings") or []
        ]
        camadas.append(f"mark {camada.get('mark') or '?'}; " + "; ".join(codificacoes))
    linha = f"- {design_id}: " + " || ".join(camadas)
    if design.get("note"):
        linha += f' (note: "{design["note"]}")'
    return linha


# --- o template: um achado -> um documento ---------------------------------

def documentos_do_artigo(arquivo, artigo):
    designs = artigo.get("Designs") or {}
    for grupo, metricas in (artigo.get("Results") or {}).items():
        for metrica, achados in (metricas or {}).items():
            for nome, achado in (achados or {}).items():
                ids = [i for i in achatar(achado.get("rank")) if i in designs]
                tipos = sorted({
                    re.sub(r"-\d+$", "", e.get("data-type") or "")
                    for i in ids for c in designs[i].get("layers") or []
                    for e in c.get("encodings") or []
                } - {""})
                sig = achado.get("significance")  # as vezes vem como lista vazia
                significativo = isinstance(sig, dict) and bool(sig.get("pairs"))
                tarefa = tarefa_do_achado(nome)

                texto = "\n".join([
                    f"task: {tarefa}",
                    f"data types: {', '.join(tipos)}",
                    f"metric: {metrica} ({grupo.lower()})",
                    f"study: {artigo.get('Title', '')}",
                    f"result, best to worst: {ranking_em_texto(achado.get('rank'))}",
                    f"statistically significant differences: {'yes' if significativo else 'not reported'}",
                    "designs compared:",
                    *[design_em_texto(i, designs[i]) for i in ids],
                ])
                yield Document(
                    page_content=texto,
                    metadata={"fonte": f"{arquivo}#{grupo}/{metrica}/{nome}", "tarefa": tarefa},
                )


def todos_os_documentos():
    docs = []
    for caminho in sorted(config.RAW_DIR.glob("*.json")):
        artigo = json.loads(caminho.read_text(encoding="utf-8"))
        docs.extend(documentos_do_artigo(caminho.name, artigo))
    return docs


def abrir_banco(embeddings=None):
    return Chroma(
        collection_name=config.COLLECTION,
        embedding_function=embeddings or config.get_embeddings(),
        persist_directory=str(config.CHROMA_DIR),
        collection_metadata={"hnsw:space": "cosine"},  # similaridade de cosseno
    )


def main():
    docs = todos_os_documentos()
    print(f"{len(docs)} achados encontrados na base")

    banco = abrir_banco()
    ja_indexados = set(banco.get(include=[])["ids"])
    faltam = [d for d in docs if d.metadata["fonte"] not in ja_indexados]
    print(f"{len(ja_indexados)} ja indexados, {len(faltam)} a indexar")

    # O plano gratuito aceita ~100 embeddings por minuto: lotes de 40 com pausa.
    for i in range(0, len(faltam), 40):
        lote = faltam[i : i + 40]
        config.com_retentativa(
            lambda: banco.add_documents(lote, ids=[d.metadata["fonte"] for d in lote])
        )
        print(f"  {min(i + 40, len(faltam))}/{len(faltam)}")
        if i + 40 < len(faltam):
            time.sleep(60)
    print("pronto.")


if __name__ == "__main__":
    main()

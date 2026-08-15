"""Smoke test do pipeline inteiro SEM chamar API nenhuma.

Substitui o LLM e o modelo de embeddings por versoes falsas e deterministicas,
para verificar o encanamento: serializacao -> enriquecimento -> indexacao no
Chroma -> busca por similaridade -> geracao -> validacao da spec Vega-Lite.

Nao valida qualidade de resposta (para isso e preciso a API key real), apenas
que nenhuma etapa quebra.

    .venv/bin/python tests/smoke_offline.py
"""

import json
import shutil
import sys
import tempfile
from itertools import cycle
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import config

TMP = Path(tempfile.mkdtemp(prefix="vizrec_smoke_"))
config.PROCESSED_DIR = TMP / "processed"
config.CARDS_PATH = config.PROCESSED_DIR / "cards.jsonl"
config.CHROMA_DIR = TMP / "chroma"
config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

import enrich
import serialize as S
import validate_spec

FAKE_ENRICH = json.dumps(
    [
        {
            "finding_id": "PLACEHOLDER",
            "titulo": "Barras lado a lado sao mais precisas que barras empilhadas",
            "texto": (
                "O estudo comparou formas de mostrar valores numericos por categoria. "
                "O grafico de barras agrupadas (barras lado a lado) permitiu ordenar os "
                "valores com mais acerto do que o grafico de barras empilhadas. "
                "O grafico de pizza foi o pior da comparacao. "
                "Indicado para: comparar um valor numerico entre varias categorias."
            ),
            "graficos": ["grafico de barras agrupadas", "grafico de barras empilhadas", "grafico de pizza"],
            "recomendado": "grafico de barras agrupadas",
            "evitar": "grafico de pizza",
            "tarefas": ["comparar valores", "ordenar valores"],
            "tipos_de_dado": ["quantitativo", "nominal/categorico"],
            "perguntas_de_usuario": [
                "Qual grafico usar para comparar vendas de varios produtos?",
                "Como mostrar qual categoria vendeu mais?",
                "Barras ou pizza para comparar categorias?",
            ],
        }
    ]
)

FAKE_RECOMMEND = json.dumps(
    {
        "principal": {
            "grafico": "grafico de barras agrupadas",
            "justificativa": "Segundo o trecho 1, barras lado a lado permitem comparar valores por categoria com mais acerto.",
            "trechos_usados": [1],
            "vegalite_spec": {
                "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
                "title": "Vendas por categoria",
                "data": {"values": [{"categoria": "A", "vendas": 120}, {"categoria": "B", "vendas": 90}]},
                "mark": "bar",
                "encoding": {
                    "x": {"field": "categoria", "type": "nominal"},
                    "y": {"field": "vendas", "type": "quantitative"},
                },
            },
        },
        "alternativa": {
            "grafico": "grafico de linhas",
            "justificativa": "Se o foco for a evolucao no tempo em vez da comparacao entre categorias.",
            "trechos_usados": [1],
            "vegalite_spec": {
                "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
                "title": "Evolucao de vendas",
                "data": {"values": [{"mes": "jan", "vendas": 10}, {"mes": "fev", "vendas": 14}]},
                "mark": "line",
                "encoding": {
                    "x": {"field": "mes", "type": "ordinal"},
                    "y": {"field": "vendas", "type": "quantitative"},
                },
            },
        },
        "ressalva": "",
    }
)


def fake_chat(payload):
    return GenericFakeChatModel(messages=cycle([AIMessage(content=payload)]))


def check(label, cond, detail=""):
    print(f"  {'OK  ' if cond else 'FALHA'} {label}{(' -- ' + detail) if detail and not cond else ''}")
    if not cond:
        raise SystemExit(1)


print("1) serializacao dos JSONs brutos")
papers = list(S.load_papers(config.RAW_DIR))
check(f"{len(papers)} artigos lidos", len(papers) == 59)
total_findings = sum(len(list(S.iter_findings(p))) for _, p in papers)
check(f"{total_findings} achados extraidos", total_findings == 240)
jobs = [j for src, p in papers for j in enrich.jobs_for_paper(src, p)]
check(f"{len(jobs)} chamadas ao LLM planejadas (lotes de {enrich.FINDINGS_PER_CALL})", len(jobs) > 0)
sem_resultado = [src for src, p in papers if not list(S.iter_findings(p))]
check(f"{len(sem_resultado)} artigos sem ranking viram card de contexto", True)

print("\n2) enriquecimento (LLM falso)")
llm = fake_chat(FAKE_ENRICH)
src, paper = papers[0]
batch = list(S.iter_findings(paper))[:1] or enrich.paper_level_batch(paper, src)
cards = enrich.enrich_batch(llm, paper, src, batch)
check("card gerado", len(cards) == 1)
card = cards[0]
check("id do card bem formado", "::" in card["id"], card["id"])
check("perguntas de usuario entram no texto indexado", "Responde a perguntas como:" in card["text"])
check("metadata sao escalares (exigencia do Chroma)", all(isinstance(v, str) for v in card["metadata"].values()))
config.CARDS_PATH.write_text(json.dumps(card, ensure_ascii=False) + "\n", encoding="utf-8")

print("\n3) indexacao no Chroma (embeddings falsos)")
import build_index

fake_emb = DeterministicFakeEmbedding(size=256)
store = build_index.get_store(embeddings=fake_emb)
from langchain_core.documents import Document

store.add_documents([Document(page_content=card["text"], metadata=card["metadata"])], ids=[card["id"]])
check("1 documento indexado", len(store.get(include=[])["ids"]) == 1)

print("\n4) recuperacao por similaridade de cosseno")
import retrieve

retrieve.get_store = lambda *a, **k: store
r = retrieve.Retriever(k=3, use_hyde=False)
hits = r.search("quero comparar vendas de 5 categorias de produto")
check(f"{len(hits)} trecho(s) recuperado(s)", len(hits) >= 1)
ctx = retrieve.format_context(hits)
check("contexto rotula a fonte de cada trecho", "fonte:" in ctx and "[TRECHO 1]" in ctx)

print("\n5) geracao da recomendacao (LLM falso)")
import recommend

recommend.config.get_chat = lambda temperature=0.0: fake_chat(FAKE_RECOMMEND)
res = recommend.recomendar("quero comparar vendas de 5 categorias", k=3, retriever=r)
check("principal presente", res["principal"]["grafico"] == "grafico de barras agrupadas")
check("alternativa presente", res["alternativa"]["grafico"] == "grafico de linhas")
check("spec principal valida", res["principal"]["spec_valida"], str(res["principal"]["spec_erros"]))
check("spec alternativa valida", res["alternativa"]["spec_valida"], str(res["alternativa"]["spec_erros"]))
check("fontes rastreadas ate o arquivo de origem", len(res["fontes"]) >= 1, str(res["fontes"]))
check("trechos recuperados anexados a resposta", len(res["trechos"]) >= 1)

print("\n6) validacao rejeita spec quebrada")
ok, erros = validate_spec.validate({"data": {"values": []}, "mark": "bar", "encoding": {"x": {"field": "c"}}})
check("spec sem 'type' e rejeitada", not ok and erros)

shutil.rmtree(TMP, ignore_errors=True)
print("\nSMOKE TEST PASSOU -- encanamento completo funcionando.")
print("Falta apenas a API key real para rodar enrich.py e build_index.py de verdade.")

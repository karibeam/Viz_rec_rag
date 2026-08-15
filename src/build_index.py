"""Gera os embeddings dos cards enriquecidos e persiste no Chroma local.

Uso:
    python src/build_index.py
    python src/build_index.py --reset   # apaga e reconstroi o indice
"""

import argparse
import json
import re
import shutil
import sys
import time

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

import config
from langchain_chroma import Chroma
from langchain_core.documents import Document

# O free tier do Gemini limita embeddings por minuto (100 no momento da escrita).
# Lotes menores que o limite, com pausa entre eles, mantem o processo dentro da cota.
BATCH = 40
PAUSA = 60


def load_cards():
    if not config.CARDS_PATH.exists():
        raise SystemExit(
            f"{config.CARDS_PATH} nao existe. Rode antes: python src/enrich.py"
        )
    cards = []
    for line in config.CARDS_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            cards.append(json.loads(line))
    return cards


def get_store(embeddings=None):
    return Chroma(
        collection_name=config.COLLECTION,
        embedding_function=embeddings or config.get_embeddings(),
        persist_directory=str(config.CHROMA_DIR),
        # busca por similaridade de cosseno
        collection_metadata={"hnsw:space": "cosine"},
    )


def _espera_sugerida(exc, padrao: int) -> int:
    """Le o 'retry in Xs' devolvido pela API; cai no padrao se nao houver."""
    m = re.search(r"retry in ([\d.]+)s", str(exc), re.IGNORECASE)
    return int(float(m.group(1))) + 5 if m else padrao


def indexar_lote(store, docs, ids, tentativas: int = 5):
    """Indexa um lote, respeitando o 429 de cota em vez de abortar."""
    for tentativa in range(1, tentativas + 1):
        try:
            store.add_documents(documents=docs, ids=ids)
            return True
        except Exception as exc:
            if "RESOURCE_EXHAUSTED" not in str(exc) and "429" not in str(exc):
                raise
            if tentativa == tentativas:
                raise
            espera = _espera_sugerida(exc, PAUSA)
            print(f"    cota atingida; aguardando {espera}s (tentativa {tentativa}/{tentativas})")
            time.sleep(espera)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true")
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--pausa", type=int, default=PAUSA, help="segundos entre lotes")
    args = ap.parse_args()

    if args.reset and config.CHROMA_DIR.exists():
        shutil.rmtree(config.CHROMA_DIR)
        print(f"indice anterior removido: {config.CHROMA_DIR}")

    cards = load_cards()
    print(f"{len(cards)} cards carregados de {config.CARDS_PATH}")

    store = get_store()

    existing = set(store.get(include=[])["ids"])
    novos = [c for c in cards if c["id"] not in existing]
    print(f"{len(existing)} ja indexados | {len(novos)} novos")
    if not novos:
        print("indice ja atualizado.")
        return

    docs = [Document(page_content=c["text"], metadata=c["metadata"]) for c in novos]
    ids = [c["id"] for c in novos]

    lotes = list(range(0, len(docs), args.batch))
    for n, i in enumerate(lotes, 1):
        indexar_lote(store, docs[i : i + args.batch], ids[i : i + args.batch])
        print(f"  indexados {min(i + args.batch, len(docs))}/{len(docs)}")
        if n < len(lotes):
            time.sleep(args.pausa)

    print(f"\nindice salvo em {config.CHROMA_DIR} (colecao '{config.COLLECTION}')")


if __name__ == "__main__":
    main()

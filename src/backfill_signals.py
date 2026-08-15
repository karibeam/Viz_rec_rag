"""Preenche os sinais de forca de evidencia nos cards ja gerados.

Os sinais (significancia estatistica, experimental vs teorica) sao extraidos
deterministicamente do JSON original -- nao exigem LLM. Como metadados nao
entram no vetor, o indice e atualizado sem recalcular embeddings.

    python src/backfill_signals.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
import serialize as S
from build_index import get_store

CAMPOS = ("evidencia", "metrica", "significancia", "n_designs", "tem_empate")


def sinais_por_card():
    """{card_id: {sinais}} para todos os achados da base."""
    mapa = {}
    for source_file, paper in S.load_papers(config.RAW_DIR):
        prefixo = source_file.replace(".json", "")
        for finding_id, sinais in S.finding_signals(paper).items():
            mapa[f"{prefixo}::{finding_id}"] = sinais
    return mapa


def main():
    mapa = sinais_por_card()
    cards = [json.loads(l) for l in config.CARDS_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]

    atualizados = 0
    for card in cards:
        sinais = mapa.get(card["id"])
        if sinais is None:
            # cards de contexto (artigos sem ranking) nao tem achado associado
            card["metadata"].update(
                evidencia="nao-aplicavel", metrica="", significancia="nao-aplicavel",
                n_designs=0, tem_empate=False,
            )
            continue
        card["metadata"].update(sinais)
        atualizados += 1

    with config.CARDS_PATH.open("w", encoding="utf-8") as f:
        for card in cards:
            f.write(json.dumps(card, ensure_ascii=False) + "\n")
    print(f"{atualizados}/{len(cards)} cards com sinais de evidencia")

    if not config.CHROMA_DIR.exists():
        print("indice ausente; rode build_index.py depois.")
        return

    store = get_store()
    existentes = set(store.get(include=[])["ids"])
    alvo = [c for c in cards if c["id"] in existentes]
    # update() do chromadb altera metadados sem recalcular embeddings
    store._collection.update(
        ids=[c["id"] for c in alvo], metadatas=[c["metadata"] for c in alvo]
    )
    print(f"{len(alvo)} documentos atualizados no indice (sem recalcular embeddings)")


if __name__ == "__main__":
    main()

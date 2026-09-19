"""Compara a v1 (cards + placar) com a v2 (traducao da pergunta) nas 18 perguntas.

As respostas da v1 estao guardadas em tests/resultados_v1.json (geradas antes
de a v1 sair desta branch). A v2 roda duas vezes por pergunta, para medir
tambem se a resposta se repete.

    .venv/bin/python tests/comparar_v1_v2.py
"""

import json
import re
import sys
import unicodedata
from pathlib import Path

PASTA = Path(__file__).resolve().parent
sys.path.insert(0, str(PASTA.parent / "src"))

import config
from indexar import abrir_banco
from recomendar import recomendar

# Familias de grafico, para comparar nomes diferentes do mesmo tipo
# ("gráfico de barras agrupadas" e "barras horizontais" sao ambos barras).
FAMILIAS = {
    "barras": ["barra", "colunas"], "linhas": ["linha"], "dispersao": ["dispers", "pontos"],
    "pizza": ["pizza", "rosca", "circular", "fatias"], "calor": ["calor"],
    "caixa": ["caixa", "boxplot", "box plot"], "histograma": ["histograma"],
    "bolhas": ["bolha"], "tabela": ["tabela", "texto"],
}


def sem_acento(t):
    t = unicodedata.normalize("NFKD", (t or "").lower())
    return "".join(c for c in t if not unicodedata.combining(c))


def familias(texto):
    t = sem_acento(texto)
    return {f for f, chaves in FAMILIAS.items() if any(c in t for c in chaves)}


def carregar_perguntas():
    """Pergunta + resposta esperada, lidas de test_questions.md."""
    itens, vistas = [], set()
    for linha in (PASTA / "test_questions.md").read_text(encoding="utf-8").splitlines():
        m = re.match(r"^\|\s*\d+\s*\|\s*([^|]+?)\s*\|\s*([^|]*?)\s*\|", linha)
        if m and m.group(1) not in vistas:
            vistas.add(m.group(1))
            esperado = m.group(2)
            armadilha = esperado.startswith("ressalva")
            # "barras (pizza é pior)": o que esta entre parenteses NAO e resposta aceita
            itens.append((m.group(1), "(recusada)" if armadilha else re.sub(r"\(.*?\)", "", esperado)))
    return itens


def acertou(resposta, esperado):
    if esperado == "(recusada)":
        return resposta == "(recusada)"
    return bool(familias(resposta) & familias(esperado))


def main():
    v1 = {r["pergunta"]: r["v1"] for r in json.loads((PASTA / "resultados_v1.json").read_text())}
    banco, llm = abrir_banco(), config.get_chat()

    linhas = []
    for i, (pergunta, esperado) in enumerate(carregar_perguntas(), 1):
        v2 = []
        for _ in range(2):
            try:
                r = recomendar(pergunta, llm=llm, banco=banco)
                v2.append("(recusada)" if r.get("fora_de_escopo") else r["grafico"])
            except Exception as exc:
                v2.append(f"ERRO {type(exc).__name__}")
        linhas.append((pergunta, esperado, v1.get(pergunta, "?"), v2))
        print(f"[{i}/18] {pergunta[:50]}", flush=True)

    print(f"\n{'pergunta':<40} {'esperado':<16} {'v1':<24} {'v2':<24} v2 repetiu?")
    print("-" * 118)
    for pergunta, esperado, r1, (a, b) in linhas:
        ok1 = "✓" if acertou(r1, esperado) else "✗"
        ok2 = "✓" if acertou(a, esperado) else "✗"
        igual = "sim" if familias(a) == familias(b) and (a == "(recusada)") == (b == "(recusada)") else f"NAO ({b[:18]})"
        print(f"{pergunta[:39]:<40} {esperado.strip()[:15]:<16} {ok1} {r1[:21]:<22} {ok2} {a[:21]:<22} {igual}")

    n = len(linhas)
    print("\nRESUMO")
    print(f"  bate com o esperado  v1: {sum(acertou(r1, e) for _, e, r1, _ in linhas)}/{n}"
          f"   v2: {sum(acertou(v[0], e) for _, e, _, v in linhas)}/{n}")
    print(f"  v2 deu a mesma familia de grafico nas 2 execucoes: "
          f"{sum(familias(v[0]) == familias(v[1]) for *_, v in linhas)}/{n}")
    print("\n  'esperado' e um gabarito aproximado (tests/test_questions.md), nao uma verdade:")
    print("  a base pode legitimamente discordar do senso comum.")


if __name__ == "__main__":
    main()

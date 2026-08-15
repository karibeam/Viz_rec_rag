"""Teste de estabilidade: a mesma pergunta produz a mesma recomendacao?

Roda cada pergunta N vezes e compara os resultados. E o dado que decide se o
placar deterministico e necessario:

  - alta estabilidade  -> o desempate no prompt basta para o MVP
  - baixa estabilidade -> a escolha esta sendo arbitraria; e preciso agregar
                          os vencedores em Python antes de gerar

A instabilidade tem duas fontes: o HyDE gera um texto novo a cada busca (mudando
o que e recuperado) e a geracao em si tem temperatura > 0.

    .venv/bin/python tests/stability_test.py
    .venv/bin/python tests/stability_test.py --runs 3 --k 6 --hyde
"""

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import config
from recommend import recomendar
from retrieve import Retriever

QUESTIONS_MD = ROOT / "tests" / "test_questions.md"
SAIDA = ROOT / "tests" / "stability_report.json"


def carregar_perguntas():
    """Le as perguntas da tabela de avaliacao (mantem fonte unica da verdade)."""
    perguntas = []
    for linha in QUESTIONS_MD.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^\|\s*(\d+)\s*\|\s*([^|]+?)\s*\|", linha)
        if m:
            perguntas.append((int(m.group(1)), m.group(2).strip()))
    # numeros se repetem entre a tabela principal e a de armadilhas
    vistos, saida = set(), []
    for n, q in perguntas:
        if q not in vistos:
            vistos.add(q)
            saida.append(q)
    return saida


def normalizar(texto: str) -> str:
    """Compara nomes de grafico ignorando acento, caixa e pontuacao."""
    t = unicodedata.normalize("NFKD", (texto or "").lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", " ", t).strip()


def rodar(pergunta, k, use_hyde, runs, limpar_cache=False):
    import retrieve

    resultados = []
    for _ in range(runs):
        if limpar_cache:
            # mede a estabilidade sem o cache mascarar: e o pior caso, o de
            # reproduzir o experimento numa maquina limpa
            retrieve._CACHE_HYDE.pop(pergunta, None)
        try:
            r = recomendar(pergunta, k=k, retriever=Retriever(k=k, use_hyde=use_hyde))
            resultados.append(
                {
                    "fora_de_escopo": bool(r.get("fora_de_escopo")),
                    "principal": (r.get("principal") or {}).get("grafico", "") or ("(recusada)" if r.get("fora_de_escopo") else ""),
                    "alternativa": (r.get("alternativa") or {}).get("grafico", ""),
                    "fontes": sorted(r.get("fontes") or []),
                    "criterio": ((r.get("placar") or {}).get("tarefa_identificada") or ""),
                    "placar": [(c["nome"], c["pontos"]) for c in ((r.get("placar") or {}).get("ranking") or [])],
                    "houve_conflito": bool(r.get("conflito")),
                }
            )
        except Exception as exc:
            resultados.append({"erro": f"{type(exc).__name__}: {exc}"[:200]})
    return pergunta, resultados


def analisar(pergunta, resultados):
    validos = [r for r in resultados if "erro" not in r]
    if len(validos) < 2:
        return {"pergunta": pergunta, "status": "ERRO", "detalhe": resultados}

    principais = [normalizar(r["principal"]) for r in validos]
    alternativas = [normalizar(r["alternativa"]) for r in validos]
    fontes = [tuple(r["fontes"]) for r in validos]

    return {
        "pergunta": pergunta,
        "principal_estavel": len(set(principais)) == 1,
        "alternativa_estavel": len(set(alternativas)) == 1,
        "fontes_estaveis": len(set(fontes)) == 1,
        "principais_vistos": [r["principal"] for r in validos],
        "criterios": [r["criterio"] for r in validos],
        "conflitos": sum(r["houve_conflito"] for r in validos),
        "placar_estavel": len({tuple(r.get("placar") or []) for r in validos}) == 1,
        "sobreposicao_fontes": round(
            len(set(fontes[0]) & set(fontes[-1])) / max(len(set(fontes[0]) | set(fontes[-1])), 1), 2
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=2, help="execucoes por pergunta")
    ap.add_argument("-k", type=int, default=6)
    ap.add_argument("--hyde", action="store_true", help="liga o HyDE (desligado por padrao)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sem-cache", action="store_true",
                    help="descarta o cache do HyDE entre execucoes (pior caso)")
    args = ap.parse_args()

    perguntas = carregar_perguntas()
    if args.limit:
        perguntas = perguntas[: args.limit]
    use_hyde = args.hyde

    print(f"{len(perguntas)} perguntas x {args.runs} execucoes | k={args.k} | HyDE={use_hyde}")
    print(f"provedor: {config.provider()}\n")

    analises = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futuros = [
            pool.submit(rodar, q, args.k, use_hyde, args.runs, getattr(args, 'sem_cache', False))
            for q in perguntas
        ]
        for i, fut in enumerate(futuros, 1):
            pergunta, resultados = fut.result()
            a = analisar(pergunta, resultados)
            analises.append(a)
            if a.get("status") == "ERRO":
                print(f"[{i}/{len(perguntas)}] ERRO  {pergunta[:50]}")
                continue
            marca = "OK  " if a["principal_estavel"] else "MUDA"
            print(f"[{i}/{len(perguntas)}] {marca} {pergunta[:48]:<50} {' | '.join(dict.fromkeys(a['principais_vistos']))[:60]}")

    validas = [a for a in analises if a.get("status") != "ERRO"]
    if not validas:
        print("\nnenhuma analise valida.")
        return

    n = len(validas)
    est_p = sum(a["principal_estavel"] for a in validas)
    est_a = sum(a["alternativa_estavel"] for a in validas)
    est_f = sum(a["fontes_estaveis"] for a in validas)
    sobre = sum(a["sobreposicao_fontes"] for a in validas) / n

    print("\n" + "=" * 72)
    print(f"ESTABILIDADE ({n} perguntas, {args.runs} execucoes cada)")
    print("=" * 72)
    print(f"  recomendacao PRINCIPAL identica:   {est_p}/{n}  ({100*est_p/n:.0f}%)")
    print(f"  recomendacao ALTERNATIVA identica: {est_a}/{n}  ({100*est_a/n:.0f}%)")
    print(f"  FONTES citadas identicas:          {est_f}/{n}  ({100*est_f/n:.0f}%)")
    print(f"  sobreposicao media das citacoes:   {sobre:.2f}")
    if not use_hyde:
        print("    (sem HyDE a recuperacao e deterministica: os k chunks sao sempre\n     os mesmos. Esta metrica mede quais deles o LLM escolheu citar.)")
    print(f"  perguntas com conflito declarado:  {sum(1 for a in validas if a['conflitos'])}/{n}")
    fora = sum(1 for a in validas if all(p == "(recusada)" for p in a["principais_vistos"]))
    print(f"  perguntas recusadas (fora de escopo): {fora}/{n}")
    crit = Counter(c for a in validas for c in a["criterios"] if c)
    est_pl = sum(a.get("placar_estavel", False) for a in validas)
    print(f"  PLACAR identico (deterministico):  {est_pl}/{n}  ({100*est_pl/n:.0f}%)")
    print(f"  tarefas identificadas:             {dict(crit)}")

    taxa = est_p / n
    print("\nLEITURA:")
    if est_pl < n:
        print("  ATENCAO: o placar deveria ser 100% deterministico e nao foi.")
        print("  Investigue antes de confiar nos demais numeros.")
    elif taxa >= 0.95:
        print("  Reproduzivel. O placar decide e o LLM respeita a decisao.")
    elif taxa >= 0.8:
        print("  Quase reproduzivel. O placar e estavel, mas o LLM ocasionalmente")
        print("  renomeia o grafico vencedor -- reforce a instrucao de copiar o nome.")
    else:
        print("  O placar e estavel mas a saida final nao. O LLM esta ignorando o")
        print("  placar: revise o bloco de instrucoes em recommend.SYSTEM.")

    SAIDA.write_text(
        json.dumps(
            {"config": vars(args), "resumo": {"principal_estavel": est_p, "total": n}, "analises": analises},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nrelatorio completo: {SAIDA}")


if __name__ == "__main__":
    main()

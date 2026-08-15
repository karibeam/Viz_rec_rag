"""Serializacao deterministica dos JSONs da base para texto legivel.

Este modulo NAO interpreta nem recomenda nada: ele apenas reformata o JSON
tecnico (encodings, marks, rankings) em texto plano para servir de entrada ao
passo de enriquecimento por LLM. Toda a interpretacao ("isto e um grafico de
barras", "serve para comparar categorias") e feita pelo LLM em enrich.py.
"""

import json
import re
from pathlib import Path

# Rotulos de metricas: informam ao LLM como ler o ranking (maior = melhor?).
METRIC_HINTS = {
    "accuracy": "acuracia (posicoes iniciais do ranking = respostas mais corretas)",
    "time": "tempo de resposta (posicoes iniciais do ranking = leitura mais rapida)",
    "JND": "diferenca minima perceptivel (posicoes iniciais = diferencas menores sao percebidas)",
    "effectiveness": "efetividade teorica proposta pelos autores",
    "user-preference": "preferencia declarada pelos participantes",
    "bias": "vies de estimativa",
    "bias-underestimate": "vies de subestimacao",
    "bias-overestimate": "vies de superestimacao",
    "bias-perceptual-pull": "vies de atracao perceptual",
}


def _encoding_str(enc: dict) -> str:
    parts = [f"{enc.get('data-type') or '?'} -> canal {enc.get('channel') or '?'}"]
    if enc.get("scale"):
        parts.append(f"escala {enc['scale']}")
    charcs = enc.get("data-charcs") or {}
    if charcs:
        parts.append(
            "caracteristicas do dado: "
            + ", ".join(f"{k}={v}" for k, v in charcs.items())
        )
    trans = enc.get("data-trans") or {}
    if trans:
        parts.append(
            "transformacoes: " + ", ".join(f"{k}={v}" for k, v in trans.items())
        )
    return " | ".join(parts)


def design_str(design_id: str, design: dict) -> str:
    """Uma linha por design, com marca, encodings e nota do artigo."""
    chunks = []
    for layer in design.get("layers") or []:
        mark = layer.get("mark") or "(marca nao especificada)"
        encs = "; ".join(_encoding_str(e) for e in (layer.get("encodings") or []))
        chunks.append(f"marca={mark} :: {encs}")
    body = " || ".join(chunks) if chunks else "(sem camadas)"
    note = design.get("note")
    if note:
        body += f' :: nota do artigo: "{note}"'
    return f"- {design_id}: {body}"


def _unwrap(rank):
    """Remove colchetes redundantes de um so elemento.

    Alguns arquivos envolvem o ranking inteiro em uma lista extra
    (`[[[...], "E-6"]]`), o que desloca a leitura de ordem/empate em um nivel.
    """
    while isinstance(rank, list) and len(rank) == 1 and isinstance(rank[0], list):
        rank = rank[0]
    return rank


def _rank_str(rank, nivel: int = 0) -> str:
    """Ranking do melhor para o pior; empates unidos por '='.

    O aninhamento alterna o significado: nivel par = ordem ('>'),
    nivel impar = empate ('='). A profundidade varia entre os arquivos
    (ha rankings com ate tres niveis), por isso o tratamento e recursivo.
    """
    if nivel == 0:
        rank = _unwrap(rank)
    if not isinstance(rank, list):
        return str(rank)
    sep = " > " if nivel % 2 == 0 else " = "
    partes = []
    for item in rank:
        texto = _rank_str(item, nivel + 1)
        if isinstance(item, list) and len(item) > 1:
            texto = f"({texto})"
        partes.append(texto)
    return sep.join(partes)


def _flatten(valor):
    """Todos os ids de design de um ranking, em qualquer profundidade."""
    if isinstance(valor, list):
        return [x for item in valor for x in _flatten(item)]
    return [valor] if valor is not None else []


def _significance_str(sig) -> str:
    if not isinstance(sig, dict) or not sig.get("pairs"):
        return "sem pares com diferenca significativa reportada"
    pairs = ", ".join("/".join(map(str, p)) for p in sig["pairs"])
    extra = []
    if sig.get("significance-method"):
        extra.append(f"metodo={sig['significance-method']}")
    if sig.get("significance-threshold") is not None:
        extra.append(f"limiar={sig['significance-threshold']}")
    if sig.get("effect-size-value") is not None:
        extra.append(
            f"tamanho de efeito={sig.get('effect-size-method')}:{sig['effect-size-value']}"
        )
    suffix = f" ({'; '.join(extra)})" if extra else ""
    return f"diferencas estatisticamente significativas entre: {pairs}{suffix}"


def iter_findings(paper: dict):
    """Gera (finding_id, texto_do_achado, ids_dos_designs_citados) por resultado."""
    for group, metrics in (paper.get("Results") or {}).items():
        for metric, entries in (metrics or {}).items():
            if not isinstance(entries, dict):
                continue
            for entry_id, entry in entries.items():
                if not isinstance(entry, dict):
                    continue
                rank = entry.get("rank")
                hint = METRIC_HINTS.get(metric, metric)
                text = (
                    f"[{group}/{metric}/{entry_id}] metrica={metric} ({hint})\n"
                    f"  ranking do melhor para o pior: {_rank_str(rank)}\n"
                    f"  {_significance_str(entry.get('significance'))}"
                )
                yield f"{group}/{metric}/{entry_id}", text, _flatten(rank)


# As 10 tarefas analiticas do campo `Tasks` da base original. Usar esta lista em
# vez dos rotulos que o LLM gerou evita a fragmentacao do vocabulario
# ("ver correlacao" vs "ver correlação" viravam categorias distintas).
TAREFAS_CANONICAS = (
    "aggregate",
    "characterize-distribution",
    "cluster",
    "correlate",
    "determine-range",
    "filter",
    "find-anomalies",
    "find-extremum",
    "retrieve-value",
    "sort",
)


def tarefa_canonica(entry_id: str) -> str:
    """Extrai a tarefa do id do achado ('sort-1' -> 'sort').

    Os ids dos resultados sao nomeados pela tarefa no dataset original, entao a
    extracao e deterministica. Retorna "" quando nao ha tarefa associada
    (ranking teorico geral, por exemplo 'overall').
    """
    base = re.sub(r"[-_ ]?\d+$", "", str(entry_id or "").strip()).replace(" ", "-")
    return base if base in TAREFAS_CANONICAS else ""


def finding_signals(paper: dict) -> dict:
    """Sinais de forca de evidencia por achado, extraidos do JSON original.

    Deterministico e sem LLM: sao fatos do artigo, nao interpretacao. Servem
    como criterio de desempate quando trechos recuperados discordam entre si.
    """
    sinais = {}
    for group, metrics in (paper.get("Results") or {}).items():
        for metric, entries in (metrics or {}).items():
            if not isinstance(entries, dict):
                continue
            for entry_id, entry in entries.items():
                if not isinstance(entry, dict):
                    continue
                sig = entry.get("significance")
                pares = (sig or {}).get("pairs") if isinstance(sig, dict) else None
                rank = _unwrap(entry.get("rank"))
                sinais[f"{group}/{metric}/{entry_id}"] = {
                    "evidencia": "experimental" if group == "Experimental" else "teorica",
                    "tarefa_canonica": tarefa_canonica(entry_id),
                    "metrica": metric,
                    "significancia": "sim" if pares else "nao-reportada",
                    "n_designs": len(_flatten(rank)),
                    "tem_empate": any(
                        isinstance(i, list) and len(i) > 1 for i in (rank or [])
                    ),
                }
    return sinais


def paper_header(paper: dict, source_file: str) -> str:
    return (
        f"ARTIGO: {paper.get('Title', '(sem titulo)')}\n"
        f"ARQUIVO: {source_file}\n"
        f"CATEGORIA: {paper.get('Category', '?')}\n"
        f"TAREFAS ANALITICAS ESTUDADAS: {', '.join(paper.get('Tasks') or []) or '(nenhuma)'}"
    )


def serialize_paper(paper: dict, source_file: str, design_ids=None) -> str:
    """Texto completo do artigo; design_ids limita os designs incluidos."""
    designs = paper.get("Designs") or {}
    if design_ids is not None:
        wanted = set(design_ids)
        designs = {k: v for k, v in designs.items() if k in wanted}
    lines = [paper_header(paper, source_file), "", "DESIGNS AVALIADOS:"]
    lines += [design_str(k, v) for k, v in designs.items()] or ["(nenhum)"]
    return "\n".join(lines)


def load_papers(raw_dir: Path):
    for path in sorted(raw_dir.glob("*.json")):
        yield path.name, json.loads(path.read_text(encoding="utf-8"))

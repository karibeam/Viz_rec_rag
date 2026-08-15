"""Enriquecimento da base: JSON tecnico -> cards narrativos indexaveis.

Este e o passo que resolve o problema de similaridade. A base original descreve
os estudos em vocabulario de artigo academico ("channel: positionX",
"mark: area-rect"), que nao tem sobreposicao semantica com a pergunta de um
usuario leigo ("quero comparar vendas por categoria"). Aqui um LLM reescreve
cada achado em linguagem natural, ja incluindo o nome usual do grafico e
exemplos de perguntas do dia a dia que aquele achado responde -- de modo que o
vetor do documento passe a viver perto do vetor da pergunta.

Roda uma unica vez (offline). Suporta retomada: cards ja gerados sao pulados.

Uso:
    python src/enrich.py                 # base inteira
    python src/enrich.py --limit 3       # amostra, para testar o prompt
    python src/enrich.py --force         # regera tudo do zero
"""

import argparse
import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

import config
import serialize as S

FINDINGS_PER_CALL = 6

SYSTEM = """Voce e um especialista em visualizacao de dados e percepcao grafica.
Sua tarefa e traduzir achados de artigos cientificos (descritos em notacao tecnica
de codificacao visual) para fichas em portugues do Brasil, escritas em linguagem
acessivel a quem NAO e especialista em visualizacao.

Como ler a notacao tecnica:
- "canal" e a propriedade visual usada para codificar o dado (positionX/positionY =
  posicao nos eixos, length = comprimento, angle = angulo, area = area,
  color-hue = matiz, color-saturation = saturacao, orientation = inclinacao,
  row/column = facetas/paineis lado a lado, shape = formato do simbolo).
- "marca" e a forma desenhada (area-rect = barra/retangulo, line = linha,
  point = ponto, area-arc = fatia de pizza/rosca, area-circle = circulo/bolha,
  area-arc-inner-50 = rosca com furo de 50%).
- Combinacoes usuais: area-rect + positionX nominal + positionY quantitative =
  grafico de barras; line + positionX + positionY = grafico de linhas;
  point + positionX + positionY quantitativos = grafico de dispersao;
  area-arc + angle = grafico de pizza.
- O ranking vem SEMPRE do melhor para o pior segundo a metrica indicada.
  Empates aparecem unidos por "=".

Regras de escrita:
- Escreva em portugues do Brasil, sem jargao. Se precisar usar um termo tecnico,
  explique entre parenteses.
- Use o nome popular do grafico ("grafico de barras agrupadas", "grafico de pizza",
  "grafico de dispersao", "mapa de calor"), nunca "area-rect" ou "positionY".
- Nunca invente resultados: descreva apenas o que o ranking e as notas informam.
- Se o ranking for teorico (nao experimental), deixe isso explicito no texto.

Responda SEMPRE com um array JSON valido, sem cercas de codigo e sem comentarios."""

USER_TMPL = """{paper}

ACHADOS A CONVERTER (um objeto JSON por achado):
{findings}

Para cada achado, produza um objeto com EXATAMENTE estas chaves:
- "finding_id": string, copie exatamente o identificador entre colchetes do achado.
- "titulo": string curta (ate 100 caracteres) resumindo o achado.
- "texto": 4 a 7 frases em portugues claro. Deve dizer: qual tarefa analitica foi
  estudada, quais alternativas de grafico foram comparadas (pelo nome popular),
  qual saiu melhor e qual saiu pior segundo a metrica, e uma frase final comecando
  com "Indicado para:" explicando em que situacao pratica usar a opcao vencedora.
- "graficos": array de strings com os nomes populares de todos os graficos comparados.
- "recomendado": string, nome popular da opcao que ficou em primeiro no ranking.
- "evitar": string, nome popular da opcao que ficou em ultimo (ou "" se nao houver).
- "tarefas": array de strings com as tarefas analiticas envolvidas, em portugues
  (ex: "comparar valores", "ver correlacao", "encontrar o maior valor",
  "ver distribuicao", "identificar grupos", "ler um valor especifico",
  "detectar anomalias", "estimar uma media").
- "tipos_de_dado": array com os tipos de dado envolvidos, em portugues
  ("quantitativo", "nominal/categorico", "ordinal", "temporal").
- "perguntas_de_usuario": array de 3 perguntas curtas, do jeito que uma pessoa
  LEIGA perguntaria no dia a dia e que este achado ajuda a responder.
  Exemplo de estilo: "Qual grafico usar para comparar vendas de varios produtos?"

Array JSON com {n} objeto(s):"""

_lock = threading.Lock()


def parse_json_array(text: str):
    text = re.sub(r"^\s*```(?:json)?|```\s*$", "", text.strip(), flags=re.MULTILINE)
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1:
        raise ValueError(f"resposta sem array JSON: {text[:200]}")
    return json.loads(text[start : end + 1])


def as_list(value):
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    return [str(value).strip()] if str(value or "").strip() else []


def build_card(obj, paper, source_file, finding_id):
    """Monta o registro final. O campo `text` e o que sera vetorizado."""
    titulo = str(obj.get("titulo") or "").strip()
    texto = str(obj.get("texto") or "").strip()
    graficos = as_list(obj.get("graficos"))
    tarefas = as_list(obj.get("tarefas"))
    tipos = as_list(obj.get("tipos_de_dado"))
    perguntas = as_list(obj.get("perguntas_de_usuario"))
    recomendado = str(obj.get("recomendado") or "").strip()
    evitar = str(obj.get("evitar") or "").strip()

    # As perguntas de usuario entram no texto indexado de proposito: e o que
    # aproxima o vetor do documento do vetor da pergunta de um nao-especialista.
    partes = [titulo, texto]
    if tarefas:
        partes.append("Tarefas analiticas: " + ", ".join(tarefas) + ".")
    if tipos:
        partes.append("Tipos de dado: " + ", ".join(tipos) + ".")
    if perguntas:
        partes.append("Responde a perguntas como: " + " ".join(perguntas))
    partes.append(f"Fonte: {paper.get('Title', '')} ({source_file}).")

    return {
        "id": f"{source_file.replace('.json', '')}::{finding_id}",
        "text": "\n".join(p for p in partes if p),
        "metadata": {
            "source_file": source_file,
            "paper_title": str(paper.get("Title") or ""),
            "category": str(paper.get("Category") or ""),
            "finding_id": finding_id,
            # artigos sem ranking comparativo viram card de contexto, nao de achado
            "kind": "overview" if finding_id.startswith("paper/") else "finding",
            "titulo": titulo,
            "graficos": ", ".join(graficos),
            "recomendado": recomendado,
            "evitar": evitar,
            "tarefas": ", ".join(tarefas),
            "tipos_de_dado": ", ".join(tipos),
        },
    }


def enrich_batch(llm, paper, source_file, batch):
    """batch: lista de (finding_id, texto_do_achado, design_ids)."""
    design_ids = sorted({d for _, _, ids in batch for d in ids})
    paper_text = S.serialize_paper(paper, source_file, design_ids or None)
    findings_text = "\n\n".join(t for _, t, _ in batch)
    prompt = USER_TMPL.format(paper=paper_text, findings=findings_text, n=len(batch))

    resp = llm.invoke([("system", SYSTEM), ("human", prompt)])
    objs = parse_json_array(config.text_of(resp))

    by_id = {str(o.get("finding_id", "")).strip(): o for o in objs if isinstance(o, dict)}
    cards = []
    for i, (fid, _, _) in enumerate(batch):
        obj = by_id.get(fid) or (objs[i] if i < len(objs) else None)
        if isinstance(obj, dict):
            cards.append(build_card(obj, paper, source_file, fid))
    return cards


def paper_level_batch(paper, source_file):
    """Artigos sem resultados ranqueados ainda entram como card de contexto."""
    designs = paper.get("Designs") or {}
    listing = "\n".join(S.design_str(k, v) for k, v in designs.items()) or "(nenhum)"
    text = (
        f"[paper/overview] Este artigo nao reporta um ranking comparativo explicito.\n"
        f"  designs descritos no artigo:\n{listing}\n"
        f"  Resuma o que o artigo cobre e para que serve, sem inventar resultados."
    )
    return [("paper/overview", text, list(designs.keys()))]


def jobs_for_paper(source_file, paper):
    findings = list(S.iter_findings(paper))
    if not findings:
        findings = paper_level_batch(paper, source_file)
    return [
        (source_file, paper, findings[i : i + FINDINGS_PER_CALL])
        for i in range(0, len(findings), FINDINGS_PER_CALL)
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="processa so N artigos")
    ap.add_argument("--force", action="store_true", help="ignora cards ja gerados")
    ap.add_argument(
        "--workers",
        type=int,
        default=2,
        help="chamadas em paralelo (o free tier limita requisicoes por minuto)",
    )
    ap.add_argument(
        "--model",
        default=os.getenv("ENRICH_CHAT_MODEL") or None,
        help="modelo do enriquecimento; use um mais leve/rapido que o do runtime",
    )
    args = ap.parse_args()

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    done_ids = set()
    if config.CARDS_PATH.exists() and not args.force:
        for line in config.CARDS_PATH.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done_ids.add(json.loads(line)["id"])
    elif args.force and config.CARDS_PATH.exists():
        config.CARDS_PATH.unlink()

    papers = list(S.load_papers(config.RAW_DIR))
    if not papers:
        raise SystemExit(f"nenhum JSON em {config.RAW_DIR}")
    if args.limit:
        papers = papers[: args.limit]

    jobs = [j for src, p in papers for j in jobs_for_paper(src, p)]
    jobs = [
        j
        for j in jobs
        if any(f"{j[0].replace('.json', '')}::{fid}" not in done_ids for fid, _, _ in j[2])
    ]

    print(f"{len(papers)} artigos | {len(jobs)} chamadas ao LLM | ja feitos: {len(done_ids)}")
    if not jobs:
        print("nada a fazer.")
        return

    llm = config.get_chat(temperature=0.2, model=args.model)
    print(f"modelo: {args.model or 'default do .env'} | workers: {args.workers}")
    written = failed = 0

    with config.CARDS_PATH.open("a", encoding="utf-8") as out:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(enrich_batch, llm, paper, src, batch): src
                for src, paper, batch in jobs
            }
            for i, fut in enumerate(as_completed(futures), 1):
                src = futures[fut]
                try:
                    cards = fut.result()
                except Exception as exc:  # rede, quota, JSON invalido
                    failed += 1
                    print(f"  [{i}/{len(jobs)}] FALHOU {src}: {type(exc).__name__}: {exc}")
                    continue
                with _lock:
                    for card in cards:
                        if card["id"] in done_ids:
                            continue
                        out.write(json.dumps(card, ensure_ascii=False) + "\n")
                        done_ids.add(card["id"])
                        written += 1
                    out.flush()
                print(f"  [{i}/{len(jobs)}] {src}: +{len(cards)} cards")

    print(f"\n{written} cards escritos em {config.CARDS_PATH}")
    if failed:
        print(f"{failed} chamadas falharam -- rode de novo para retomar de onde parou.")


if __name__ == "__main__":
    main()

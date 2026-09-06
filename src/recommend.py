"""Geracao da recomendacao final a partir dos trechos recuperados.

Saida: UMA recomendacao de grafico, com spec Vega-Lite validada e justificativa
ancorada nas fontes recuperadas. O ranking completo dos candidatos vai junto,
no campo "placar", para auditoria -- mas so o 1o colocado e recomendado.

Uso como CLI (util para depurar sem interface):
    python src/recommend.py "quero comparar vendas de 5 categorias"
    python src/recommend.py --hyde -k 8 "minha pergunta"   # HyDE desligado por padrao
"""

import argparse
import json
import re
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

import config
import tally
import validate_spec
from retrieve import Retriever, format_context

SYSTEM = """Voce recomenda visualizacoes de dados para pessoas que NAO sao
especialistas no assunto.

REGRA CENTRAL: baseie a recomendacao exclusivamente nos TRECHOS de conhecimento
fornecidos, que vem de estudos empiricos de percepcao grafica. Voce nao possui
regras proprias de recomendacao: se os trechos nao sustentarem uma escolha, diga
isso no campo "ressalva" em vez de inventar. Toda justificativa deve remeter ao
que os trechos afirmam, citando o numero do trecho usado.

A ESCOLHA DO GRAFICO NAO E SUA. Ela ja foi calculada por um placar
deterministico, a partir dos mesmos trechos que voce recebeu, ponderando:
compatibilidade com a tarefa analitica, significancia estatistica, evidencia
experimental, numero de alternativas comparadas e, por ultimo, similaridade.

  - "recomendacao" DEVE ser o 1o colocado do placar, e apenas ele.
  - Use exatamente o nome de grafico do placar.
  - Nao proponha outro grafico, nem sugira segundas opcoes.

Seu trabalho e EXPLICAR essa decisao em linguagem simples e desenhar a spec
Vega-Lite. Justifique citando os trechos que votaram nela (a coluna "a favor"
do placar diz quais foram).

Se o 1o colocado for claramente inadequado para a pergunta, nao o troque: diga
o problema no campo "ressalva".

Escreva em portugues do Brasil, em linguagem simples, sem jargao academico.
Responda SEMPRE com um unico objeto JSON valido, sem cercas de codigo."""

USER_TMPL = """PERGUNTA DO USUARIO:
{pergunta}

{placar}

TRECHOS RECUPERADOS DA BASE DE CONHECIMENTO:
{contexto}

Produza um objeto JSON com EXATAMENTE estas chaves:

{{
  "recomendacao": {{
    "grafico": "copie o nome do 1o colocado do placar",
    "justificativa": "2 a 4 frases explicando por que este grafico venceu, citando os trechos que votaram nele (ex: 'segundo o trecho 2, ...'). Sem jargao.",
    "trechos_usados": [numeros dos trechos que sustentam esta escolha],
    "vegalite_spec": {{ spec Vega-Lite v5 completa e renderizavel }}
  }},
  "ressalva": "string: escreva aqui se os trechos recuperados nao cobrirem bem a pergunta, ou \\"\\" se a base sustenta bem a resposta.",
  "conflito": "string: se outros graficos do placar ficaram perto do vencedor, explique em 1-2 frases, para um leigo, o que fez o 1o colocado vencer. Deixe \\"\\" se a escolha foi folgada ou se so houve um candidato."
}}

Regras para as specs Vega-Lite:
- Use a versao 5 e inclua "$schema": "https://vega.github.io/schema/vega-lite/v5.json".
- Embuta um exemplo pequeno e plausivel em "data": {{"values": [...]}}, com 4 a 8
  linhas, usando nomes de campos coerentes com a pergunta do usuario (em portugues).
- Sempre defina "encoding" com "type" em cada canal ("quantitative", "nominal",
  "ordinal" ou "temporal").
- Inclua "title" no grafico.
- Nao use "datasets", "transform" complexos, URLs externas nem "config" extenso.

Objeto JSON:"""


def parse_json_object(text: str) -> dict:
    text = re.sub(r"^\s*```(?:json)?|```\s*$", "", text.strip(), flags=re.MULTILINE)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"resposta sem objeto JSON: {text[:300]}")
    return json.loads(text[start : end + 1])


def _fontes_de(trechos_usados, hits):
    fontes = []
    for n in trechos_usados or []:
        try:
            doc, _ = hits[int(n) - 1]
        except (ValueError, TypeError, IndexError):
            continue
        m = doc.metadata
        ref = f"{m.get('source_file', '?')}#{m.get('finding_id', '?')}"
        if ref not in fontes:
            fontes.append(ref)
    return fontes


def _finalizar_opcao(opcao, hits):
    if not isinstance(opcao, dict):
        return None
    spec = opcao.get("vegalite_spec")
    validate_spec.ensure_inline_data(spec)
    spec, reparos = validate_spec.repair(spec)
    ok, erros = validate_spec.validate(spec)
    return {
        "grafico": str(opcao.get("grafico") or "").strip(),
        "justificativa": str(opcao.get("justificativa") or "").strip(),
        "vegalite_spec": spec,
        "spec_valida": ok,
        "spec_erros": erros,
        "spec_reparos": reparos,
        "fontes": _fontes_de(opcao.get("trechos_usados"), hits),
    }


def recomendar(pergunta: str, k: int = 9, use_hyde: bool = False, retriever=None,
               usar_placar: bool = True):
    r = retriever or Retriever(k=k, use_hyde=use_hyde)

    tarefa = None
    if usar_placar:
        tarefa, _ = tally.tarefa_da_pergunta(pergunta)
        hits = r.search_com_tarefa(pergunta, tarefa, k=k)
    else:
        hits = r.search(pergunta, k=k)

    if not hits:
        return {
            "pergunta": pergunta,
            "erro": "nenhum trecho recuperado -- o indice esta vazio? rode build_index.py",
        }

    contexto = format_context(hits)
    placar = tally.montar_placar(pergunta, hits, tarefa_alvo=tarefa) if usar_placar else None

    if placar and placar["fora_de_escopo"]:
        # Recusa deterministica e sem custo de API: a pergunta nao corresponde a
        # nenhuma das tarefas analiticas cobertas pela base.
        return {
            "pergunta": pergunta,
            "recomendacao": None,
            "fora_de_escopo": True,
            "ressalva": (
                "Esta pergunta nao parece ser sobre qual grafico usar para uma analise. "
                "A base cobre estudos de percepcao grafica para 10 tarefas (comparar valores, "
                "ver correlacao, encontrar extremos, ver distribuicao, entre outras), e a "
                f"pergunta ficou abaixo do piso de relevancia "
                f"({placar['similaridade_tarefa']} < {placar['limiar_tarefa']}). "
                "Tente descrever a analise que quer fazer com seus dados."
            ),
            "conflito": "",
            "placar": placar,
            "fontes": [],
            "trechos": [],
        }
    bloco_placar = tally.formatar_placar(placar) if placar else "(placar desativado)"

    # temperatura 0: a recomendacao precisa ser reproduzivel entre execucoes
    llm = config.get_chat(temperature=0.0)
    resp = config.invoke_with_retry(
        llm,
        [
            ("system", SYSTEM),
            ("human", USER_TMPL.format(pergunta=pergunta, contexto=contexto, placar=bloco_placar)),
        ],
    )
    obj = parse_json_object(config.text_of(resp))

    recomendacao = _finalizar_opcao(obj.get("recomendacao"), hits)

    return {
        "pergunta": pergunta,
        "recomendacao": recomendacao,
        "ressalva": str(obj.get("ressalva") or "").strip(),
        "conflito": str(obj.get("conflito") or "").strip(),
        "placar": placar,
        "fontes": (recomendacao or {}).get("fontes", []),
        "trechos": [
            {
                "n": i,
                "fonte": f"{d.metadata.get('source_file')}#{d.metadata.get('finding_id')}",
                "similaridade": round(float(s), 4),
                "titulo": d.metadata.get("titulo", ""),
                "texto": d.page_content,
            }
            for i, (d, s) in enumerate(hits, 1)
        ],
    }


def imprimir(res: dict):
    if res.get("erro"):
        print("ERRO:", res["erro"])
        return
    if res.get("fora_de_escopo"):
        print(f"\nPERGUNTA: {res['pergunta']}\n")
        print(f"FORA DE ESCOPO: {res['ressalva']}")
        return
    print(f"\nPERGUNTA: {res['pergunta']}\n")
    opc = res.get("recomendacao")
    if not opc:
        print("RECOMENDACAO: (nao gerada)\n")
    else:
        flag = "" if opc["spec_valida"] else "  [SPEC INVALIDA: " + "; ".join(opc["spec_erros"]) + "]"
        print(f"RECOMENDACAO: {opc['grafico']}{flag}")
        print(f"  {opc['justificativa']}")
        print(f"  fontes: {', '.join(opc['fontes']) or '(nenhuma citada)'}\n")
    placar = res.get("placar")
    if placar:
        print(f"PLACAR (tarefa identificada: {placar.get('tarefa_identificada') or '?'})")
        for c in placar["ranking"][:5]:
            favor = ", ".join(str(a["trecho"]) for a in c["a_favor"]) or "-"
            print(f"  {c['posicao']}. {c['pontos']:>7.3f}  {c['nome'][:48]:<50} trechos {favor}")
        print()
    if res.get("conflito"):
        print(f"CONFLITO: {res['conflito']}\n")
    if res.get("ressalva"):
        print(f"RESSALVA: {res['ressalva']}\n")
    print("TRECHOS RECUPERADOS:")
    for t in res["trechos"]:
        print(f"  [{t['n']}] {t['similaridade']:.3f}  {t['fonte']}  {t['titulo'][:70]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pergunta", nargs="+")
    ap.add_argument("-k", type=int, default=9)
    ap.add_argument("--hyde", action="store_true", help="liga o HyDE (desligado por padrao)")
    ap.add_argument("--json", action="store_true", help="imprime o JSON completo")
    args = ap.parse_args()

    res = recomendar(" ".join(args.pergunta), k=args.k, use_hyde=args.hyde)
    if args.json:
        print(json.dumps(res, ensure_ascii=False, indent=2))
    else:
        imprimir(res)


if __name__ == "__main__":
    main()

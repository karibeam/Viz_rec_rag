"""Geracao da recomendacao final a partir dos trechos recuperados.

Saida: recomendacao PRINCIPAL + ALTERNATIVA, cada uma com spec Vega-Lite
validada e justificativa ancorada nas fontes recuperadas.

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
import validate_spec
from retrieve import Retriever, format_context

SYSTEM = """Voce recomenda visualizacoes de dados para pessoas que NAO sao
especialistas no assunto.

REGRA CENTRAL: baseie a recomendacao exclusivamente nos TRECHOS de conhecimento
fornecidos, que vem de estudos empiricos de percepcao grafica. Voce nao possui
regras proprias de recomendacao: se os trechos nao sustentarem uma escolha, diga
isso no campo "ressalva" em vez de inventar. Toda justificativa deve remeter ao
que os trechos afirmam, citando o numero do trecho usado.

CRITERIO DE DESEMPATE (obrigatorio quando os trechos apontam graficos diferentes):
os trechos vem de estudos independentes, que compararam conjuntos diferentes de
alternativas. Quando discordarem, aplique NESTA ORDEM:

  1. Prefira o trecho cuja TAREFA ANALITICA corresponde ao que o usuario pediu.
     Um estudo sobre "encontrar o maior valor" nao sustenta uma recomendacao
     sobre "ver correlacao", por mais parecido que o texto pareca.
  2. Entre os que sobrarem, prefira `diferenca estatisticamente significativa=sim`
     sobre `nao-reportada`: sem significancia, o estudo nao demonstrou que uma
     opcao e realmente melhor que a outra.
  3. Prefira `evidencia=experimental` sobre `teorica`.
  4. Prefira quem comparou MAIS alternativas (`alternativas comparadas`).
  5. So entao use a similaridade como ultimo desempate.

A similaridade e o criterio MAIS FRACO de propósito: ela mede parecenca textual
com a pergunta, nao forca da evidencia. As similaridades costumam ficar muito
proximas entre si, entao nao trate uma diferenca pequena como significativa.

Diga SEMPRE, no campo "conflito", quais trechos discordaram e qual criterio acima
resolveu. Se nao houve discordancia, deixe "".

Escreva em portugues do Brasil, em linguagem simples, sem jargao academico.
Responda SEMPRE com um unico objeto JSON valido, sem cercas de codigo."""

USER_TMPL = """PERGUNTA DO USUARIO:
{pergunta}

TRECHOS RECUPERADOS DA BASE DE CONHECIMENTO:
{contexto}

Produza um objeto JSON com EXATAMENTE estas chaves:

{{
  "principal": {{
    "grafico": "nome popular do grafico recomendado",
    "justificativa": "2 a 4 frases explicando por que este grafico, citando os trechos (ex: 'segundo o trecho 2, ...'). Sem jargao.",
    "trechos_usados": [numeros dos trechos que sustentam esta escolha],
    "vegalite_spec": {{ spec Vega-Lite v5 completa e renderizavel }}
  }},
  "alternativa": {{
    "grafico": "nome popular de um segundo grafico plausivel",
    "justificativa": "2 a 3 frases dizendo em que situacao esta opcao seria preferivel a principal, citando os trechos.",
    "trechos_usados": [numeros dos trechos],
    "vegalite_spec": {{ spec Vega-Lite v5 completa e renderizavel }}
  }},
  "ressalva": "string: escreva aqui se os trechos recuperados nao cobrirem bem a pergunta, ou \\"\\" se a base sustenta bem a resposta.",
  "conflito": "string: se os trechos apontaram graficos diferentes, diga em 1-2 frases quais discordaram e qual criterio de desempate resolveu (ex: 'os trechos 2 e 4 indicavam graficos diferentes; prevaleceu o trecho 2 por ter diferenca estatisticamente significativa'). Deixe \\"\\" se nao houve discordancia.",
  "criterio_desempate": "string: qual criterio decidiu, exatamente um de: \\"tarefa\\", \\"significancia\\", \\"experimental\\", \\"n_alternativas\\", \\"similaridade\\", \\"sem conflito\\"."
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


def recomendar(pergunta: str, k: int = 6, use_hyde: bool = False, retriever=None):
    r = retriever or Retriever(k=k, use_hyde=use_hyde)
    hits = r.search(pergunta, k=k)

    if not hits:
        return {
            "pergunta": pergunta,
            "erro": "nenhum trecho recuperado -- o indice esta vazio? rode build_index.py",
        }

    contexto = format_context(hits)
    # temperatura 0: a recomendacao precisa ser reproduzivel entre execucoes
    llm = config.get_chat(temperature=0.0)
    resp = config.invoke_with_retry(
        llm,
        [("system", SYSTEM), ("human", USER_TMPL.format(pergunta=pergunta, contexto=contexto))],
    )
    obj = parse_json_object(config.text_of(resp))

    principal = _finalizar_opcao(obj.get("principal"), hits)
    alternativa = _finalizar_opcao(obj.get("alternativa"), hits)

    todas = []
    for opc in (principal, alternativa):
        for f in (opc or {}).get("fontes", []):
            if f not in todas:
                todas.append(f)

    return {
        "pergunta": pergunta,
        "principal": principal,
        "alternativa": alternativa,
        "ressalva": str(obj.get("ressalva") or "").strip(),
        "conflito": str(obj.get("conflito") or "").strip(),
        "criterio_desempate": str(obj.get("criterio_desempate") or "").strip(),
        "fontes": todas,
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
    print(f"\nPERGUNTA: {res['pergunta']}\n")
    for rotulo, chave in (("PRINCIPAL", "principal"), ("ALTERNATIVA", "alternativa")):
        opc = res.get(chave)
        if not opc:
            print(f"{rotulo}: (nao gerada)\n")
            continue
        flag = "" if opc["spec_valida"] else "  [SPEC INVALIDA: " + "; ".join(opc["spec_erros"]) + "]"
        print(f"{rotulo}: {opc['grafico']}{flag}")
        print(f"  {opc['justificativa']}")
        print(f"  fontes: {', '.join(opc['fontes']) or '(nenhuma citada)'}\n")
    if res.get("conflito"):
        print(f"CONFLITO: {res['conflito']}")
        print(f"  criterio que decidiu: {res.get('criterio_desempate') or '(nao informado)'}\n")
    if res.get("ressalva"):
        print(f"RESSALVA: {res['ressalva']}\n")
    print("TRECHOS RECUPERADOS:")
    for t in res["trechos"]:
        print(f"  [{t['n']}] {t['similaridade']:.3f}  {t['fonte']}  {t['titulo'][:70]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pergunta", nargs="+")
    ap.add_argument("-k", type=int, default=6)
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

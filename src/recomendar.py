"""ETAPA ONLINE (roda a cada pergunta): pergunta -> recomendacao de um grafico.

Passo 1. TRADUZIR a pergunta leiga para o vocabulario tecnico da base
         (tarefa analitica + tipos de dado). E aqui que se fecha o gap
         semantico: a pergunta passa a "falar a lingua" dos documentos.
Passo 2. BUSCAR os k achados mais parecidos (similaridade de cosseno).
Passo 3. GERAR: o LLM le os achados e recomenda UM grafico, citando as fontes.

    python src/recomendar.py "quero comparar as vendas de 5 categorias"
"""

import difflib
import json
import re
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

import config
import validate_spec
from indexar import TAREFAS, abrir_banco

K = 9  # quantos achados recuperar (abaixo disso, um unico artigo tende a dominar)


# --- Passo 1: traduzir a pergunta -------------------------------------------

PROMPT_TRADUCAO = """Traduza a pergunta de um usuario leigo para o vocabulario tecnico
de uma base de estudos sobre percepcao de graficos.

Pergunta: "{pergunta}"

DECISAO 1 -- a pergunta e sobre mostrar ou analisar DADOS num grafico?
Se NAO (ex.: escolher software, cores ou layout de um documento, duvidas
gerais), responda com tarefa "nenhuma" e pare aqui.

DECISAO 2 -- se SIM, escolha SEMPRE a tarefa mais proxima da lista abaixo,
mesmo que nenhuma encaixe perfeitamente (ex.: "ver a evolucao ao longo do
tempo" nao tem tarefa propria: escolha a mais proxima). Nunca use "nenhuma"
para uma pergunta sobre dados.
- aggregate: estimar media, total ou valor agregado
- characterize-distribution: ver como os valores se distribuem
- cluster: identificar grupos de itens parecidos
- correlate: ver a relacao entre duas variaveis
- determine-range: ver o intervalo (minimo e maximo)
- filter: selecionar itens que atendem a um criterio
- find-anomalies: achar valores fora do padrao
- find-extremum: achar o maior ou o menor valor
- retrieve-value: ler o valor exato de um item
- sort: ordenar ou comparar valores entre categorias

Tipos de dado envolvidos, entre: quantitative (numeros), nominal (categorias
sem ordem), ordinal (categorias com ordem, inclusive tempo: meses, anos, dias).

NAO sugira nenhum grafico: apenas classifique.
Responda so com JSON: {{"tarefa": "...", "tipos_de_dado": ["..."], "descricao": "uma frase em ingles descrevendo a analise em termos tecnicos, sem citar tipos de grafico"}}"""


def traduzir_pergunta(pergunta, llm):
    resposta = config.com_retentativa(lambda: llm.invoke(PROMPT_TRADUCAO.format(pergunta=pergunta)))
    traducao = ler_json(config.texto(resposta))
    # O LLM as vezes escreve um nome quase certo ("retrieving-value"):
    # troca pela tarefa valida mais parecida.
    tarefa = traducao.get("tarefa", "")
    if tarefa != "nenhuma" and tarefa not in TAREFAS:
        parecida = difflib.get_close_matches(tarefa, TAREFAS, n=1, cutoff=0)
        traducao["tarefa"] = parecida[0]
    return traducao


def consulta_tecnica(traducao):
    """Monta a consulta no MESMO formato dos documentos (ver indexar.py)."""
    return (
        f"task: {traducao['tarefa']}\n"
        f"data types: {', '.join(traducao.get('tipos_de_dado') or [])}\n"
        f"analysis: {traducao.get('descricao', '')}"
    )


# --- Passo 2: buscar ---------------------------------------------------------

def buscar(consulta, banco, k=K):
    """Os k achados mais parecidos. No Chroma (cosseno): similaridade = 1 - distancia."""
    return [(doc, 1 - dist) for doc, dist in banco.similarity_search_with_score(consulta, k=k)]


# --- Passo 3: gerar a recomendacao ------------------------------------------

PROMPT_RECOMENDACAO = """Voce ajuda pessoas que NAO sao especialistas a escolher um grafico.

PERGUNTA: {pergunta}

ACHADOS DE ESTUDOS CIENTIFICOS (sua unica fonte de conhecimento):
{achados}

Como ler os achados: "result, best to worst" ordena os designs do melhor para
o pior no estudo ('=' e empate). Cada design e descrito pela marca e pelo canal
visual: mark area-rect = barras; line = linhas; point = pontos (dispersao);
area-arc = pizza/rosca; area-circle = bolhas; text = tabela de texto;
length/position = comprimento/posicao; angle = angulo; color-hue = cor.

Recomende UM unico grafico, com base apenas nesses achados:
- de preferencia os achados cuja "task" combina com a pergunta e com
  "statistically significant differences: yes";
- se os achados discordarem, escolha o mais bem sustentado e diga por que.

Responda so com JSON:
{{
  "grafico": "nome popular do grafico, em portugues",
  "justificativa": "2 a 4 frases simples, citando os achados pelo numero (ex: 'o achado 2 mostrou que...')",
  "achados_usados": [numeros dos achados que sustentam a escolha],
  "vegalite_spec": {{ spec Vega-Lite v5 completa, com "$schema", "title", "data": {{"values": [4 a 8 linhas de exemplo coerentes com a pergunta]}}, "mark" e "encoding" com "type" em cada canal }},
  "ressalva": "limitacao importante, ou \\"\\" se nao houver"
}}"""


def referencia(fonte):
    """'Aigner2011bertin.json#...' -> 'Aigner (2011). Bertin was Right: ...'

    Autor e ano vem do nome do arquivo; o titulo, do proprio JSON da base.
    Nao precisa reindexar: e so para exibir.
    """
    arquivo = fonte.split("#")[0]
    autor, ano = re.match(r"([A-Za-z-]+?)(\d{4})", arquivo).groups()
    titulo = json.loads((config.RAW_DIR / arquivo).read_text(encoding="utf-8")).get("Title", "")
    return f"{autor} ({ano}). {titulo}"


def formatar_achados(resultados):
    return "\n\n".join(
        f"[ACHADO {i}] (fonte: {doc.metadata['fonte']})\n{doc.page_content}"
        for i, (doc, _) in enumerate(resultados, 1)
    )


def recomendar(pergunta, k=K, llm=None, banco=None):
    llm = llm or config.get_chat()
    banco = banco or abrir_banco()

    # Passo 1
    traducao = traduzir_pergunta(pergunta, llm)
    if traducao.get("tarefa") == "nenhuma":
        return {
            "pergunta": pergunta,
            "traducao": traducao,
            "fora_de_escopo": True,
            "mensagem": "Essa pergunta nao parece ser sobre qual grafico usar. "
                        "Descreva a analise que voce quer fazer com seus dados.",
        }

    # Passo 2
    consulta = consulta_tecnica(traducao)
    resultados = buscar(consulta, banco, k)

    # Passo 3
    prompt = PROMPT_RECOMENDACAO.format(pergunta=pergunta, achados=formatar_achados(resultados))
    resposta = ler_json(config.texto(config.com_retentativa(lambda: llm.invoke(prompt))))

    spec = resposta.get("vegalite_spec")
    validate_spec.ensure_inline_data(spec)
    spec, _ = validate_spec.repair(spec)
    spec_ok, _ = validate_spec.validate(spec)

    usados = [n for n in resposta.get("achados_usados") or [] if isinstance(n, int) and 1 <= n <= len(resultados)]
    return {
        "pergunta": pergunta,
        "traducao": traducao,
        "consulta": consulta,
        "grafico": resposta.get("grafico", ""),
        "justificativa": resposta.get("justificativa", ""),
        "ressalva": resposta.get("ressalva", ""),
        "vegalite_spec": spec,
        "spec_valida": spec_ok,
        "fontes": [resultados[n - 1][0].metadata["fonte"] for n in usados],
        "achados_usados": usados,
        "achados": [
            {"n": i, "fonte": d.metadata["fonte"], "artigo": referencia(d.metadata["fonte"]),
             "similaridade": round(s, 3), "texto": d.page_content}
            for i, (d, s) in enumerate(resultados, 1)
        ],
    }


def ler_json(texto):
    """Extrai o objeto JSON da resposta (o modelo as vezes o envolve em ```json)."""
    inicio, fim = texto.find("{"), texto.rfind("}")
    if inicio == -1:
        raise ValueError(f"resposta sem JSON: {texto[:200]}")
    return json.loads(texto[inicio : fim + 1])


if __name__ == "__main__":
    r = recomendar(" ".join(sys.argv[1:]) or "quero comparar as vendas de 5 categorias de produto")
    print(f"\nPERGUNTA: {r['pergunta']}")
    print(f"TRADUCAO: tarefa={r['traducao'].get('tarefa')} | dados={r['traducao'].get('tipos_de_dado')}")
    if r.get("fora_de_escopo"):
        print(f"\n{r['mensagem']}")
    else:
        print(f"\nRECOMENDACAO: {r['grafico']}")
        print(f"  {r['justificativa']}")
        if r["ressalva"]:
            print(f"  ressalva: {r['ressalva']}")
        print(f"  fontes: {', '.join(r['fontes'])}")
        print("\nACHADOS RECUPERADOS:")
        for a in r["achados"]:
            print(f"  [{a['n']}] {a['similaridade']:.3f}  {a['fonte']}")

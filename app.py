"""Interface do recomendador.

    streamlit run app.py
"""

import sys
from pathlib import Path
from urllib.parse import quote_plus

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import streamlit as st

from recomendar import recomendar

st.set_page_config(page_title="Qual gráfico usar?", page_icon="📊", layout="centered")

EXEMPLOS = [
    "Quero comparar as vendas de 5 categorias de produto",
    "Como mostrar a evolução da temperatura ao longo de 12 meses?",
    "Preciso ver se há relação entre horas de estudo e nota da prova",
    "Quero mostrar a distribuição das idades dos meus clientes",
    "Quero identificar os valores fora do padrão numa lista de preços",
]

# Os termos tecnicos da base, na lingua de quem pergunta (so para exibir).
TAREFAS_PT = {
    "aggregate": "estimar uma média ou um total",
    "characterize-distribution": "ver como os valores se distribuem",
    "cluster": "identificar grupos parecidos",
    "correlate": "ver a relação entre duas variáveis",
    "determine-range": "ver o intervalo entre o mínimo e o máximo",
    "filter": "selecionar itens por um critério",
    "find-anomalies": "encontrar valores fora do padrão",
    "find-extremum": "achar o maior ou o menor valor",
    "retrieve-value": "ler o valor exato de um item",
    "sort": "comparar ou ordenar categorias",
}
TIPOS_PT = {"quantitative": "números", "nominal": "categorias", "ordinal": "categorias ordenadas ou tempo"}


def link_artigo(referencia):
    """Referencia com link de busca do titulo no Google Scholar."""
    titulo = referencia.split(". ", 1)[-1]
    return f"[{referencia}](https://scholar.google.com/scholar?q={quote_plus(titulo)})"


def usar_exemplo():
    """Clicar num exemplo copia o texto para a caixa de pergunta."""
    st.session_state["pergunta"] = st.session_state["exemplo"]
    st.session_state["exemplo"] = None


def mostrar_erro(exc):
    if any(c in str(exc) for c in ("429", "RESOURCE_EXHAUSTED")):
        st.error("O limite de uso da API foi atingido. Espere um minuto e tente de novo.")
    else:
        st.error("Não foi possível gerar a recomendação. Tente de novo em instantes.")
    with st.expander("Detalhes do erro"):
        st.code(f"{type(exc).__name__}: {exc}")


# --- entrada -------------------------------------------------------------------

st.title("Qual gráfico usar?")
st.write(
    "Descreva, com suas palavras, o que você quer mostrar com seus dados. "
    "A recomendação vem de estudos científicos sobre como as pessoas leem gráficos."
)

pergunta = st.text_area("O que você quer mostrar?", key="pergunta", height=90,
                        placeholder="Ex.: quero comparar o faturamento de 5 lojas")
st.pills("Ou experimente um exemplo", EXEMPLOS, key="exemplo", on_change=usar_exemplo)

if not st.button("Recomendar gráfico", type="primary", disabled=not (pergunta or "").strip()):
    st.stop()

# --- resultado -----------------------------------------------------------------

try:
    with st.spinner("Entendendo a pergunta e consultando 240 resultados de 59 estudos..."):
        r = recomendar(pergunta.strip())
except Exception as exc:
    mostrar_erro(exc)
    st.stop()

t = r["traducao"]
if r.get("fora_de_escopo"):
    st.warning(r["mensagem"])
    st.stop()

tipos = ", ".join(TIPOS_PT.get(x, x) for x in t.get("tipos_de_dado") or []) or "não identificados"
st.caption(f"Entendi que você quer **{TAREFAS_PT.get(t['tarefa'], t['tarefa'])}**, com dados do tipo **{tipos}**.")

with st.container(border=True):
    st.subheader(r["grafico"][:1].upper() + r["grafico"][1:])
    if r["spec_valida"]:
        st.vega_lite_chart(r["vegalite_spec"], width="stretch")
    else:
        st.caption("Não foi possível desenhar o exemplo deste gráfico.")
    st.write(r["justificativa"])
    if r["ressalva"]:
        st.info(r["ressalva"], icon="💡")

if r["achados_usados"]:
    st.markdown("**Estudos que sustentam a recomendação**")
    for n in r["achados_usados"]:
        st.markdown(f"- **[{n}]** {link_artigo(r['achados'][n - 1]['artigo'])}")

with st.expander("Como o sistema chegou a essa recomendação"):
    st.markdown("**1. Pergunta traduzida para os termos da base**")
    st.code(r["consulta"], language=None)
    st.markdown(f"**2. Os {len(r['achados'])} resultados mais parecidos** (texto original, sem reescrita)")
    for a in r["achados"]:
        st.markdown(f"**[{a['n']}]** {link_artigo(a['artigo'])}  \n"
                    f"similaridade {a['similaridade']} · `{a['fonte']}`")
        st.text(a["texto"])
    st.markdown("**3. Especificação Vega-Lite do gráfico**")
    st.json(r["vegalite_spec"], expanded=False)

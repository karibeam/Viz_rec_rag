"""Interface do recomendador.

    streamlit run app.py
"""

import sys
from pathlib import Path
from urllib.parse import quote_plus

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import streamlit as st

from recomendar import recomendar

def link_artigo(referencia):
    """Referencia com link de busca do titulo no Google Scholar."""
    titulo = referencia.split(". ", 1)[-1]
    return f"[{referencia}](https://scholar.google.com/scholar?q={quote_plus(titulo)})"


st.set_page_config(page_title="Recomendador de Visualizacoes", page_icon="📊", layout="wide")

EXEMPLOS = [
    "Quero comparar as vendas de 5 categorias de produto",
    "Como mostrar a evolucao da temperatura ao longo de 12 meses?",
    "Preciso ver se ha relacao entre horas de estudo e nota da prova",
    "Quero mostrar a distribuicao das idades dos meus clientes",
    "Quero identificar os valores fora do padrao numa lista de precos",
]

st.title("📊 Recomendador de Visualizacoes")
st.caption(
    "Descreva sua analise em linguagem do dia a dia. A recomendacao vem de estudos "
    "cientificos sobre percepcao de graficos, recuperados por RAG."
)

with st.sidebar:
    st.subheader("Exemplos")
    for exemplo in EXEMPLOS:
        if st.button(exemplo, use_container_width=True):
            st.session_state["pergunta"] = exemplo

# `key` guarda o texto entre recarregamentos da pagina (e os exemplos escrevem nele)
pergunta = st.text_area("Sua pergunta", key="pergunta", height=90,
                        placeholder="Ex: quero comparar o faturamento de 5 lojas")

if st.button("Recomendar", type="primary", disabled=not (pergunta or "").strip()):
    try:
        with st.spinner("Buscando nos estudos..."):
            r = recomendar(pergunta.strip())
    except Exception as exc:
        st.error(f"{type(exc).__name__}: {exc}")
        st.stop()

    t = r["traducao"]
    st.caption(f"Como o sistema entendeu: tarefa **{t.get('tarefa')}** · dados **{', '.join(t.get('tipos_de_dado') or []) or '—'}**")

    if r.get("fora_de_escopo"):
        st.warning(r["mensagem"])
        st.stop()

    st.markdown(f"### Recomendação: {r['grafico']}")
    st.write(r["justificativa"])
    if r["ressalva"]:
        st.info(r["ressalva"])

    if r["spec_valida"]:
        st.vega_lite_chart(r["vegalite_spec"], use_container_width=True)
    else:
        st.warning("Não foi possível desenhar o gráfico de exemplo.")

    if r["achados_usados"]:
        st.markdown("**Estudos que sustentam a recomendação**")
        for n in r["achados_usados"]:
            st.markdown(f"- **[{n}]** {link_artigo(r['achados'][n - 1]['artigo'])}")

    with st.expander(f"Achados recuperados da base ({len(r['achados'])}) — texto original, sem reescrita"):
        st.caption(f"Consulta usada na busca:\n\n{r['consulta']}")
        for a in r["achados"]:
            st.markdown(f"**[{a['n']}]** {link_artigo(a['artigo'])}  \n"
                        f"similaridade {a['similaridade']} · `{a['fonte']}`")
            st.text(a["texto"])

    with st.expander("Spec Vega-Lite"):
        st.json(r["vegalite_spec"])

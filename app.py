"""Interface Streamlit do MVP.

    streamlit run app.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import streamlit as st

import config
from recommend import recomendar
from retrieve import Retriever

st.set_page_config(page_title="Recomendador de Visualizacoes", page_icon="📊", layout="wide")

EXEMPLOS = [
    "Quero comparar as vendas de 5 categorias de produto",
    "Como mostrar a evolucao da temperatura ao longo de 12 meses?",
    "Preciso ver se ha relacao entre horas de estudo e nota da prova",
    "Qual grafico usa para mostrar a participacao de cada regiao no total?",
    "Quero identificar os valores fora do padrao numa lista de precos",
]


@st.cache_resource(show_spinner=False)
def get_retriever(k: int, use_hyde: bool):
    return Retriever(k=k, use_hyde=use_hyde)


def render_opcao(rotulo: str, opc: dict, cor: str):
    if not opc:
        st.warning(f"{rotulo}: nao gerada.")
        return
    st.markdown(f"### :{cor}[{rotulo}] — {opc['grafico']}")
    st.write(opc["justificativa"])

    spec = opc.get("vegalite_spec")
    if opc.get("spec_valida") and isinstance(spec, dict) and spec.get("data", {}).get("values"):
        st.vega_lite_chart(spec, use_container_width=True)
    else:
        motivo = "; ".join(opc.get("spec_erros") or ["spec sem dados de exemplo"])
        st.warning(f"Nao foi possivel renderizar o grafico ({motivo}).")

    if opc.get("fontes"):
        st.caption("Fontes: " + " · ".join(opc["fontes"]))
    if opc.get("spec_reparos"):
        st.caption("Tipos preenchidos automaticamente: " + ", ".join(opc["spec_reparos"]))
    with st.expander("Ver spec Vega-Lite"):
        st.json(spec)


st.title("📊 Recomendador de Visualizacoes")
st.caption(
    "Descreva sua analise em linguagem do dia a dia. A recomendacao vem de estudos "
    "empiricos de percepcao grafica recuperados por RAG — nao de regras fixas."
)

with st.sidebar:
    st.subheader("Configuracao")
    st.text(f"provedor: {config.provider()}")
    with st.expander("Avançado"):
        # Piso 6, e nao 3: os trechos sao ACHADOS, nao artigos, e um mesmo artigo
        # rende varios. Medido nas 15 perguntas, com k=3 em 10 delas metade ou
        # mais dos trechos vinha de um unico artigo -- o placar deixava de ser
        # uma votacao entre estudos independentes. Em k=9 isso cai para 2 de 15.
        k = st.slider(
            "Trechos recuperados (k)", 6, 12, 9,
            help="Quantos achados entram no placar. Abaixo de 9, um único artigo tende a dominar a votação.",
        )
        use_hyde = st.checkbox(
            "Usar HyDE", value=False,
            help="Expande a pergunta antes de buscar. Medido: reduz a estabilidade de 89% para ~50% e dobra o consumo de cota.",
        )
    # if/else de verdade, nao expressao condicional solta: o "magic" do Streamlit
    # auto-exibe expressoes de nivel superior e quebra ao tentar renderizar o
    # DeltaGenerator devolvido por st.success/st.error.
    if config.CHROMA_DIR.exists():
        st.success("Indice encontrado")
    else:
        st.error("Indice ausente. Rode: python src/enrich.py && python src/build_index.py")
    st.divider()
    st.caption("Exemplos")
    for ex in EXEMPLOS:
        if st.button(ex, use_container_width=True):
            st.session_state["pergunta"] = ex

# `key` em vez de `value`: sem a chave, o campo perde o conteudo a cada rerun
# (clicar em "Recomendar" apagava a pergunta antes de processa-la). Com a chave,
# o valor vive em st.session_state e os botoes de exemplo escrevem nele direto.
pergunta = st.text_area(
    "Sua pergunta",
    key="pergunta",
    placeholder="Ex: quero comparar o faturamento de 5 lojas no ultimo trimestre",
    height=90,
)

if st.button("Recomendar", type="primary", disabled=not (pergunta or "").strip()):
    try:
        with st.spinner("Buscando na base e montando a recomendacao..."):
            res = recomendar(pergunta.strip(), k=k, retriever=get_retriever(k, use_hyde))
    except Exception as exc:
        st.error(f"{type(exc).__name__}: {exc}")
    else:
        if res.get("erro"):
            st.error(res["erro"])
        elif res.get("fora_de_escopo"):
            st.warning(f"**Fora do escopo da base.** {res['ressalva']}")
        else:
            if res.get("ressalva"):
                st.info(f"⚠️ {res['ressalva']}")
            if res.get("conflito"):
                st.warning(f"**Por que este e não outro.** {res['conflito']}")

            render_opcao("RECOMENDAÇÃO", res.get("recomendacao"), "green")

            st.divider()
            placar = res.get("placar")
            if placar:
                tarefa = placar.get("tarefa_identificada") or "não identificada"

                # Quantos ARTIGOS distintos sustentam a decisao. Mais informativo
                # que o k para quem nao e especialista: k conta achados, e um
                # mesmo artigo rende varios deles.
                artigos = {t["fonte"].split("#")[0] for t in res["trechos"]}
                st.caption(
                    f"Baseado em **{len(artigos)} estudos independentes** "
                    f"({len(res['trechos'])} achados recuperados) · tarefa identificada: `{tarefa}`"
                )

                with st.expander(f"Como o sistema decidiu — placar (tarefa: {tarefa})", expanded=True):
                    st.caption(
                        "A escolha é calculada em Python, não pelo modelo. Cada trecho recuperado "
                        "vota no gráfico que venceu no seu estudo, com peso derivado da força "
                        "daquela evidência. Só o 1º colocado é recomendado; os demais aparecem "
                        "aqui para você conferir a decisão."
                    )
                    st.dataframe(
                        [
                            {
                                "#": c["posicao"],
                                "gráfico": c["nome"],
                                "pontos": c["pontos"],
                                "trechos a favor": ", ".join(str(a["trecho"]) for a in c["a_favor"]) or "—",
                                "trechos contra": ", ".join(str(a["trecho"]) for a in c["contra"]) or "—",
                            }
                            for c in placar["ranking"]
                        ],
                        hide_index=True,
                        use_container_width=True,
                    )
                    vencedor = placar.get("vencedor") or {}
                    if vencedor.get("a_favor"):
                        st.caption("Memória de cálculo do 1º colocado:")
                        for a in vencedor["a_favor"]:
                            fatores = " × ".join(f"{k}={v}" for k, v in a["fatores"].items())
                            st.text(f"  trecho {a['trecho']}:  {fatores}  =  {a['peso']}")

            with st.expander(f"Trechos recuperados da base ({len(res['trechos'])})"):
                for t in res["trechos"]:
                    st.markdown(
                        f"**[{t['n']}]** `{t['fonte']}` — similaridade **{t['similaridade']:.3f}**"
                    )
                    st.text(t["texto"])
                    st.divider()
            with st.expander("Resposta completa (JSON)"):
                st.json(res)

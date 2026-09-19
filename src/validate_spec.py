"""Validacao das specs Vega-Lite geradas pelo LLM.

Impede que uma spec quebrada chegue na interface. Usa o schema embutido no
altair (funciona offline); se o altair nao estiver disponivel, cai para
verificacoes estruturais minimas.
"""


def _structural_check(spec):
    if not isinstance(spec, dict):
        return ["spec nao e um objeto JSON"]
    erros = []
    if "mark" not in spec and "layer" not in spec and "hconcat" not in spec:
        erros.append("spec sem 'mark' nem 'layer'")
    enc = spec.get("encoding")
    if enc is not None and not isinstance(enc, dict):
        erros.append("'encoding' deve ser um objeto")
    if isinstance(enc, dict):
        for canal, definicao in enc.items():
            if isinstance(definicao, dict) and "type" not in definicao and "value" not in definicao:
                erros.append(f"encoding.{canal} sem 'type'")
    return erros


def _infere_tipo(valores):
    """Deduz o tipo Vega-Lite a partir dos valores de exemplo da propria spec.

    Inferencia sobre a forma do dado gerado, nao sobre qual grafico usar --
    nao introduz nenhuma regra de recomendacao.
    """
    amostra = [v for v in valores if v is not None]
    if not amostra:
        return "nominal"
    if all(isinstance(v, bool) for v in amostra):
        return "nominal"
    if all(isinstance(v, (int, float)) for v in amostra):
        return "quantitative"
    return "nominal"


def repair(spec):
    """Preenche `type` faltante nos encodings, usando os dados embutidos.

    O LLM costuma esquecer o `type` em canais secundarios (xOffset, color,
    size). Como a spec ja traz `data.values`, da para completar sem chutar --
    e barato demais para descartar uma recomendacao boa por isso.
    """
    if not isinstance(spec, dict):
        return spec, []
    enc = spec.get("encoding")
    if not isinstance(enc, dict):
        return spec, []

    linhas = ((spec.get("data") or {}).get("values")) or []
    corrigidos = []
    for canal, definicao in enc.items():
        if not isinstance(definicao, dict):
            continue
        # Categorias ordenadas (meses, faixas) mantem a ordem em que o LLM
        # escreveu os dados; sem isso o Vega-Lite ordena alfabeticamente
        # (Jan, Jul, Mai, Mar...).
        if definicao.get("type") == "ordinal" and "sort" not in definicao:
            definicao["sort"] = None
            corrigidos.append(f"{canal}: ordem dos dados")
        if "type" in definicao or "value" in definicao:
            continue
        campo = definicao.get("field")
        if not campo:
            continue
        valores = [l.get(campo) for l in linhas if isinstance(l, dict)]
        definicao["type"] = _infere_tipo(valores)
        corrigidos.append(f"{canal}={definicao['type']}")
    return spec, corrigidos


def validate(spec):
    """Retorna (valido: bool, erros: list[str])."""
    erros = _structural_check(spec)
    if erros:
        return False, erros

    try:
        import altair as alt
    except ImportError:
        return True, []

    try:
        alt.Chart.from_dict(spec, validate=True)
    except Exception as exc:
        msg = str(exc).splitlines()[0][:300]
        return False, [f"schema Vega-Lite: {msg}"]
    return True, []


def ensure_inline_data(spec, dados=None):
    """Garante que a spec tenha dados para renderizar.

    O LLM recebe instrucao de embutir um pequeno exemplo em `data.values`; se
    ainda assim vier sem dados, marcamos com um placeholder nomeado para que a
    interface consiga avisar em vez de renderizar um grafico vazio.
    """
    if not isinstance(spec, dict):
        return spec
    if dados:
        spec["data"] = {"values": dados}
    elif "data" not in spec:
        spec["data"] = {"values": []}
    return spec


if __name__ == "__main__":
    import json
    import sys

    spec = json.load(open(sys.argv[1]) if len(sys.argv) > 1 else sys.stdin)
    ok, erros = validate(spec)
    print("VALIDA" if ok else "INVALIDA")
    for e in erros:
        print(" -", e)

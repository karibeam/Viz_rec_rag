"""Configuracao central: provedor de LLM/embeddings e caminhos do projeto."""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
CARDS_PATH = PROCESSED_DIR / "cards.jsonl"
CHROMA_DIR = ROOT / "data" / "chroma"
COLLECTION = "graphical_perception"

load_dotenv(ROOT / ".env")


def provider() -> str:
    return os.getenv("LLM_PROVIDER", "gemini").strip().lower()


def _require(var: str) -> str:
    val = os.getenv(var, "").strip()
    if not val:
        raise RuntimeError(
            f"{var} nao definida. Copie .env.example para .env e preencha a chave."
        )
    return val


def text_of(resposta) -> str:
    """Extrai o texto de uma resposta de chat.

    Modelos mais novos do Gemini devolvem `content` como lista de partes
    (texto + assinaturas de raciocinio) em vez de string. Normalizar aqui evita
    que cada modulo tenha que lidar com os dois formatos.
    """
    content = getattr(resposta, "content", resposta)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        partes = []
        for parte in content:
            if isinstance(parte, str):
                partes.append(parte)
            elif isinstance(parte, dict) and parte.get("type") == "text":
                partes.append(str(parte.get("text", "")))
        return "".join(partes)
    return str(content)


def invoke_with_retry(llm, entrada, tentativas: int = 4, espera_padrao: int = 30):
    """Chama o modelo tolerando 429 transitorio (limite por minuto).

    O free tier limita requisicoes por minuto, e uma pergunta do usuario custa
    duas chamadas (HyDE + geracao). Sem isso, um pico momentaneo derruba a
    pergunta inteira. Cota DIARIA esgotada nao e recuperavel por espera: nesse
    caso a excecao sobe para quem chamou.
    """
    import re
    import time

    for tentativa in range(1, tentativas + 1):
        try:
            return llm.invoke(entrada)
        except Exception as exc:
            msg = str(exc)
            transitorio = "RESOURCE_EXHAUSTED" in msg or "429" in msg
            diario = "PerDay" in msg
            if not transitorio or diario or tentativa == tentativas:
                raise
            m = re.search(r"retry in ([\d.]+)s", msg, re.IGNORECASE)
            time.sleep(int(float(m.group(1))) + 2 if m else espera_padrao * tentativa)
    raise RuntimeError("inalcancavel")


def get_chat(temperature: float = 0.0, model: str = None, timeout: int = 180):
    """Modelo de chat do provedor configurado.

    `model` sobrescreve o default do .env. Serve para usar um modelo mais leve
    no enriquecimento em massa (dezenas de chamadas) e um melhor no runtime
    (uma chamada por pergunta do usuario).

    `timeout` evita que uma chamada travada segure o pipeline indefinidamente.
    """
    if provider() == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model or os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
            temperature=temperature,
            api_key=_require("OPENAI_API_KEY"),
            timeout=timeout,
            max_retries=3,
        )

    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=model or os.getenv("GEMINI_CHAT_MODEL", "gemini-3.5-flash-lite"),
        temperature=temperature,
        google_api_key=_require("GOOGLE_API_KEY"),
        timeout=timeout,
        max_retries=3,
    )


def get_embeddings():
    """Modelo de embeddings do provedor configurado (ambos sao multilingues)."""
    if provider() == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small"),
            api_key=_require("OPENAI_API_KEY"),
        )

    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    return GoogleGenerativeAIEmbeddings(
        model=os.getenv("GEMINI_EMBED_MODEL", "models/gemini-embedding-001"),
        google_api_key=_require("GOOGLE_API_KEY"),
    )

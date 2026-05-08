import os
from pathlib import Path

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
KNOWLEDGE_BASE_PATH = DATA_DIR / "knowledge_base.txt"
VINFAST_MARKDOWN_DIR = ROOT_DIR / "vinfast_markdown_clean"


def load_settings() -> None:
    load_dotenv(ROOT_DIR / ".env")

    langsmith_key = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
    if langsmith_key:
        os.environ["LANGCHAIN_API_KEY"] = langsmith_key

    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_PROJECT", "day22-vinfast-rag-local")
    os.environ.setdefault("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")


load_settings()


def get_env(name: str, default: str = "") -> str:
    return os.getenv(name, default)


def get_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def get_embeddings():
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(
        model=get_env("EMBEDDING_MODEL", "text-embedding-qwen3-embedding-4b"),
        api_key=get_env("EMBEDDING_API_KEY", "lm-studio"),
        base_url=get_env("EMBEDDING_BASE_URL", get_env("LM_STUDIO_BASE_URL", "http://127.0.0.1:1234/v1")),
        check_embedding_ctx_length=False,
    )


def get_llm(temperature: float = 0.1):
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=get_env("LLM_MODEL", "gpt-4o-mini"),
        api_key=get_env("LLM_API_KEY", get_env("SHOPAIKEY_API_KEY")),
        base_url=get_env("LLM_BASE_URL", get_env("SHOPAIKEY_BASE_URL", "https://api.shopaikey.com/v1")),
        temperature=temperature,
    )


def load_knowledge_base() -> str:
    if not KNOWLEDGE_BASE_PATH.exists():
        raise FileNotFoundError(
            f"Missing {KNOWLEDGE_BASE_PATH}. Create it or run the workspace setup."
        )
    return KNOWLEDGE_BASE_PATH.read_text(encoding="utf-8")


def print_config() -> None:
    print("Config loaded successfully")
    print(f"   LangSmith project : {get_env('LANGCHAIN_PROJECT', 'day22-vinfast-rag-local')}")
    print(f"   LangSmith tracing : {get_env('LANGCHAIN_TRACING_V2', 'true')}")
    print(f"   LLM endpoint      : {get_env('LLM_BASE_URL', get_env('LM_STUDIO_BASE_URL'))}")
    print(f"   Default LLM model : {get_env('LLM_MODEL', 'gpt-4o-mini')}")
    print(f"   Embedding endpoint: {get_env('EMBEDDING_BASE_URL', get_env('LM_STUDIO_BASE_URL'))}")
    print(f"   Embedding model   : {get_env('EMBEDDING_MODEL', 'text-embedding-qwen3-embedding-4b')}")
    print(f"   Knowledge base    : {KNOWLEDGE_BASE_PATH}")


if __name__ == "__main__":
    print_config()

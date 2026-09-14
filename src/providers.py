from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_postgres import PGVector
from pydantic import SecretStr

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OPENAI = "openai"
GOOGLE = "google"
SUPPORTED_PROVIDERS = (OPENAI, GOOGLE)

DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_OPENAI_CHAT_MODEL = "gpt-4o-mini"
DEFAULT_GOOGLE_EMBEDDING_MODEL = "models/gemini-embedding-001"
DEFAULT_GOOGLE_CHAT_MODEL = "gemini-2.5-flash-lite"
DEFAULT_COLLECTION_NAME = "documentos"


class ConfigurationError(RuntimeError):
    pass


def required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ConfigurationError(
            f"Variável de ambiente obrigatória ausente ou vazia: {name}. "
            "Copie .env.example para .env e preencha os valores."
        )
    return value


def optional_env(name: str, default: str) -> str:
    return os.getenv(name, "").strip() or default


def resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def selected_provider() -> str:
    provider = optional_env("LLM_PROVIDER", OPENAI).lower()
    if provider not in SUPPORTED_PROVIDERS:
        raise ConfigurationError(
            f"LLM_PROVIDER inválido: {provider!r}. Use um de {', '.join(SUPPORTED_PROVIDERS)}."
        )
    return provider


def build_embeddings() -> Embeddings:
    provider = selected_provider()
    if provider == OPENAI:
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=optional_env("OPENAI_EMBEDDING_MODEL", DEFAULT_OPENAI_EMBEDDING_MODEL),
            api_key=SecretStr(required_env("OPENAI_API_KEY")),
        )

    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    return GoogleGenerativeAIEmbeddings(
        model=optional_env("GOOGLE_EMBEDDING_MODEL", DEFAULT_GOOGLE_EMBEDDING_MODEL),
        google_api_key=SecretStr(required_env("GOOGLE_API_KEY")),
    )


def build_llm() -> BaseChatModel:
    provider = selected_provider()
    if provider == OPENAI:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=optional_env("OPENAI_CHAT_MODEL", DEFAULT_OPENAI_CHAT_MODEL),
            temperature=0,
            api_key=SecretStr(required_env("OPENAI_API_KEY")),
        )

    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=optional_env("GOOGLE_CHAT_MODEL", DEFAULT_GOOGLE_CHAT_MODEL),
        temperature=0,
        google_api_key=SecretStr(required_env("GOOGLE_API_KEY")),
    )


def build_vector_store() -> PGVector:
    return PGVector(
        embeddings=build_embeddings(),
        collection_name=optional_env("PG_VECTOR_COLLECTION_NAME", DEFAULT_COLLECTION_NAME),
        connection=required_env("DATABASE_URL"),
        use_jsonb=True,
    )

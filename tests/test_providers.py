from __future__ import annotations

import pytest

import providers
from providers import (
    ConfigurationError,
    build_embeddings,
    build_llm,
    optional_env,
    required_env,
    resolve_path,
    selected_provider,
)


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "LLM_PROVIDER",
        "OPENAI_API_KEY",
        "OPENAI_EMBEDDING_MODEL",
        "OPENAI_CHAT_MODEL",
        "GOOGLE_API_KEY",
        "GOOGLE_EMBEDDING_MODEL",
        "GOOGLE_CHAT_MODEL",
        "DATABASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)


def test_openai_is_the_default_provider() -> None:
    assert selected_provider() == "openai"


def test_provider_name_is_case_insensitive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "Google")

    assert selected_provider() == "google"


def test_unknown_provider_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "claude")

    with pytest.raises(ConfigurationError, match="LLM_PROVIDER"):
        selected_provider()


def test_blank_variable_counts_as_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "   ")

    with pytest.raises(ConfigurationError, match="DATABASE_URL"):
        required_env("DATABASE_URL")


def test_optional_variable_falls_back_to_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    assert optional_env("OPENAI_CHAT_MODEL", "gpt-4o-mini") == "gpt-4o-mini"

    monkeypatch.setenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini")
    assert optional_env("OPENAI_CHAT_MODEL", "gpt-4o-mini") == "gpt-4.1-mini"


def test_relative_pdf_path_is_resolved_from_the_project_root() -> None:
    assert resolve_path("document.pdf") == providers.PROJECT_ROOT / "document.pdf"
    assert resolve_path("/tmp/outro.pdf").as_posix() == "/tmp/outro.pdf"


def test_missing_api_key_fails_before_any_call_is_made() -> None:
    with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
        build_embeddings()

    with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
        build_llm()


def test_google_provider_requires_the_google_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "google")

    with pytest.raises(ConfigurationError, match="GOOGLE_API_KEY"):
        build_embeddings()


def test_each_provider_builds_its_own_clients(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "chave-de-teste")
    assert type(build_embeddings()).__name__ == "OpenAIEmbeddings"
    assert type(build_llm()).__name__ == "ChatOpenAI"

    monkeypatch.setenv("LLM_PROVIDER", "google")
    monkeypatch.setenv("GOOGLE_API_KEY", "chave-de-teste")
    assert type(build_embeddings()).__name__ == "GoogleGenerativeAIEmbeddings"
    assert type(build_llm()).__name__ == "ChatGoogleGenerativeAI"

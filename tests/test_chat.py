from __future__ import annotations

from collections.abc import Iterator

import httpx
import pytest
from google.api_core.exceptions import ResourceExhausted
from langchain_google_genai._common import GoogleGenerativeAIError
from openai import RateLimitError
from sqlalchemy.exc import OperationalError

import chat
from providers import ConfigurationError


class EchoChain:
    def __init__(self) -> None:
        self.questions: list[str] = []

    def invoke(self, question: str) -> str:
        self.questions.append(question)
        return "O faturamento foi de 10 milhões de reais."


def answer_with(monkeypatch: pytest.MonkeyPatch, inputs: list[str]) -> EchoChain:
    chain = EchoChain()
    pending: Iterator[str] = iter(inputs)
    monkeypatch.setattr(chat, "search_prompt", lambda: chain)
    monkeypatch.setattr("builtins.input", lambda _: next(pending))
    return chain


def test_question_is_answered_and_printed(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    chain = answer_with(monkeypatch, ["Qual o faturamento da SuperTechIABrazil?", "sair"])

    chat.main()

    assert chain.questions == ["Qual o faturamento da SuperTechIABrazil?"]
    assert "RESPOSTA: O faturamento foi de 10 milhões de reais." in capsys.readouterr().out


def test_blank_lines_do_not_reach_the_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    chain = answer_with(monkeypatch, ["", "   ", "sair"])

    chat.main()

    assert chain.questions == []


def test_end_of_input_closes_the_chat(monkeypatch: pytest.MonkeyPatch) -> None:
    chain = EchoChain()
    monkeypatch.setattr(chat, "search_prompt", lambda: chain)
    monkeypatch.setattr("builtins.input", lambda _: (_ for _ in ()).throw(EOFError()))

    chat.main()

    assert chain.questions == []


def test_missing_configuration_stops_the_cli_with_an_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def explode() -> None:
        raise ConfigurationError(
            "Variável de ambiente obrigatória ausente ou vazia: OPENAI_API_KEY"
        )

    monkeypatch.setattr(chat, "search_prompt", explode)

    with pytest.raises(SystemExit) as exit_info:
        chat.main()

    assert exit_info.value.code == 1
    assert "OPENAI_API_KEY" in capsys.readouterr().err


class FailingOnceChain:
    def __init__(self, failure: Exception) -> None:
        self.failure = failure
        self.questions: list[str] = []

    def invoke(self, question: str) -> str:
        self.questions.append(question)
        if len(self.questions) == 1:
            raise self.failure
        return "O faturamento foi de 10 milhões de reais."


def rate_limit_error() -> Exception:
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    return RateLimitError(
        "limite atingido", response=httpx.Response(429, request=request), body=None
    )


@pytest.mark.parametrize(
    "failure",
    [
        pytest.param(rate_limit_error(), id="openai"),
        pytest.param(ResourceExhausted("cota esgotada"), id="gemini-llm"),
        pytest.param(GoogleGenerativeAIError("Error embedding content"), id="gemini-embeddings"),
    ],
)
def test_provider_failure_is_reported_and_the_chat_keeps_running(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], failure: Exception
) -> None:
    chain = FailingOnceChain(failure)
    pending: Iterator[str] = iter(["primeira", "segunda", "sair"])
    monkeypatch.setattr(chat, "search_prompt", lambda: chain)
    monkeypatch.setattr("builtins.input", lambda _: next(pending))

    chat.main()

    output = capsys.readouterr()
    assert chain.questions == ["primeira", "segunda"]
    assert "Não foi possível consultar o provedor" in output.err
    assert output.out.count("RESPOSTA:") == 1


def database_down() -> OperationalError:
    return OperationalError("SELECT 1", {}, ConnectionRefusedError("recusada"))


def test_database_down_mid_chat_is_reported_without_leaking_the_url(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    chain = FailingOnceChain(database_down())
    pending: Iterator[str] = iter(["primeira", "sair"])
    monkeypatch.setattr(chat, "search_prompt", lambda: chain)
    monkeypatch.setattr("builtins.input", lambda _: next(pending))

    chat.main()

    error_output = capsys.readouterr().err
    assert "docker compose up -d" in error_output
    assert "postgres:postgres" not in error_output


def test_database_down_at_startup_stops_the_cli(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def refuse() -> None:
        raise database_down()

    monkeypatch.setattr(chat, "search_prompt", refuse)

    with pytest.raises(SystemExit) as exit_info:
        chat.main()

    assert exit_info.value.code == 1
    assert "docker compose up -d" in capsys.readouterr().err

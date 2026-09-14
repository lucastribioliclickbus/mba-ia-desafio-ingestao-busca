from __future__ import annotations

from collections.abc import Iterator

import pytest

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

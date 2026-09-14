from __future__ import annotations

from langchain_core.documents import Document

from search import PROMPT_TEMPLATE, SEARCH_RESULTS, format_context, retrieve_context

REFUSAL = "Não tenho informações necessárias para responder sua pergunta."


class RecordingStore:
    def __init__(self, results: list[tuple[Document, float]]) -> None:
        self.results = results
        self.calls: list[tuple[str, int]] = []

    def similarity_search_with_score(self, query: str, k: int = 4) -> list[tuple[Document, float]]:
        self.calls.append((query, k))
        return self.results


def test_context_joins_every_result_without_the_scores() -> None:
    results = [
        (Document(page_content="  primeiro trecho  "), 0.1),
        (Document(page_content="segundo trecho"), 0.2),
    ]

    assert format_context(results) == "primeiro trecho\n\nsegundo trecho"


def test_context_is_empty_when_the_database_returns_nothing() -> None:
    assert format_context([]) == ""


def test_retrieval_asks_the_database_for_ten_results() -> None:
    store = RecordingStore([(Document(page_content="faturamento de 10 milhões"), 0.05)])

    context = retrieve_context(store, "Qual o faturamento?")

    assert store.calls == [("Qual o faturamento?", 10)]
    assert SEARCH_RESULTS == 10
    assert context == "faturamento de 10 milhões"


def test_prompt_carries_the_rules_and_both_placeholders() -> None:
    assert "{contexto}" in PROMPT_TEMPLATE
    assert "{pergunta}" in PROMPT_TEMPLATE
    assert "Responda somente com base no CONTEXTO." in PROMPT_TEMPLATE
    assert REFUSAL in PROMPT_TEMPLATE

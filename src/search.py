from __future__ import annotations

from typing import Protocol

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda, RunnableParallel, RunnablePassthrough

from providers import build_llm, build_vector_store

SEARCH_RESULTS = 10

PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""


class SemanticSearch(Protocol):
    def similarity_search_with_score(
        self, query: str, k: int = ...
    ) -> list[tuple[Document, float]]: ...


def format_context(results: list[tuple[Document, float]]) -> str:
    return "\n\n".join(document.page_content.strip() for document, _ in results)


def retrieve_context(store: SemanticSearch, question: str) -> str:
    return format_context(store.similarity_search_with_score(question, k=SEARCH_RESULTS))


def search_prompt() -> Runnable[str, str]:
    store = build_vector_store()
    prompt = PromptTemplate(input_variables=["contexto", "pergunta"], template=PROMPT_TEMPLATE)

    return (
        RunnableParallel(
            contexto=RunnableLambda(lambda question: retrieve_context(store, question)),
            pergunta=RunnablePassthrough(),
        )
        | prompt
        | build_llm()
        | StrOutputParser()
    )

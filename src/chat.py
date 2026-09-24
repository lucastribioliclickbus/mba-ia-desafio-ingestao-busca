from __future__ import annotations

import sys

from google.api_core.exceptions import GoogleAPIError
from langchain_google_genai._common import GoogleGenerativeAIError
from openai import APIError
from sqlalchemy.exc import OperationalError

from providers import ConfigurationError
from search import search_prompt

EXIT_COMMANDS = frozenset({"sair", "exit", "quit"})
PROVIDER_ERRORS = (APIError, GoogleAPIError, GoogleGenerativeAIError)
DATABASE_UNAVAILABLE = (
    "Banco de dados indisponível. Suba o Postgres com 'docker compose up -d' "
    "e confira DATABASE_URL no .env."
)


def main() -> None:
    try:
        chain = search_prompt()
    except ConfigurationError as error:
        print(f"Configuração inválida: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    except OperationalError as error:
        print(DATABASE_UNAVAILABLE, file=sys.stderr)
        raise SystemExit(1) from error

    print("Faça sua pergunta sobre o documento (digite 'sair' para encerrar).")

    while True:
        try:
            question = input("\nPERGUNTA: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not question:
            continue
        if question.lower() in EXIT_COMMANDS:
            break

        try:
            answer = chain.invoke(question)
        except PROVIDER_ERRORS as error:
            print(f"Não foi possível consultar o provedor: {error}", file=sys.stderr)
            continue
        except OperationalError:
            print(DATABASE_UNAVAILABLE, file=sys.stderr)
            continue

        print(f"RESPOSTA: {answer}")


if __name__ == "__main__":
    main()

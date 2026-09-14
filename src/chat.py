from __future__ import annotations

import sys

from providers import ConfigurationError
from search import search_prompt

EXIT_COMMANDS = frozenset({"sair", "exit", "quit"})


def main() -> None:
    try:
        chain = search_prompt()
    except ConfigurationError as error:
        print(f"Configuração inválida: {error}", file=sys.stderr)
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

        print(f"RESPOSTA: {chain.invoke(question)}")


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from providers import build_vector_store, required_env, resolve_path, selected_provider

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def build_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=False,
    )


def load_chunks(pdf_path: Path) -> list[Document]:
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF não encontrado em {pdf_path}")
    pages = PyPDFLoader(str(pdf_path)).load()
    return build_splitter().split_documents(pages)


def without_empty_metadata(chunks: list[Document]) -> list[Document]:
    return [
        Document(
            page_content=chunk.page_content,
            metadata={
                key: value for key, value in chunk.metadata.items() if value not in ("", None)
            },
        )
        for chunk in chunks
    ]


def chunk_ids(source: str, total: int) -> list[str]:
    return [f"{source}:{index}" for index in range(total)]


def ingest_pdf() -> None:
    pdf_path = resolve_path(required_env("PDF_PATH"))
    chunks = without_empty_metadata(load_chunks(pdf_path))
    if not chunks:
        raise ValueError(f"Nenhum texto extraído de {pdf_path}")

    store = build_vector_store()
    store.add_documents(documents=chunks, ids=chunk_ids(pdf_path.name, len(chunks)))

    print(
        f"Ingestão concluída: {len(chunks)} chunks de {pdf_path.name} "
        f"(provider={selected_provider()}, chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})."
    )


if __name__ == "__main__":
    ingest_pdf()

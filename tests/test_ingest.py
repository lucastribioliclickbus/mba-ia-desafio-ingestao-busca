from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.documents import Document

from ingest import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    build_splitter,
    chunk_ids,
    load_chunks,
    without_empty_metadata,
)

DOCUMENT_PDF = Path(__file__).resolve().parent.parent / "document.pdf"


def test_splitter_uses_the_sizes_required_by_the_challenge() -> None:
    assert (CHUNK_SIZE, CHUNK_OVERLAP) == (1000, 150)


def test_long_text_is_split_into_chunks_within_the_size_limit() -> None:
    text = " ".join(f"palavra{index}" for index in range(2000))

    chunks = build_splitter().split_documents([Document(page_content=text)])

    assert len(chunks) > 1
    assert all(len(chunk.page_content) <= CHUNK_SIZE for chunk in chunks)


def test_consecutive_chunks_share_the_configured_overlap() -> None:
    text = "".join(str(index % 10) for index in range(3000))

    chunks = build_splitter().split_documents([Document(page_content=text)])

    first, second = chunks[0].page_content, chunks[1].page_content
    assert first[-CHUNK_OVERLAP:] == second[:CHUNK_OVERLAP]


def test_empty_metadata_values_are_dropped_and_content_is_preserved() -> None:
    chunk = Document(page_content="conteúdo", metadata={"page": 1, "title": "", "author": None})

    cleaned = without_empty_metadata([chunk])[0]

    assert cleaned.metadata == {"page": 1}
    assert cleaned.page_content == "conteúdo"


def test_chunk_ids_are_unique_and_stable_between_runs() -> None:
    first_run = chunk_ids("document.pdf", 3)

    assert first_run == ["document.pdf:0", "document.pdf:1", "document.pdf:2"]
    assert first_run == chunk_ids("document.pdf", 3)
    assert len(set(first_run)) == 3


def test_missing_pdf_is_reported_instead_of_ingesting_nothing() -> None:
    with pytest.raises(FileNotFoundError):
        load_chunks(Path("/tmp/arquivo-que-nao-existe.pdf"))


@pytest.mark.skipif(not DOCUMENT_PDF.is_file(), reason="document.pdf ausente")
def test_challenge_pdf_produces_chunks() -> None:
    chunks = load_chunks(DOCUMENT_PDF)

    assert chunks
    assert all(chunk.page_content.strip() for chunk in chunks)
    assert all(len(chunk.page_content) <= CHUNK_SIZE for chunk in chunks)

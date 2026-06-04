from typing import Literal


ChunkingStrategy = Literal["fixed", "semantic"]


def chunk_fixed(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Strategy 1 — Fixed Size Chunking.

    Splits text into chunks of `chunk_size` characters with `overlap`
    characters shared between consecutive chunks. Good for uniform
    retrieval window sizes.
    """
    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap  # slide forward with overlap

    return chunks


def chunk_semantic(text: str, min_length: int = 100) -> list[str]:
    """
    Strategy 2 — Semantic / Paragraph Chunking.

    Splits text on double newlines (paragraph breaks), then groups
    short paragraphs together so no chunk is smaller than `min_length`
    characters. Preserves natural semantic boundaries.
    """
    raw_paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks: list[str] = []
    buffer = ""

    for paragraph in raw_paragraphs:
        if not buffer:
            buffer = paragraph
        else:
            # If the buffer is still short, merge the next paragraph in
            if len(buffer) < min_length:
                buffer += "\n\n" + paragraph
            else:
                chunks.append(buffer)
                buffer = paragraph

    if buffer:
        chunks.append(buffer)

    return chunks


def get_chunks(text: str, strategy: ChunkingStrategy) -> list[str]:
    """Entry point — pick the right strategy and return chunks."""
    if strategy == "fixed":
        return chunk_fixed(text)
    elif strategy == "semantic":
        return chunk_semantic(text)
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy}")

import logging
from pathlib import Path

from pypdf import PdfReader

logger = logging.getLogger(__name__)


def extract_text(file_path: str) -> str:
    """
    Extract all text from a PDF or TXT file.
    Returns a single string of the full document text.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    suffix = path.suffix.lower()

    if suffix == ".txt":
        return _extract_from_txt(path)
    elif suffix == ".pdf":
        return _extract_from_pdf(path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}. Only .pdf and .txt are supported.")


def _extract_from_txt(path: Path) -> str:
    """Read a plain text file."""
    logger.info("Extracting text from TXT: %s", path.name)
    text = path.read_text(encoding="utf-8", errors="ignore")
    logger.info("Extracted %d characters from %s", len(text), path.name)
    return text


def _extract_from_pdf(path: Path) -> str:
    """Extract text from all pages of a PDF using pypdf."""
    logger.info("Extracting text from PDF: %s", path.name)
    reader = PdfReader(str(path))
    pages_text: list[str] = []

    for page_num, page in enumerate(reader.pages):
        page_text = page.extract_text() or ""
        pages_text.append(page_text)
        logger.debug("Page %d: %d chars", page_num + 1, len(page_text))

    full_text = "\n\n".join(pages_text)
    logger.info("Extracted %d characters from %d pages in %s", len(full_text), len(reader.pages), path.name)
    return full_text

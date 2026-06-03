from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

from pypdf import PdfReader


@dataclass(frozen=True)
class PdfExtractResult:
    text: str
    page_count: int
    extracted_pages: int
    char_count: int


def extract_pdf_text(file_bytes: bytes) -> PdfExtractResult:
    """Extract all readable text from a PDF.

    We intentionally do not summarize or truncate here. Different teachers
    structure slides very differently, so ClassVoice stores the full extracted
    material text and lets the LLM decide how to summarize it later.
    """

    with NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
        temp_file.write(file_bytes)
        temp_path = Path(temp_file.name)

    try:
        reader = PdfReader(str(temp_path))
        parts: list[str] = []
        for page_index, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                parts.append(f"第 {page_index} 页\n{text.strip()}")

        full_text = "\n\n".join(parts).strip()
        return PdfExtractResult(
            text=full_text,
            page_count=len(reader.pages),
            extracted_pages=len(parts),
            char_count=len(full_text),
        )
    finally:
        temp_path.unlink(missing_ok=True)

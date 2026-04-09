"""PDF to page-image rendering utilities with stable page-id mapping."""

from __future__ import annotations

from pathlib import Path

from src.utils.io_utils import ensure_dir

try:
    import fitz
except ImportError:  # pragma: no cover
    fitz = None


def render_pdf_to_images(pdf_path: str | Path, output_dir: str | Path, dpi: int = 150) -> dict[str, list[str] | dict[str, str]]:
    """Render a PDF into per-page PNG images and return paths plus page-id mapping."""
    if fitz is None:
        raise ImportError("PyMuPDF is required for PDF rendering.")

    pdf_path = Path(pdf_path)
    output_dir = ensure_dir(output_dir)
    document = fitz.open(pdf_path)
    doc_id = pdf_path.stem
    page_paths: list[str] = []
    page_id_map: dict[str, str] = {}
    matrix = fitz.Matrix(dpi / 72.0, dpi / 72.0)

    for page_index in range(document.page_count):
        page = document.load_page(page_index)
        page_number = page_index + 1
        page_id = f"{doc_id}_page_{page_number:03d}"
        image_path = output_dir / f"{page_id}.png"
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        pixmap.save(image_path)
        page_paths.append(str(image_path))
        page_id_map[page_id] = str(image_path)

    document.close()
    return {"image_paths": page_paths, "page_id_map": page_id_map}

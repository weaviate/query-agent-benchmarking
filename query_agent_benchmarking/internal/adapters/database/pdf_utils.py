import base64
from pathlib import Path


def get_pdf_page_counts(pdf_files: list[Path]) -> dict[Path, int]:
    """Quick pass to count pages per PDF (fast, no rendering)."""
    import fitz

    counts = {}
    for pdf_path in pdf_files:
        doc = fitz.open(str(pdf_path))
        counts[pdf_path] = len(doc)
        doc.close()
    return counts


def process_single_pdf(pdf_path: Path) -> list[dict]:
    """Convert all pages of a single PDF to base64-encoded images."""
    import fitz  # PyMuPDF

    pages = []
    pdf_doc = fitz.open(str(pdf_path))
    for page_num in range(len(pdf_doc)):
        page = pdf_doc[page_num]
        pix = page.get_pixmap(dpi=72)
        img_bytes = pix.tobytes("jpeg")
        image_base64 = base64.b64encode(img_bytes).decode("utf-8")
        pages.append({
            "image": image_base64,
            "source_pdf": pdf_path.stem,
            "page_number": page_num + 1,
        })
    pdf_doc.close()
    return pages

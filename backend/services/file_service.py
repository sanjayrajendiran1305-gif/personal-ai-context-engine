import csv
import io
import itertools
import re
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
UPLOAD_DIR = ROOT / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
MAX_FILE_BYTES = 10 * 1024 * 1024
ALLOWED = {".pdf", ".pptx", ".docx", ".txt", ".md", ".csv", ".xlsx", ".png", ".jpg", ".jpeg", ".webp"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def safe_name(name):
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(name or "upload").name).strip("._") or "upload"
    return cleaned[:100]


def process_upload(filename, data):
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED:
        raise ValueError(f"Unsupported file type: {suffix or 'no extension'}. Supported: PDF, PPTX, DOCX, TXT, MD, CSV, XLSX, PNG, JPG, WEBP.")
    if len(data) > MAX_FILE_BYTES:
        raise ValueError(f"{filename} exceeds the 10 MB per-file limit.")
    info = {"name": safe_name(filename), "extension": suffix, "size": len(data), "text": "", "image": None}
    if suffix in IMAGE_EXTENSIONS:
        mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}[suffix]
        info["image"] = {"data": data, "mime_type": mime}
        info["text"] = f"[Image attached: {safe_name(filename)}. Inspect the image directly.]"
        return info
    stream = io.BytesIO(data)
    if suffix == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(stream)
        text = "\n".join((page.extract_text() or "") for page in reader.pages[:80])
        info["text"] = f"[PDF: {filename}, {len(reader.pages)} pages]\n{text[:18000]}"
    elif suffix == ".pptx":
        from pptx import Presentation
        deck = Presentation(stream)
        parts = []
        for index, slide in enumerate(itertools.islice(deck.slides, 80), 1):
            lines = []
            for shape in slide.shapes:
                if getattr(shape, "has_text_frame", False) and shape.text.strip():
                    lines.append(shape.text.strip())
                if getattr(shape, "has_table", False):
                    for row in shape.table.rows:
                        lines.append(" | ".join(cell.text for cell in row.cells))
            parts.append(f"Slide {index}:\n" + "\n".join(lines))
        info["text"] = f"[PowerPoint: {filename}, {len(deck.slides)} slides]\n" + "\n\n".join(parts)[:18000]
    elif suffix == ".docx":
        from docx import Document
        doc = Document(stream)
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        for table in doc.tables:
            for row in table.rows:
                paragraphs.append(" | ".join(cell.text for cell in row.cells))
        info["text"] = f"[Word document: {filename}]\n" + "\n".join(paragraphs)[:18000]
    elif suffix == ".xlsx":
        from openpyxl import load_workbook
        book = load_workbook(stream, read_only=True, data_only=True)
        parts = []
        for sheet in book.worksheets[:12]:
            parts.append(f"Sheet: {sheet.title}")
            for row in sheet.iter_rows(min_row=1, max_row=150, values_only=True):
                values = [str(value) if value is not None else "" for value in row]
                if any(values):
                    parts.append(" | ".join(values))
        info["text"] = f"[Spreadsheet: {filename}]\n" + "\n".join(parts)[:18000]
    else:
        try:
            info["text"] = f"[File: {filename}]\n" + data.decode("utf-8-sig")[:18000]
        except UnicodeDecodeError:
            raise ValueError(f"Could not read {filename} as UTF-8 text.")
    return info


def persist_upload(interaction_id, filename, data):
    extension = Path(filename).suffix.lower()
    stored_name = f"{interaction_id}_{uuid.uuid4().hex[:8]}{extension}"
    destination = UPLOAD_DIR / stored_name
    destination.write_bytes(data)
    return str(destination.relative_to(ROOT)), safe_name(filename)

"""Document extraction and conservative identity detection; never name-only merge."""
from __future__ import annotations

import io
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from PIL import Image, UnidentifiedImageError
from pypdf import PdfReader
import pypdfium2 as pdfium

from .model import ModelUnavailable

MAX_UPLOAD = 8 * 1024 * 1024
MAX_PAGES = 12
MAX_TEXT = 160_000
Image.MAX_IMAGE_PIXELS = 16_000_000


@dataclass
class Extraction:
    text: str
    method: str
    identity: dict
    summary: str
    summary_kind: str
    warnings: list[str]
    mime: str


def identity(text: str) -> dict:
    patterns = {
        "name": r"(?im)^\s*(?:patient(?:\s+name)?|name)\s*:\s*([^\r\n]+)",
        "dob": r"(?im)^\s*(?:dob|date\s+of\s+birth)\s*:\s*(\d{4}-\d{2}-\d{2})\s*$",
        "mrn": r"(?im)^\s*(?:mrn|medical\s+record\s+(?:number|id))\s*:\s*([\w-]{1,64})\s*$",
    }
    found = {k: sorted(set(v.strip() for v in re.findall(p, text))) for k,p in patterns.items()}
    result = {k: values[0] if len(values) == 1 else None for k, values in found.items()}
    result["ambiguous"] = any(len(values) > 1 for values in found.values())
    if result["dob"]:
        try:
            parsed = date.fromisoformat(result["dob"])
            if parsed > date.today():
                result["dob"] = None
        except ValueError:
            result["dob"] = None
    if result["name"] and len(result["name"]) > 120:
        result["name"] = None
    return result


def image_png(raw: bytes) -> bytes:
    with Image.open(io.BytesIO(raw)) as image:
        if image.format not in {"PNG", "JPEG", "WEBP"}:
            raise ValueError("Only PNG, JPEG and WebP images are accepted.")
        if image.width * image.height > 16_000_000:
            raise ValueError("Image exceeds 16 megapixels.")
        image.load()
        image = image.convert("RGB")
        image.thumbnail((1800,1800))
        stream = io.BytesIO()
        image.save(stream, format="PNG")
        return stream.getvalue()


def extract(filename: str, raw: bytes, model, use_model: bool) -> Extraction:
    if not raw or len(raw) > MAX_UPLOAD:
        raise ValueError("File must contain data and be no larger than 8 MiB.")
    ext = Path(filename).suffix.lower()
    warnings = []
    text = ""
    method = "text"
    mime = "text/plain"
    if ext in {".txt", ".md"}:
        if b"\x00" in raw:
            raise ValueError("Text files must not contain binary data.")
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ValueError("Text files must be UTF-8.") from exc
    elif ext == ".pdf":
        mime = "application/pdf"
        if not raw.startswith(b"%PDF-"):
            raise ValueError("Invalid PDF signature.")
        try:
            pdf = PdfReader(io.BytesIO(raw), strict=True)
            if pdf.is_encrypted:
                raise ValueError("Password-protected PDFs are not supported. Use a synthetic unlocked copy.")
            if len(pdf.pages) > MAX_PAGES:
                raise ValueError(f"Maximum {MAX_PAGES} pages per PDF in this prototype.")
            pieces = [(page.extract_text() or "").strip() for page in pdf.pages]
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError("The PDF could not be parsed safely.") from exc
        method = "pdf_text"
        missing = [i for i,piece in enumerate(pieces) if len(piece) < 30]
        if missing:
            # No repeated OCR loop: at most ONE scanned page is sent to the VLM.
            method = "pdf_partial"
            warnings.append("Some pages lack a text layer. Check every page; extraction may be incomplete.")
            if len(pieces) == 1 and use_model:
                try:
                    doc = pdfium.PdfDocument(raw)
                    page = doc[0]
                    bitmap = page.render(scale=1.6)
                    pil = bitmap.to_pil().convert("RGB")
                    pil.thumbnail((1800,1800))
                    buf = io.BytesIO(); pil.save(buf, format="PNG")
                    bitmap.close(); page.close(); doc.close()
                    pieces[0] = model.transcribe(buf.getvalue())
                    method = "vlm_image"
                except ModelUnavailable:
                    warnings.append("Local model unavailable. Add a verified transcription during review.")
                except Exception:
                    warnings.append("Scanned page could not be rendered. Add a verified transcription during review.")
        text = "\n\n".join(f"[Page {i+1}]\n{piece or '[No text extracted]'}" for i,piece in enumerate(pieces))
    elif ext in {".png", ".jpg", ".jpeg", ".webp"}:
        mime = {".png":"image/png", ".jpg":"image/jpeg", ".jpeg":"image/jpeg", ".webp":"image/webp"}[ext]
        try:
            png = image_png(raw)
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
            raise ValueError("The image could not be decoded safely.") from exc
        method = "vlm_image"
        if use_model:
            try:
                text = model.transcribe(png)
            except ModelUnavailable:
                warnings.append("Local model unavailable. Add a verified transcription during review.")
        else:
            warnings.append("Model was not requested. Add a verified transcription during review.")
    else:
        raise ValueError("Supported files: TXT, MD, PDF, PNG, JPEG and WebP. No archives or office macros.")
    if len(text) > MAX_TEXT:
        raise ValueError("Extracted text exceeds 160,000 characters. Split the source into smaller reports.")
    if method in {"vlm_image", "pdf_partial"}:
        warnings.append("Visual/partial extraction requires human identity and transcription review; never auto-routed.")
    summary_kind = "source_excerpt"
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    summary = "\n".join(lines[:8])[:900] or "No text extracted. Review the original and enter a verified transcription."
    if use_model and text.strip():
        try:
            summary = model.summarize(text)
            summary_kind = "model_draft"
        except ModelUnavailable:
            warnings.append("Summary model unavailable. Showing an exact source excerpt instead.")
    return Extraction(text, method, identity(text), summary, summary_kind, warnings, mime)

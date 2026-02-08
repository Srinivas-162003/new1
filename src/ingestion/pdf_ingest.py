import os
from typing import List, Tuple

import fitz
import pdfplumber

from config import CHUNK_OVERLAP_CHARS, MAX_CHUNK_CHARS
from models import Chunk, DocumentRecord, SectionRecord
from utils.text import chunk_text, normalize_text
from ingestion.vision_enhancer import extract_visual_elements


def ingest_pdfs(paths: List[str], use_vision: bool = False) -> Tuple[List[DocumentRecord], List[Chunk]]:
    documents: List[DocumentRecord] = []
    all_chunks: List[Chunk] = []

    for path in paths:
        document = extract_document(path, use_vision)
        documents.append(document)
        all_chunks.extend(build_chunks(document))

    return documents, all_chunks


def extract_document(path: str, use_vision: bool) -> DocumentRecord:
    doc_id = os.path.splitext(os.path.basename(path))[0]
    doc = fitz.open(path)
    title = doc.metadata.get("title") or doc_id

    sections: List[SectionRecord] = []

    with pdfplumber.open(path) as plumber_doc:
        for page_index in range(len(doc)):
            page = doc[page_index]
            page_num = page_index + 1
            text = page.get_text() or ""
            tables_md = _extract_tables(plumber_doc, page_index)

            vision_notes = ""
            if use_vision:
                pix = page.get_pixmap(dpi=200)
                vision = extract_visual_elements(pix.tobytes("png"))
                vision_notes = _format_vision_notes(vision)

            sections.append(
                SectionRecord(
                    title=f"Page {page_num}",
                    page=page_num,
                    content=normalize_text(text),
                    tables=tables_md,
                    vision_notes=vision_notes,
                )
            )

    doc.close()
    return DocumentRecord(doc_id=doc_id, title=title, path=path, sections=sections)


def build_chunks(document: DocumentRecord) -> List[Chunk]:
    chunks: List[Chunk] = []

    for section in document.sections:
        parts = [section.content]
        if section.tables:
            parts.append("Tables:\n" + section.tables)
        if section.vision_notes:
            parts.append("Figures/Equations:\n" + section.vision_notes)

        combined = "\n\n".join([part for part in parts if part])
        for chunk_text_item in chunk_text(combined, MAX_CHUNK_CHARS, CHUNK_OVERLAP_CHARS):
            metadata = {
                "doc_id": document.doc_id,
                "title": document.title,
                "page": section.page,
                "section": section.title,
                "path": document.path,
            }
            chunks.append(Chunk(text=chunk_text_item, metadata=metadata))

    return chunks


def _extract_tables(plumber_doc: pdfplumber.PDF, page_index: int) -> str:
    page = plumber_doc.pages[page_index]
    tables = page.extract_tables() or []
    if not tables:
        return ""

    table_lines: List[str] = []
    for table in tables:
        table_lines.append(_table_to_markdown(table))

    return "\n\n".join(table_lines)


def _table_to_markdown(table: List[List[str]]) -> str:
    if not table:
        return ""

    header = table[0]
    rows = table[1:]
    header_line = "| " + " | ".join(cell or "" for cell in header) + " |"
    separator = "| " + " | ".join(["---"] * len(header)) + " |"
    row_lines = ["| " + " | ".join(cell or "" for cell in row) + " |" for row in rows]

    return "\n".join([header_line, separator] + row_lines)


def _format_vision_notes(vision: dict) -> str:
    parts = []
    if vision.get("equations"):
        parts.append("Equations: " + vision["equations"])
    if vision.get("tables"):
        parts.append("Tables: " + vision["tables"])
    if vision.get("figures"):
        parts.append("Figures: " + vision["figures"])
    return "\n".join(parts)

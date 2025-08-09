from __future__ import annotations

import asyncio
from email import policy
from email.parser import BytesParser
from io import BytesIO
from typing import Dict, List, Tuple

import fitz  # PyMuPDF
import httpx
from docx import Document

from .utils import logger, sanitize_url, read_env

DOWNLOAD_MAX_MB = int(read_env("DOWNLOAD_MAX_MB", "25") or "25")


async def _download_one(client: httpx.AsyncClient, url: str) -> bytes:
    sanitized = sanitize_url(url)
    if sanitized.startswith("file://"):
        # local file path
        path = sanitized.replace("file://", "")
        with open(path, "rb") as f:
            content = f.read()
    else:
        resp = await client.get(sanitized, timeout=30.0, headers={"User-Agent": "hackrx/1.0"})
        resp.raise_for_status()
        content = resp.content
    size_mb = len(content) / (1024 * 1024)
    if size_mb > DOWNLOAD_MAX_MB:
        raise ValueError(f"File too large: {size_mb:.1f} MB exceeds limit of {DOWNLOAD_MAX_MB} MB")
    return content


def _extract_pdf(content: bytes) -> List[Tuple[int, str]]:
    texts: List[Tuple[int, str]] = []
    with fitz.open(stream=content, filetype="pdf") as doc:
        for i, page in enumerate(doc, start=1):
            text = page.get_text("text")
            texts.append((i, text))
    return texts


def _extract_docx(content: bytes) -> List[Tuple[int, str]]:
    bio = BytesIO(content)
    doc = Document(bio)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    text = "\n".join(paragraphs)
    return [(1, text)]


def _extract_email(content: bytes) -> List[Tuple[int, str]]:
    msg = BytesParser(policy=policy.default).parsebytes(content)
    parts: List[str] = []
    if msg["Subject"]:
        parts.append(f"Subject: {msg['Subject']}")
    if msg["From"]:
        parts.append(f"From: {msg['From']}")
    if msg["To"]:
        parts.append(f"To: {msg['To']}")
    # prefer text/plain
    body = None
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                body = part.get_content()
                break
    else:
        body = msg.get_content()
    if body:
        parts.append(str(body))
    return [(1, "\n".join(parts))]


def _detect_and_extract(url: str, content: bytes) -> List[Tuple[int, str]]:
    lower = url.lower()
    if lower.endswith(".pdf"):
        return _extract_pdf(content)
    if lower.endswith(".docx"):
        return _extract_docx(content)
    if lower.endswith(".eml") or lower.endswith(".msg"):
        return _extract_email(content)
    # heuristic: try PDF first, else docx, else email
    try:
        return _extract_pdf(content)
    except Exception:
        try:
            return _extract_docx(content)
        except Exception:
            return _extract_email(content)


async def download_and_extract(urls: List[str]) -> Dict[str, List[Tuple[int, str]]]:
    """Download URLs and extract text per page preserving page numbers.

    Returns mapping: url -> list of (page_number, text)
    """
    async with httpx.AsyncClient(follow_redirects=True) as client:
        results: Dict[str, List[Tuple[int, str]]] = {}
        tasks = [asyncio.create_task(_download_one(client, u)) for u in urls]
        contents = await asyncio.gather(*tasks)
        for u, content in zip(urls, contents):
            try:
                pages = _detect_and_extract(u, content)
                results[u] = pages
            except Exception as e:
                logger.exception("Extraction failed for %s: %s", u, e)
                results[u] = []
        return results

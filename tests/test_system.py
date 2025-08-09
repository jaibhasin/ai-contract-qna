import os
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient
from reportlab.pdfgen import canvas

from app.ingest import download_and_extract
from app.chunker import chunk_texts
from app.embeddings import EmbeddingClient
from app.main import app


def make_sample_pdf(text: str) -> Path:
    fd, path = tempfile.mkstemp(suffix=".pdf")
    os.close(fd)
    p = Path(path)
    c = canvas.Canvas(str(p))
    c.setFont("Helvetica", 12)
    for i, line in enumerate(text.split("\n")):
        c.drawString(72, 800 - 18 * i, line)
    c.showPage()
    c.save()
    return p


def test_ingest_and_chunk():
    sample = "Coverage for outpatient services is subject to a waiting period of 30 days. Exclusions apply for experimental treatments."
    pdf = make_sample_pdf(sample)
    url = f"file://{pdf}"
    pages = os.environ.copy()
    pages = None

    # Download & extract
    pages_by_url = __import__("asyncio").get_event_loop().run_until_complete(download_and_extract([url]))
    assert url in pages_by_url
    assert len(pages_by_url[url]) >= 1

    # Chunk
    chunks = chunk_texts(pages_by_url, chunk_tokens=200, overlap=50)
    assert len(chunks) >= 1
    assert chunks[0].meta.doc_url == url


def test_embeddings_dummy_provider():
    os.environ["EMBEDDING_PROVIDER"] = "dummy"
    emb = EmbeddingClient()
    v = emb.embed(["hello world", "another test"])
    assert v.shape[0] == 2


def test_endpoint_integration():
    os.environ["TEAM_TOKEN"] = "dev-token"
    os.environ["EMBEDDING_PROVIDER"] = "dummy"
    client = TestClient(app)

    text = "Benefits include coverage after 60 days waiting period. The policy does not cover cosmetic procedures."
    pdf = make_sample_pdf(text)
    url = f"file://{pdf}"

    payload = {
        "documents": url,
        "questions": [
            "When does coverage start?",
            "What exclusions are listed?",
        ],
    }
    resp = client.post("/api/v1/hackrx/run", json=payload, headers={"Authorization": "Bearer dev-token"})
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert "answers" in data
    assert len(data["answers"]) == 2
    for ans in data["answers"]:
        assert isinstance(ans.get("conditions", []), list)
        assert len(ans.get("sources", [])) >= 1

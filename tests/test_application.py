import json
from pathlib import Path
from fastapi.testclient import TestClient

from backend.database import database
from backend import main


def setup_test_db(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "test_memory.db")
    database.init_db()


def test_chat_preference_memory_and_conversation_lifecycle(tmp_path, monkeypatch):
    setup_test_db(tmp_path, monkeypatch)
    monkeypatch.setattr(main, "generate_gemini", lambda prompt, images=None: "## Clear answer\n\nA useful explanation with an example.\n\n- Step one\n- Step two")
    monkeypatch.setattr(main, "generate_groq", lambda prompt, images=None: "A shorter explanation.")
    client = TestClient(main.app)

    created = client.post("/conversations/new")
    assert created.status_code == 200
    conversation_id = created.json()["conversation_id"]

    response = client.post("/ask", json={
        "message": "Explain recursion in Python with an example",
        "provider": "both",
        "conversation_id": conversation_id,
    })
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["responses"]["gemini"]
    assert payload["responses"]["groq"]
    assert payload["category"] == "programming"

    # A clear first-person preference is remembered; ordinary one-off questions are not.
    preference_turn = client.post("/ask", json={
        "message": "I prefer detailed programming explanations with practical examples",
        "provider": "gemini",
        "conversation_id": conversation_id,
    })
    assert preference_turn.status_code == 200
    assert preference_turn.json()["agent"]["memories_added"] == 1

    history = client.get(f"/conversations/{conversation_id}")
    assert history.status_code == 200
    assert len(history.json()["messages"]) == 2

    chosen = client.post("/preference", json={"request_id": payload["request_id"], "provider": "gemini"})
    assert chosen.status_code == 200
    assert chosen.json()["preferred_provider"] == "gemini"
    # If the user changes their choice, the training labels are corrected rather than duplicated.
    client.post("/preference", json={"request_id": payload["request_id"], "provider": "groq"})
    labels = {row["provider"]: row["label"] for row in database.fetch_all(
        "SELECT provider,label FROM preference_examples WHERE interaction_id=?",
        (payload["request_id"],),
    )}
    assert labels == {"gemini": 0, "groq": 1}
    client.post("/preference", json={"request_id": payload["request_id"], "provider": "gemini"})

    feedback = client.post("/preference-feedback", json={
        "request_id": payload["request_id"],
        "reasons": ["More practical examples", "Clear step-by-step structure"],
        "custom_reason": "Keep the explanation beginner friendly",
    })
    assert feedback.status_code == 200
    memories = client.get("/memory?category=programming").json()["memories"]
    assert len(memories) >= 3

    renamed = client.patch(f"/conversations/{conversation_id}", json={"title": "Python learning"})
    assert renamed.status_code == 200
    assert client.get("/conversations").json()["conversations"][0]["title"] == "Python learning"

    deleted = client.delete(f"/conversations/{conversation_id}")
    assert deleted.status_code == 200
    assert deleted.json()["deleted_messages"] == 2
    assert client.get(f"/conversations/{conversation_id}").status_code == 404
    assert client.get("/memory?category=programming").json()["memories"]


def test_health_and_agent_analysis(tmp_path, monkeypatch):
    setup_test_db(tmp_path, monkeypatch)
    client = TestClient(main.app)
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["agent"] == "Personalization Agent"
    analysis = client.post("/agent/analyze", json={"message": "Make a PowerPoint presentation about robotics", "provider": "both"})
    assert analysis.status_code == 200
    assert analysis.json()["category"] == "document_creation"
    assert analysis.json()["output_format"] == "pptx"


def test_export_generates_downloadable_pdf(tmp_path, monkeypatch):
    setup_test_db(tmp_path, monkeypatch)
    monkeypatch.setattr(main, "generate_gemini", lambda prompt, images=None: "## Summary\n\nThis is an export test.")
    client = TestClient(main.app)
    conversation_id = client.post("/conversations/new").json()["conversation_id"]
    ask = client.post("/ask", json={"message": "Summarize this", "provider": "gemini", "conversation_id": conversation_id}).json()
    export = client.post("/exports", json={"request_id": ask["request_id"], "format": "pdf", "provider": "gemini"})
    assert export.status_code == 200, export.text
    download = client.get(export.json()["download_url"])
    assert download.status_code == 200
    assert download.content.startswith(b"%PDF")


def test_file_text_extraction():
    from backend.services.file_service import process_upload
    item = process_upload("notes.txt", b"A small text attachment for the agent.")
    assert "small text attachment" in item["text"]
    assert item["name"] == "notes.txt"


def test_ask_with_text_attachment(tmp_path, monkeypatch):
    setup_test_db(tmp_path, monkeypatch)
    monkeypatch.setattr(main, "generate_gemini", lambda prompt, images=None: "The attached note says hello.")
    client = TestClient(main.app)
    conversation_id = client.post("/conversations/new").json()["conversation_id"]
    response = client.post(
        "/ask-with-files",
        data={"message": "Summarize the attached note", "provider": "gemini", "conversation_id": conversation_id},
        files=[("files", ("notes.txt", b"This document is about local memory retrieval.", "text/plain"))],
    )
    assert response.status_code == 200, response.text
    assert response.json()["attachments"][0]["name"] == "notes.txt"
    history = client.get(f"/conversations/{conversation_id}").json()["messages"]
    assert history[0]["attachments"][0]["name"] == "notes.txt"
    attachment = client.get("/attachments/" + history[0]["attachments"][0]["id"])
    assert attachment.status_code == 200
    assert b"local memory retrieval" in attachment.content


def test_docx_and_pptx_exports(tmp_path, monkeypatch):
    setup_test_db(tmp_path, monkeypatch)
    monkeypatch.setattr(main, "generate_gemini", lambda prompt, images=None: "## Overview\n\n- First point\n- Second point\n\nA closing paragraph.")
    client = TestClient(main.app)
    conversation_id = client.post("/conversations/new").json()["conversation_id"]
    ask = client.post("/ask", json={"message": "Make a report", "provider": "gemini", "conversation_id": conversation_id}).json()
    for fmt, signature in (("docx", b"PK"), ("pptx", b"PK")):
        export = client.post("/exports", json={"request_id": ask["request_id"], "format": fmt, "provider": "gemini"})
        assert export.status_code == 200, export.text
        download = client.get(export.json()["download_url"])
        assert download.status_code == 200
        assert download.content.startswith(signature)
        import io
        if fmt == "docx":
            from docx import Document
            assert Document(io.BytesIO(download.content)).paragraphs
        else:
            from pptx import Presentation
            assert len(Presentation(io.BytesIO(download.content)).slides) >= 2


def test_pdf_pptx_docx_xlsx_input_extraction():
    import io
    from reportlab.pdfgen import canvas
    from pptx import Presentation
    from docx import Document
    from openpyxl import Workbook
    from backend.services.file_service import process_upload

    pdf_buffer = io.BytesIO()
    pdf = canvas.Canvas(pdf_buffer)
    pdf.drawString(72, 720, "PDF memory extraction test")
    pdf.save()
    assert "PDF memory extraction test" in process_upload("sample.pdf", pdf_buffer.getvalue())["text"]

    deck = Presentation()
    slide = deck.slides.add_slide(deck.slide_layouts[5])
    slide.shapes.title.text = "PowerPoint memory extraction test"
    ppt_buffer = io.BytesIO(); deck.save(ppt_buffer)
    assert "PowerPoint memory extraction test" in process_upload("sample.pptx", ppt_buffer.getvalue())["text"]

    doc = Document(); doc.add_paragraph("Word memory extraction test")
    doc_buffer = io.BytesIO(); doc.save(doc_buffer)
    assert "Word memory extraction test" in process_upload("sample.docx", doc_buffer.getvalue())["text"]

    book = Workbook(); book.active["A1"] = "Spreadsheet memory extraction test"
    xlsx_buffer = io.BytesIO(); book.save(xlsx_buffer)
    assert "Spreadsheet memory extraction test" in process_upload("sample.xlsx", xlsx_buffer.getvalue())["text"]


def test_conflicting_style_preference_updates_same_category_memory(tmp_path, monkeypatch):
    setup_test_db(tmp_path, monkeypatch)
    from backend.memory.preference_manager import save_confirmed_feedback
    save_confirmed_feedback("programming", ["More detailed explanation"])
    save_confirmed_feedback("programming", ["More concise answer"])
    rows = database.fetch_all(
        "SELECT * FROM memories WHERE category='programming' AND memory_key='preference_response_detail' AND is_active=1"
    )
    assert len(rows) == 1
    assert rows[0]["memory_value"] == "More concise answer"
    assert rows[0]["evidence_count"] == 2

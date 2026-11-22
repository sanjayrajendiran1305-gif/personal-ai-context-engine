import json
import os
import re
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import closing
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env", override=False)

from backend.database.database import init_db, execute_query, fetch_one, fetch_all, current_time, get_connection
from backend.agents.personalization_agent import personalization_agent
from backend.memory.memory_manager import classify_task, capture_explicit_preferences
from backend.memory.preference_manager import get_preferences, update_memory, deactivate_preference, save_confirmed_feedback
from backend.ml.preference_model import response_features, record_pair, training_status, train_preference_models, invalidate_models
from backend.services.gemini_service import generate_response as generate_gemini, is_configured as gemini_configured, get_model as gemini_model, generate_image as gemini_generate_image
from backend.services.groq_service import generate_response as generate_groq, is_configured as groq_configured, get_model as groq_model
from backend.services.file_service import process_upload, persist_upload, UPLOAD_DIR, MAX_FILE_BYTES
from backend.services.document_service import create_document, OUTPUT_DIR

app = FastAPI(title="Personal AI Context Engine", version="2.0.0", description="A user-controlled context and preference layer for multiple AI assistants.")
init_db()


def migrate_legacy_feedback_once():
    marker = fetch_one("SELECT name FROM migration_log WHERE name=?", ("legacy_feedback_to_memory_v1",))
    if marker:
        return
    rows = fetch_all("""
        SELECT id,message,selected_reasons,custom_reason,category
        FROM interactions
        WHERE selected_provider IS NOT NULL
    """)
    for row in rows:
        try:
            reasons = json.loads(row["selected_reasons"] or "[]")
        except json.JSONDecodeError:
            reasons = []
        custom = row["custom_reason"] or ""
        if not reasons and not custom:
            continue
        category = row["category"] or "general"
        if category == "general":
            category = classify_task(row["message"])["category"]
            execute_query("UPDATE interactions SET category=? WHERE id=?", (category, row["id"]))
            execute_query("UPDATE preference_examples SET category=? WHERE interaction_id=?", (category, row["id"]))
        save_confirmed_feedback(category, reasons, custom, row["id"])
    execute_query("INSERT OR IGNORE INTO migration_log(name,applied_at) VALUES(?,?)", ("legacy_feedback_to_memory_v1", current_time()))


migrate_legacy_feedback_once()


def refine_legacy_memory_categories_once():
    marker = fetch_one("SELECT name FROM migration_log WHERE name=?", ("legacy_memory_category_refinement_v1",))
    if marker:
        return
    rows = fetch_all("SELECT id,message FROM interactions WHERE category='general'")
    for row in rows:
        inferred_category = classify_task(row["message"])["category"]
        if inferred_category != "general":
            execute_query("UPDATE interactions SET category=? WHERE id=?", (inferred_category, row["id"]))
            execute_query("UPDATE preference_examples SET category=? WHERE interaction_id=?", (inferred_category, row["id"]))
            execute_query("UPDATE memories SET category=?,updated_at=? WHERE source_interaction_id=? AND category='general'", (inferred_category, current_time(), row["id"]))
    execute_query("INSERT OR IGNORE INTO migration_log(name,applied_at) VALUES(?,?)", ("legacy_memory_category_refinement_v1", current_time()))


refine_legacy_memory_categories_once()


class Query(BaseModel):
    message: str = Field(min_length=1, max_length=12000)
    provider: Literal["both", "gemini", "groq"] = "both"
    conversation_id: str | None = None


class Preference(BaseModel):
    request_id: str
    provider: Literal["gemini", "groq"]


class PreferenceFeedback(BaseModel):
    request_id: str
    reasons: list[str] = Field(default_factory=list, max_length=8)
    custom_reason: str = Field(default="", max_length=500)


class RenameConversation(BaseModel):
    title: str = Field(min_length=1, max_length=100)


class ExportRequest(BaseModel):
    request_id: str
    format: Literal["pdf", "docx", "pptx", "txt", "md"] = "pdf"
    provider: Literal["gemini", "groq"] | None = None
    title: str | None = None


class ImageRequest(BaseModel):
    prompt: str = Field(min_length=3, max_length=3000)


class MemoryUpdate(BaseModel):
    memory_value: str = Field(min_length=1, max_length=1000)


def create_title(message):
    title = re.sub(r"\s+", " ", str(message or "").strip())
    return title[:55] + ("…" if len(title) > 55 else "") or "New conversation"


def ensure_conversation(conversation_id, message):
    now = current_time()
    if conversation_id:
        existing = fetch_one("SELECT id,title FROM conversations WHERE id=?", (conversation_id,))
        if existing:
            if existing["title"] == "New conversation":
                execute_query("UPDATE conversations SET title=?,updated_at=? WHERE id=?", (create_title(message), now, conversation_id))
            return conversation_id
    conversation_id = str(uuid.uuid4())
    execute_query("INSERT INTO conversations(id,title,created_at,updated_at) VALUES(?,?,?,?)",
                  (conversation_id, create_title(message), now, now))
    return conversation_id


def _history_for(conversation_id, limit=6):
    rows = fetch_all("""
        SELECT message,gemini_response,groq_response,selected_provider
        FROM interactions WHERE conversation_id=? ORDER BY created_at DESC LIMIT ?
    """, (conversation_id, limit))
    history = []
    for row in reversed(rows):
        selected = row["selected_provider"]
        if selected == "gemini" and row["gemini_response"]:
            response, provider = row["gemini_response"], "Gemini"
        elif selected == "groq" and row["groq_response"]:
            response, provider = row["groq_response"], "Groq"
        elif row["gemini_response"] and row["groq_response"]:
            response = "Gemini: " + row["gemini_response"][:1200] + "\n\nGroq: " + row["groq_response"][:1200]
            provider = "Both models"
        elif row["gemini_response"]:
            response, provider = row["gemini_response"], "Gemini"
        elif row["groq_response"]:
            response, provider = row["groq_response"], "Groq"
        else:
            continue
        history.append({"message": row["message"], "response": response[:2800], "provider": provider})
    return history


def _answer_features(text):
    return response_features(text or "")


def _suggest_reasons(selected, alternative):
    a = _answer_features(selected)
    b = _answer_features(alternative)
    suggestions = []
    if a["log_words"] > b["log_words"] + 0.25:
        suggestions.append("More detailed explanation")
    elif a["log_words"] < b["log_words"] - 0.25:
        suggestions.append("More concise answer")
    if a["examples"] > b["examples"]:
        suggestions.append("More practical examples")
    if a["code_blocks"] > b["code_blocks"]:
        suggestions.append("Useful code examples")
    if a["numbered_steps"] > b["numbered_steps"]:
        suggestions.append("Clear step-by-step structure")
    if a["headings"] + a["bullets"] > b["headings"] + b["bullets"]:
        suggestions.append("Better organized answer")
    if a["tables"] > b["tables"]:
        suggestions.append("Clearer comparison tables")
    defaults = ["Clearer explanation", "Simpler language", "More relevant to my question", "Better examples", "Right level of detail"]
    for item in defaults:
        if len(suggestions) >= 6:
            break
        if item not in suggestions:
            suggestions.append(item)
    return suggestions[:6]


def _run_provider(provider, prompt, images):
    fn = generate_gemini if provider == "gemini" else generate_groq
    return fn(prompt, images=images or None)


def _ask_core(message, provider, conversation_id, processed_files=None):
    processed_files = processed_files or []
    image_parts = [item["image"] for item in processed_files if item.get("image")]
    attachment_text = "\n\n".join(item.get("text", "") for item in processed_files if item.get("text"))
    attachment_names = [item["name"] for item in processed_files]
    total_size = sum(item.get("size", 0) for item in processed_files)
    if total_size > 20 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Combined upload size exceeds 20 MB.")
    conversation_id = ensure_conversation(conversation_id, message)
    history = _history_for(conversation_id)
    analysis = personalization_agent.prepare_context(
        message, conversation_history=history,
        attachment_names=attachment_names, attachment_text=attachment_text,
    )
    providers = ["gemini", "groq"] if provider == "both" else [provider]
    services = {name: (generate_gemini if name == "gemini" else generate_groq) for name in providers}
    responses, errors = {}, {}
    with ThreadPoolExecutor(max_workers=len(providers)) as pool:
        futures = {pool.submit(services[name], analysis["context_prompt"], images=image_parts or None): name for name in providers}
        for future in as_completed(futures):
            name = futures[future]
            try:
                responses[name] = future.result()
            except Exception as exc:
                errors[name] = str(exc)[:1000]
    if not responses:
        raise HTTPException(status_code=502, detail={"message": "No connected assistant returned a response.", "provider_errors": errors})

    request_id = str(uuid.uuid4())
    now = current_time()
    attachment_descriptors = []
    execute_query("""
        INSERT INTO interactions(
            id,conversation_id,message,requested_provider,gemini_response,groq_response,
            gemini_error,groq_error,created_at,category,subcategory,output_format,attachments_json
        ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        request_id, conversation_id, message, provider,
        responses.get("gemini"), responses.get("groq"),
        errors.get("gemini"), errors.get("groq"), now,
        analysis["category"], analysis["subcategory"], analysis["output_format"], "[]",
    ))
    for item in processed_files:
        stored_path, safe_filename = persist_upload(request_id, item["name"], item["raw_data"])
        ext = Path(safe_filename).suffix.lower()
        mime = item.get("mime_type") or ("image/" + ext.lstrip("."))
        attachment_id = str(uuid.uuid4())
        execute_query("""
            INSERT INTO attachments(id,interaction_id,original_filename,stored_path,mime_type,size_bytes,created_at)
            VALUES(?,?,?,?,?,?,?)
        """, (attachment_id, request_id, safe_filename, stored_path, mime, item.get("size", 0), now))
        attachment_descriptors.append({"id": attachment_id, "name": safe_filename, "size": item.get("size", 0)})
    execute_query("UPDATE interactions SET attachments_json=? WHERE id=?", (json.dumps(attachment_descriptors), request_id))
    execute_query("UPDATE conversations SET updated_at=? WHERE id=?", (now, conversation_id))
    captured_memories = capture_explicit_preferences(message, analysis["category"], request_id)
    return {
        "request_id": request_id, "conversation_id": conversation_id,
        "category": analysis["category"], "subcategory": analysis["subcategory"],
        "output_format": analysis["output_format"], "agent": {
            "name": personalization_agent.name,
            "version": personalization_agent.version,
            "confidence": analysis["category_confidence"],
            "memories_used": len(analysis["relevant_memories"]),
            "memories_added": len(captured_memories),
            "learned_style_applied": analysis["learned_style_applied"],
            "history_turns_used": analysis["history_turns_used"],
        },
        "responses": responses, "errors": errors, "attachments": attachment_descriptors,
    }


@app.get("/")
def home():
    return FileResponse(ROOT / "frontend" / "index.html")


@app.get("/health")
def health():
    return {
        "status": "running", "application": "Personal AI Context Engine",
        "agent": personalization_agent.name, "agent_version": personalization_agent.version,
        "providers": {
            "gemini": {"configured": gemini_configured(), "model": gemini_model()},
            "groq": {"configured": groq_configured(), "model": groq_model()},
        },
        "capabilities": {"file_upload": True, "pdf_export": True, "pptx_export": True,
                         "docx_export": True, "image_generation_configured": bool(os.getenv("GEMINI_IMAGE_MODEL"))},
    }


@app.get("/providers")
def providers():
    return {"providers": [
        {"id": "gemini", "name": "Gemini", "configured": gemini_configured(), "model": gemini_model()},
        {"id": "groq", "name": "Groq", "configured": groq_configured(), "model": groq_model()},
    ]}


@app.post("/conversations/new")
def new_conversation():
    conversation_id = str(uuid.uuid4())
    now = current_time()
    execute_query("INSERT INTO conversations(id,title,created_at,updated_at) VALUES(?,?,?,?)",
                  (conversation_id, "New conversation", now, now))
    return {"conversation_id": conversation_id, "title": "New conversation", "has_history": False}


@app.get("/conversations")
def list_conversations(search: str | None = None):
    query = """
        SELECT c.*,
        (SELECT COUNT(*) FROM interactions i WHERE i.conversation_id=c.id) AS turn_count,
        (SELECT message FROM interactions i WHERE i.conversation_id=c.id ORDER BY i.created_at DESC LIMIT 1) AS last_message
        FROM conversations c
    """
    params = ()
    if search and search.strip():
        query += " WHERE c.title LIKE ? OR EXISTS (SELECT 1 FROM interactions x WHERE x.conversation_id=c.id AND x.message LIKE ?)"
        term = f"%{search.strip()[:100]}%"
        params = (term, term)
    query += " ORDER BY c.updated_at DESC"
    return {"conversations": [dict(row) for row in fetch_all(query, params)]}


@app.get("/conversations/{conversation_id}")
def get_conversation(conversation_id: str):
    conversation = fetch_one("SELECT * FROM conversations WHERE id=?", (conversation_id,))
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    rows = fetch_all("SELECT * FROM interactions WHERE conversation_id=? ORDER BY created_at ASC", (conversation_id,))
    messages = []
    for row in rows:
        item = dict(row)
        try:
            item["selected_reasons"] = json.loads(item.get("selected_reasons") or "[]")
            item["attachments"] = json.loads(item.get("attachments_json") or "[]")
        except json.JSONDecodeError:
            item["selected_reasons"], item["attachments"] = [], []
        item["reason_suggestions"] = []
        if item.get("selected_provider"):
            selected_text = item.get(item["selected_provider"] + "_response") or ""
            alternative_provider = "groq" if item["selected_provider"] == "gemini" else "gemini"
            alternative_text = item.get(alternative_provider + "_response") or ""
            if selected_text:
                item["reason_suggestions"] = _suggest_reasons(selected_text, alternative_text)
        messages.append(item)
    return {"conversation": dict(conversation), "messages": messages}


@app.patch("/conversations/{conversation_id}")
def rename_conversation(conversation_id: str, data: RenameConversation):
    title = re.sub(r"\s+", " ", data.title.strip())[:100]
    count = execute_query("UPDATE conversations SET title=?,updated_at=? WHERE id=?", (title, current_time(), conversation_id))
    if count == 0:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return {"message": "Conversation renamed.", "title": title}


@app.post("/conversations/{conversation_id}/rename")
def rename_conversation_legacy(conversation_id: str, data: RenameConversation):
    return rename_conversation(conversation_id, data)


@app.delete("/conversations/{conversation_id}")
def delete_conversation(conversation_id: str):
    with closing(get_connection()) as conn:
        exists = conn.execute("SELECT id FROM conversations WHERE id=?", (conversation_id,)).fetchone()
        if not exists:
            raise HTTPException(status_code=404, detail="Conversation not found.")
        attachment_rows = conn.execute("""
            SELECT a.stored_path FROM attachments a JOIN interactions i ON a.interaction_id=i.id
            WHERE i.conversation_id=?
        """, (conversation_id,)).fetchall()
        artifact_rows = conn.execute("""
            SELECT a.path FROM artifacts a JOIN interactions i ON a.interaction_id=i.id
            WHERE i.conversation_id=?
        """, (conversation_id,)).fetchall()
        conn.execute("""
            UPDATE memories SET source_interaction_id=NULL,updated_at=?
            WHERE source_interaction_id IN (SELECT id FROM interactions WHERE conversation_id=?)
        """, (current_time(), conversation_id))
        conn.execute("DELETE FROM preference_examples WHERE interaction_id IN (SELECT id FROM interactions WHERE conversation_id=?)", (conversation_id,))
        conn.execute("DELETE FROM attachments WHERE interaction_id IN (SELECT id FROM interactions WHERE conversation_id=?)", (conversation_id,))
        conn.execute("DELETE FROM artifacts WHERE interaction_id IN (SELECT id FROM interactions WHERE conversation_id=?)", (conversation_id,))
        conn.execute("DELETE FROM interactions WHERE conversation_id=?", (conversation_id,))
        deleted = conn.execute("SELECT changes()").fetchone()[0]
        conn.execute("DELETE FROM conversations WHERE id=?", (conversation_id,))
        conn.commit()
    for row in attachment_rows:
        path = (ROOT / row["stored_path"]).resolve()
        if UPLOAD_DIR.resolve() in path.parents and path.exists():
            path.unlink(missing_ok=True)
    for row in artifact_rows:
        path = (ROOT / row["path"]).resolve()
        if OUTPUT_DIR.resolve() in path.parents and path.exists():
            path.unlink(missing_ok=True)
    # A cached model must not retain signals from a conversation the user deleted.
    invalidate_models()
    return {"message": "Conversation deleted successfully.", "conversation_id": conversation_id, "deleted_messages": deleted}


@app.post("/ask")
def ask(query: Query):
    return _ask_core(query.message, query.provider, query.conversation_id)


@app.post("/ask-with-files")
async def ask_with_files(
    message: str = Form(...),
    provider: Literal["both", "gemini", "groq"] = Form("both"),
    conversation_id: str | None = Form(None),
    files: list[UploadFile] = File(default=[]),
):
    if not message.strip() and not files:
        raise HTTPException(status_code=422, detail="Enter a question or attach a file.")
    processed = []
    total = 0
    for upload in files:
        data = await upload.read(MAX_FILE_BYTES + 1)
        total += len(data)
        if len(data) > MAX_FILE_BYTES:
            raise HTTPException(status_code=413, detail=f"{upload.filename} exceeds the 10 MB per-file limit.")
        try:
            item = process_upload(upload.filename or "upload", data)
            item["raw_data"] = data
            item["mime_type"] = upload.content_type or "application/octet-stream"
            processed.append(item)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    if total > 20 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Combined upload size exceeds 20 MB.")
    prompt = message.strip() or "Please analyze the attached file(s) and explain the important information."
    return _ask_core(prompt, provider, conversation_id, processed)


@app.post("/preference")
def save_preference_choice(preference: Preference):
    row = fetch_one("SELECT * FROM interactions WHERE id=?", (preference.request_id,))
    if not row:
        raise HTTPException(status_code=404, detail="Interaction not found.")
    selected = row[f"{preference.provider}_response"]
    if not selected:
        raise HTTPException(status_code=400, detail="The selected assistant response is unavailable.")
    if row["selected_provider"] != preference.provider:
        execute_query("""
            UPDATE interactions SET selected_provider=?,selected_reasons='[]',custom_reason='' WHERE id=?
        """, (preference.provider, preference.request_id))
    else:
        execute_query("UPDATE interactions SET selected_provider=? WHERE id=?", (preference.provider, preference.request_id))
    inserted = record_pair(preference.request_id, row["category"] or "general", preference.provider,
                           row["gemini_response"], row["groq_response"])
    learning = training_status()
    auto_training = None
    latest_run = learning.get("latest_run") or {}
    if learning["ready_to_train"] and (
        not learning["trained"] or int(latest_run.get("example_count") or 0) < learning["training_rows"]
    ):
        try:
            auto_training = train_preference_models()
        except Exception as exc:
            # A training failure must never prevent the user's response choice from being saved.
            auto_training = {"error": str(exc)[:300]}
    alternative_provider = "groq" if preference.provider == "gemini" else "gemini"
    alternative = row[f"{alternative_provider}_response"] or ""
    return {
        "message": "Your preferred response has been saved.", "request_id": preference.request_id,
        "conversation_id": row["conversation_id"], "preferred_provider": preference.provider,
        "reason_suggestions": _suggest_reasons(selected, alternative),
        "comparison_examples_added": inserted,
        "auto_training": auto_training,
        "feedback_optional": True,
    }


@app.post("/preference-feedback")
def save_feedback(feedback: PreferenceFeedback):
    row = fetch_one("SELECT message,selected_provider,category FROM interactions WHERE id=?", (feedback.request_id,))
    if not row or not row["selected_provider"]:
        raise HTTPException(status_code=404, detail="Select a preferred response first.")
    reasons = list(dict.fromkeys(str(reason).strip()[:120] for reason in feedback.reasons if str(reason).strip()))
    custom = feedback.custom_reason.strip()[:500]
    if not reasons and not custom:
        raise HTTPException(status_code=400, detail="Select a reason or write a custom reason. You can also skip feedback.")
    category = row["category"] or personalization_agent.analyze_request(row["message"])["category"]
    ids = personalization_agent.learn_from_feedback(category, reasons, custom, feedback.request_id)
    execute_query("UPDATE interactions SET selected_reasons=?,custom_reason=? WHERE id=?",
                  (json.dumps(reasons), custom, feedback.request_id))
    return {"message": "Feedback saved. Future requests can use it when relevant.", "category": category,
            "saved_memories": len(ids), "reasons": reasons, "custom_reason": custom}


@app.get("/preferences")
def preference_history():
    rows = fetch_all("""
        SELECT id,conversation_id,message,category,selected_provider,selected_reasons,custom_reason,created_at
        FROM interactions WHERE selected_provider IS NOT NULL ORDER BY created_at DESC LIMIT 200
    """)
    result = []
    for row in rows:
        item = dict(row)
        try:
            item["selected_reasons"] = json.loads(item.get("selected_reasons") or "[]")
        except json.JSONDecodeError:
            item["selected_reasons"] = []
        result.append(item)
    return {"preferences": result}


@app.get("/memory")
def get_memory(category: str | None = None, include_inactive: bool = False):
    return {"memories": get_preferences(category, include_inactive)}


@app.get("/memory/profile")
def get_profile():
    return personalization_agent.get_profile()


@app.patch("/memory/{memory_id}")
def edit_memory(memory_id: str, data: MemoryUpdate):
    if not update_memory(memory_id, data.memory_value):
        raise HTTPException(status_code=404, detail="Memory not found or inactive.")
    return {"message": "Memory updated.", "memory_id": memory_id}


@app.delete("/memory/{memory_id}")
def forget_memory(memory_id: str):
    if not deactivate_preference(memory_id):
        raise HTTPException(status_code=404, detail="Memory not found or already inactive.")
    return {"message": "Memory forgotten.", "memory_id": memory_id}


@app.post("/memory/clear-inferred")
def clear_inferred(category: str | None = None):
    from backend.memory.preference_manager import clear_inferred_preferences
    count = clear_inferred_preferences(category)
    return {"deactivated": count, "message": "Inferred memories cleared. User-confirmed preferences were retained."}


@app.post("/agent/analyze")
def analyze_request(query: Query):
    analysis = personalization_agent.analyze_request(query.message)
    return {
        "agent": personalization_agent.name, "version": personalization_agent.version,
        "category": analysis["category"], "subcategory": analysis["subcategory"],
        "output_format": analysis["output_format"], "confidence": analysis["category_confidence"],
        "matched_terms": analysis["matched_terms"],
        "relevant_memories": analysis["relevant_memories"],
        "learned_style_applied": bool(__import__("backend.ml.preference_model", fromlist=["learned_style_guidance"]).learned_style_guidance(analysis["category"])),
    }


@app.get("/learning/status")
def learning_status():
    return training_status()


@app.post("/learning/train")
def train_learning_models():
    try:
        result = train_preference_models()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return result


@app.post("/exports")
def export_response(request: ExportRequest):
    row = fetch_one("SELECT * FROM interactions WHERE id=?", (request.request_id,))
    if not row:
        raise HTTPException(status_code=404, detail="Response not found.")
    provider = request.provider or row["selected_provider"]
    if provider not in {"gemini", "groq"}:
        provider = "gemini" if row["gemini_response"] else "groq"
    text = row[f"{provider}_response"]
    if not text:
        raise HTTPException(status_code=400, detail="No response is available for export.")
    title = request.title or create_title(row["message"])
    try:
        path = create_document(text, request.format, title)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not create export: {exc}") from exc
    artifact_id = str(uuid.uuid4())
    filename = path.name
    execute_query("INSERT INTO artifacts(id,interaction_id,filename,format,path,created_at) VALUES(?,?,?,?,?,?)",
                  (artifact_id, request.request_id, filename, request.format, str(path.relative_to(ROOT)), current_time()))
    return {"artifact_id": artifact_id, "filename": filename, "format": request.format, "download_url": f"/downloads/{artifact_id}"}


@app.get("/downloads/{artifact_id}")
def download_artifact(artifact_id: str):
    row = fetch_one("SELECT * FROM artifacts WHERE id=?", (artifact_id,))
    if not row:
        raise HTTPException(status_code=404, detail="File not found.")
    path = (ROOT / row["path"]).resolve()
    if OUTPUT_DIR.resolve() not in path.parents or not path.is_file():
        raise HTTPException(status_code=404, detail="File is no longer available.")
    return FileResponse(path, filename=row["filename"])


@app.get("/attachments/{attachment_id}")
def download_attachment(attachment_id: str):
    row = fetch_one("SELECT * FROM attachments WHERE id=?", (attachment_id,))
    if not row:
        raise HTTPException(status_code=404, detail="Attachment not found.")
    path = (ROOT / row["stored_path"]).resolve()
    if UPLOAD_DIR.resolve() not in path.parents or not path.is_file():
        raise HTTPException(status_code=404, detail="Attachment is no longer available.")
    return FileResponse(path, filename=row["original_filename"], media_type=row["mime_type"])


@app.post("/generate-image")
def generate_image(request: ImageRequest):
    try:
        data, mime = gemini_generate_image(request.prompt)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    extension = ".png" if "png" in mime else ".jpg" if "jpeg" in mime else ".webp"
    artifact_id = str(uuid.uuid4())
    path = OUTPUT_DIR / f"{artifact_id}{extension}"
    path.write_bytes(data)
    execute_query("INSERT INTO artifacts(id,interaction_id,filename,format,path,created_at) VALUES(?,?,?,?,?,?)",
                  (artifact_id, None, path.name, extension.lstrip("."), str(path.relative_to(ROOT)), current_time()))
    return {"artifact_id": artifact_id, "filename": path.name, "download_url": f"/downloads/{artifact_id}", "mime_type": mime}


import os
import re
import json
import uuid
import sqlite3
from pathlib import Path
from collections import Counter
from contextlib import closing
from datetime import datetime,timezone
from typing import Literal

from fastapi import FastAPI,HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel,Field
from dotenv import load_dotenv
from google import genai
from groq import Groq

ROOT=Path(__file__).resolve().parent.parent
DB_PATH=ROOT/"memory.db"
load_dotenv(ROOT/".env")

gemini_client=genai.Client(api_key=os.getenv("GEMINI_API_KEY")) if os.getenv("GEMINI_API_KEY") else None
groq_client=Groq(api_key=os.getenv("GROQ_API_KEY")) if os.getenv("GROQ_API_KEY") else None

app=FastAPI()

class Query(BaseModel):
    message:str=Field(min_length=1)
    provider:Literal["both","gemini","groq"]="both"
    conversation_id:str|None=None

class Preference(BaseModel):
    request_id:str
    provider:Literal["gemini","groq"]

class PreferenceFeedback(BaseModel):
    request_id:str
    reasons:list[str]=Field(default_factory=list,max_length=8)
    custom_reason:str=Field(default="",max_length=500)

def init_db():
    with closing(sqlite3.connect(DB_PATH)) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS interactions(
                id TEXT PRIMARY KEY,
                message TEXT NOT NULL,
                requested_provider TEXT NOT NULL,
                gemini_response TEXT,
                groq_response TEXT,
                selected_provider TEXT,
                created_at TEXT NOT NULL
            )
        """)
        columns={row[1] for row in conn.execute("PRAGMA table_info(interactions)")}
        additions={
            "conversation_id":"TEXT",
            "selected_reasons":"TEXT NOT NULL DEFAULT '[]'",
            "custom_reason":"TEXT NOT NULL DEFAULT ''"
        }
        for name,definition in additions.items():
            if name not in columns:
                conn.execute(f"ALTER TABLE interactions ADD COLUMN {name} {definition}")
        conn.commit()

init_db()

def get_preference_summary():
    with closing(sqlite3.connect(DB_PATH)) as conn:
        rows=conn.execute("""
            SELECT selected_reasons,custom_reason
            FROM interactions
            WHERE selected_provider IS NOT NULL
            ORDER BY created_at DESC
            LIMIT 100
        """).fetchall()

    counts=Counter()
    notes=[]

    for saved_reasons,custom_reason in rows:
        try:
            reasons=json.loads(saved_reasons or "[]")
        except json.JSONDecodeError:
            reasons=[]

        for reason in reasons:
            if isinstance(reason,str) and reason.strip():
                counts[reason.strip()]+=1

        if custom_reason and custom_reason.strip():
            note=custom_reason.strip().replace("\n"," ")[:200]
            if note not in notes:
                notes.append(note)

    lines=[]
    if counts or notes:
        lines.append("Previously confirmed response preferences:")
        for reason,count in counts.most_common(8):
            lines.append(f"- {reason} (confirmed {count} time(s))")
        for note in notes[:3]:
            lines.append(f'- User preference note: "{note}"')

    return "\n".join(lines)

def get_conversation_history(conversation_id):
    if not conversation_id:
        return []

    with closing(sqlite3.connect(DB_PATH)) as conn:
        rows=conn.execute("""
            SELECT message,selected_provider,gemini_response,groq_response
            FROM interactions
            WHERE conversation_id=?
            ORDER BY created_at DESC
            LIMIT 8
        """,(conversation_id,)).fetchall()

    history=[]

    for message,selected,gemini_response,groq_response in reversed(rows):
        if selected=="gemini" and gemini_response:
            answer=gemini_response
            source="Gemini"
        elif selected=="groq" and groq_response:
            answer=groq_response
            source="Groq"
        elif gemini_response and not groq_response:
            answer=gemini_response
            source="Gemini"
        elif groq_response and not gemini_response:
            answer=groq_response
            source="Groq"
        else:
            continue

        history.append({
            "message":message,
            "response":answer[:4000],
            "provider":source
        })

    return history

def build_gemini_prompt(message,history,preferences):
    sections=[]

    if preferences:
        sections.append(
            "Use these user-confirmed response preferences when relevant. "
            "Accuracy and the current question take priority:\n"+preferences
        )

    if history:
        previous="\n\n".join(
            f"User: {turn['message']}\n"
            f"Previously preferred response ({turn['provider']}): {turn['response']}"
            for turn in history
        )
        sections.append("Conversation history:\n"+previous)

    sections.append("Current user message:\n"+message)
    return "\n\n".join(sections)

@app.get("/")
def home():
    return FileResponse(ROOT/"frontend"/"index.html")

@app.post("/conversations/new")
def new_conversation():
    return {"conversation_id":str(uuid.uuid4()),"has_history":False}

@app.post("/conversations/continue-latest")
def continue_latest_conversation():
    with closing(sqlite3.connect(DB_PATH)) as conn:
        row=conn.execute("""
            SELECT id,conversation_id,message,selected_provider
            FROM interactions
            WHERE selected_provider IS NOT NULL
            ORDER BY created_at DESC
            LIMIT 1
        """).fetchone()

        if row is None:
            return {
                "conversation_id":str(uuid.uuid4()),
                "has_history":False,
                "message":"No saved preferred response was found. A new conversation is ready."
            }

        interaction_id,conversation_id,message,provider=row

        if not conversation_id:
            conversation_id=str(uuid.uuid4())
            conn.execute(
                "UPDATE interactions SET conversation_id=? WHERE id=?",
                (conversation_id,interaction_id)
            )
            conn.commit()

    return {
        "conversation_id":conversation_id,
        "has_history":True,
        "seed_question":message,
        "preferred_provider":provider,
        "message":"Ready to continue from your last saved preferred response."
    }

@app.post("/ask")
def ask(query:Query):
    conversation_id=query.conversation_id or str(uuid.uuid4())
    history=get_conversation_history(conversation_id)
    preferences=get_preference_summary()

    responses={}
    errors={}
    providers=["gemini","groq"] if query.provider=="both" else [query.provider]

    for provider in providers:
        try:
            if provider=="gemini":
                if not gemini_client:
                    raise ValueError("Gemini API key is not configured.")

                result=gemini_client.models.generate_content(
                    model=os.getenv("GEMINI_MODEL"),
                    contents=build_gemini_prompt(
                        query.message,history,preferences
                    )
                )
                responses["gemini"]=result.text or "No text response returned."

            else:
                if not groq_client:
                    raise ValueError("Groq API key is not configured.")

                messages=[]

                if preferences:
                    messages.append({
                        "role":"system",
                        "content":(
                            "Use these user-confirmed response preferences when relevant. "
                            "Prioritize accuracy and the current question.\n\n"+preferences
                        )
                    })

                for turn in history:
                    messages.append({"role":"user","content":turn["message"]})
                    messages.append({
                        "role":"assistant",
                        "content":turn["response"]
                    })

                messages.append({"role":"user","content":query.message})

                result=groq_client.chat.completions.create(
                    model=os.getenv("GROQ_MODEL"),
                    messages=messages
                )
                responses["groq"]=result.choices[0].message.content or "No text response returned."

        except Exception as exc:
            errors[provider]=str(exc)

    if not responses:
        raise HTTPException(status_code=502,detail=errors)

    request_id=str(uuid.uuid4())

    with closing(sqlite3.connect(DB_PATH)) as conn:
        conn.execute("""
            INSERT INTO interactions(
                id,conversation_id,message,requested_provider,
                gemini_response,groq_response,created_at
            ) VALUES(?,?,?,?,?,?,?)
        """,(
            request_id,conversation_id,query.message,query.provider,
            responses.get("gemini"),responses.get("groq"),
            datetime.now(timezone.utc).isoformat()
        ))
        conn.commit()

    return {
        "request_id":request_id,
        "conversation_id":conversation_id,
        "responses":responses,
        "errors":errors
    }

def suggest_reasons(question,chosen_provider,chosen_text,other_text,preferences):
    fallback=[
        "Clearer explanation",
        "Better examples",
        "Better organized answer",
        "Right level of detail",
        "Simpler language",
        "More practical information"
    ]

    if not gemini_client:
        return fallback,"fallback"

    prompt=f"""
Suggest possible reasons a user may prefer one AI response.
These are guesses, not facts. The user will choose which reasons are accurate.

Previous confirmed preferences:
{preferences or "No previous preference feedback."}

Question:
{question[:1500]}

Selected {chosen_provider} response:
{chosen_text[:3500]}

Alternative response:
{(other_text or "Not available")[:3500]}

Return a JSON array of exactly 5 short, distinct reasons.
Each reason should be a label of 2 to 7 words.
Return only the JSON array.
"""

    try:
        result=gemini_client.models.generate_content(
            model=os.getenv("GEMINI_MODEL"),
            contents=prompt
        )
        raw=result.text or ""
        match=re.search(r"\[[\s\S]*\]",raw)
        parsed=json.loads(match.group(0) if match else raw)

        if isinstance(parsed,dict):
            parsed=parsed.get("reasons",[])

        suggestions=[]
        if isinstance(parsed,list):
            for item in parsed:
                if isinstance(item,str) and item.strip() and item.strip() not in suggestions:
                    suggestions.append(item.strip()[:100])

        if len(suggestions)>=3:
            return suggestions[:6],"gemini"
    except Exception:
        pass

    return fallback,"fallback"

@app.post("/preference")
def save_preference(preference:Preference):
    with closing(sqlite3.connect(DB_PATH)) as conn:
        conn.row_factory=sqlite3.Row
        row=conn.execute("""
            SELECT message,gemini_response,groq_response,conversation_id
            FROM interactions WHERE id=?
        """,(preference.request_id,)).fetchone()

        if row is None:
            raise HTTPException(status_code=404,detail="Interaction not found.")

        chosen_text=row[f"{preference.provider}_response"]

        if not chosen_text:
            raise HTTPException(status_code=400,detail="That response is not available.")

        conversation_id=row["conversation_id"] or str(uuid.uuid4())

        conn.execute("""
            UPDATE interactions
            SET selected_provider=?,conversation_id=?
            WHERE id=?
        """,(preference.provider,conversation_id,preference.request_id))
        conn.commit()

        question=row["message"]
        other_provider="groq" if preference.provider=="gemini" else "gemini"
        other_text=row[f"{other_provider}_response"]

    suggestions,source=suggest_reasons(
        question,preference.provider,chosen_text,
        other_text,get_preference_summary()
    )

    return {
        "message":"Preference saved successfully.",
        "request_id":preference.request_id,
        "conversation_id":conversation_id,
        "preferred_provider":preference.provider,
        "reason_suggestions":suggestions,
        "suggestion_source":source
    }

@app.post("/preference-feedback")
def save_preference_feedback(feedback:PreferenceFeedback):
    reasons=[]

    for reason in feedback.reasons:
        reason=reason.strip()[:100]
        if reason and reason not in reasons:
            reasons.append(reason)

    custom_reason=feedback.custom_reason.strip()

    if not reasons and not custom_reason:
        raise HTTPException(
            status_code=400,
            detail="Select at least one reason or enter an Other reason."
        )

    with closing(sqlite3.connect(DB_PATH)) as conn:
        row=conn.execute(
            "SELECT selected_provider FROM interactions WHERE id=?",
            (feedback.request_id,)
        ).fetchone()

        if row is None or row[0] is None:
            raise HTTPException(
                status_code=404,
                detail="Save the preferred response before saving feedback."
            )

        conn.execute("""
            UPDATE interactions
            SET selected_reasons=?,custom_reason=?
            WHERE id=?
        """,(
            json.dumps(reasons),custom_reason,feedback.request_id
        ))
        conn.commit()

    return {
        "message":"Preference feedback saved.",
        "saved_reasons":reasons,
        "custom_reason":custom_reason
    }

@app.get("/preferences")
def get_preferences():
    with closing(sqlite3.connect(DB_PATH)) as conn:
        conn.row_factory=sqlite3.Row
        rows=conn.execute("""
            SELECT id,message,selected_provider,selected_reasons,
                   custom_reason,created_at
            FROM interactions
            WHERE selected_provider IS NOT NULL
            ORDER BY created_at DESC
            LIMIT 50
        """).fetchall()

    preferences=[]

    for row in rows:
        item=dict(row)
        try:
            item["selected_reasons"]=json.loads(item["selected_reasons"] or "[]")
        except json.JSONDecodeError:
            item["selected_reasons"]=[]
        preferences.append(item)

    return {"preferences":preferences}

import math
import re
from datetime import datetime, timezone
from backend.memory.preference_manager import get_preferences, save_confirmed_feedback, save_preference, preference_topic

CATEGORY_TERMS = {
    "programming": {"python": 2, "java": 2, "javascript": 2, "typescript": 2, "c++": 3, "c#": 3,
        "programming": 3, "coding": 3, "code": 2, "debug": 3, "debugging": 3, "algorithm": 2,
        "api": 2, "backend": 2, "frontend": 2, "database": 2, "sql": 2, "fastapi": 3,
        "flask": 2, "react": 2, "arduino": 2, "robotics": 2, "microcontroller": 2, "recursion": 2},
    "education": {"exam": 3, "midterm": 3, "syllabus": 3, "assignment": 3, "mcq": 3, "quiz": 2,
        "study": 2, "lecture": 2, "notes": 2, "revision": 3, "course": 1, "homework": 2,
        "tutorial": 1, "textbook": 1, "worksheet": 2, "question bank": 3},
    "research": {"research": 3, "paper": 2, "literature review": 4, "citation": 3, "hypothesis": 3,
        "experiment": 2, "methodology": 3, "journal": 2, "research gap": 4, "findings": 2},
    "document_creation": {"pdf": 3, "ppt": 3, "pptx": 3, "powerpoint": 3, "presentation": 3,
        "document": 2, "report": 3, "resume": 3, "cv": 3, "proposal": 3, "letter": 2,
        "memo": 2, "docx": 3, "slides": 3, "word file": 3},
    "data_analysis": {"dataset": 3, "data analysis": 4, "statistics": 3, "visualization": 3,
        "regression": 3, "classification": 3, "clustering": 3, "csv": 2, "prediction": 2,
        "forecast": 3, "eda": 3, "feature engineering": 3, "pandas": 2, "numpy": 2},
    "creative": {"story": 2, "poem": 2, "creative writing": 3, "brainstorm": 2, "fiction": 2,
        "script": 2, "slogan": 2, "logo": 2, "image generation": 3, "illustration": 2},
    "career": {"career": 3, "interview": 3, "resume": 2, "portfolio": 2, "job": 2, "internship": 2, "salary": 2},
    "shopping": {"brand": 2, "brands": 2, "product": 2, "buy": 2, "purchase": 2, "price": 2, "budget": 1, "creatine": 3, "supplement": 3, "protein": 2, "best": 1, "recommend": 2, "compare products": 3},
    "health_wellness": {"workout": 2, "nutrition": 3, "fitness": 2, "hydration": 2, "diet": 2, "exercise": 2, "recovery": 2},
}

SUBTASK_RULES = [
    ("debugging", r"\b(debug|debugging|fix|error|exception|traceback|not working|bug)\b"),
    ("code_generation", r"\b(write|create|implement|build|generate)\b.*\b(code|function|class|program|script|api|application)\b"),
    ("code_explanation", r"\b(explain|understand|walk me through)\b.*\b(code|program|function|class|python|java|javascript|c\+\+|sql)\b"),
    ("presentation_generation", r"\b(create|make|generate|prepare|build)\b.*\b(presentation|slides|ppt|pptx|powerpoint)\b"),
    ("document_generation", r"\b(create|make|generate|prepare|format|write)\b.*\b(pdf|document|docx|report|resume|cv|proposal|letter)\b"),
    ("image_generation", r"\b(generate|create|draw|design)\b.*\b(image|picture|illustration|logo|diagram)\b"),
    ("data_analysis", r"\b(analy[sz]e|analysis|explore|visuali[sz]e|plot|forecast|predict)\b.*\b(data|dataset|csv|table|numbers)\b"),
    ("research_synthesis", r"\b(research|literature review|compare studies|summari[sz]e papers|citations)\b"),
    ("comparison", r"\b(compare|comparison|versus|vs\.?|difference|rank|which is better)\b"),
    ("summarization", r"\b(summarize|summarise|summary|key points|extract the main points)\b"),
    ("concept_explanation", r"\b(explain|what is|what are|how does|how do|teach me|describe)\b"),
    ("general_question", r".*"),
]

FORMAT_RULES = [
    ("pptx", r"\b(pptx|powerpoint|ppt|slides|presentation)\b"),
    ("pdf", r"\b(pdf)\b"),
    ("docx", r"\b(docx|word document|word file)\b"),
    ("xlsx", r"\b(xlsx|excel|spreadsheet)\b"),
    ("csv", r"\b(csv)\b"),
    ("image", r"\b(image|picture|illustration|png|jpg|jpeg|webp)\b"),
]

STOP = set("the a an is are was were to of in on for and or but with my me i you your it this that these those what how why when where can could would should please give tell make want need from about into will do does be explain show help some any all one two".split())


def normalize_text(value):
    return re.sub(r"\s+", " ", str(value or "").lower()).strip()


def tokenize(text):
    return {t for t in re.findall(r"[a-z0-9+#.]+", normalize_text(text)) if len(t) > 1 and t not in STOP}


def classify_task(message, attachment_names=None):
    text = normalize_text(message + " " + " ".join(attachment_names or []))
    tokens = tokenize(text)
    scores = {category: 0 for category in CATEGORY_TERMS}
    matched = {category: [] for category in CATEGORY_TERMS}
    for category, terms in CATEGORY_TERMS.items():
        for term, weight in terms.items():
            if (" " in term and term in text) or (" " not in term and term in tokens):
                scores[category] += weight
                matched[category].append(term)
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    category, best = ranked[0]
    second = ranked[1][1]
    if best == 0:
        category, confidence = "general", 0.50
    else:
        confidence = min(0.94, 0.55 + best * 0.045 + (best - second) * 0.035)
    subtask = next((name for name, pattern in SUBTASK_RULES if re.search(pattern, text)), "general_question")
    if subtask == "concept_explanation" and category == "programming" and tokens.intersection({"python", "java", "javascript", "typescript", "c++", "sql"}):
        subtask = "code_explanation"
    output_format = next((name for name, pattern in FORMAT_RULES if re.search(pattern, text)), None)
    if not output_format and re.search(r"\b(create|generate|export|download)\b.*\b(document|file|report)\b", text):
        output_format = "pdf"
    return {
        "category": category, "subcategory": subtask, "output_format": output_format,
        "confidence": round(confidence, 2), "scores": scores,
        "matched_terms": matched.get(category, []), "classifier": "rules+signals-v1",
    }


def _recency(updated_at):
    try:
        stamp = datetime.fromisoformat(str(updated_at).replace("Z", "+00:00"))
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=timezone.utc)
        days = max(0, (datetime.now(timezone.utc) - stamp.astimezone(timezone.utc)).total_seconds() / 86400)
        return 0.4 * math.exp(-days / 180)
    except (TypeError, ValueError):
        return 0.0


def _tfidf_scores(query, documents):
    # Optional local sentence embeddings improve paraphrase matching without an API call.
    # They are opt-in because the first use downloads a model and needs disk/RAM.
    import os
    if os.getenv("ENABLE_SEMANTIC_EMBEDDINGS", "false").lower() in {"1", "true", "yes"}:
        try:
            from sentence_transformers import SentenceTransformer
            global _embedding_model
            try:
                model = _embedding_model
            except NameError:
                _embedding_model = SentenceTransformer(os.getenv("SENTENCE_TRANSFORMER_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
                model = _embedding_model
            vectors = model.encode([query] + documents, normalize_embeddings=True)
            return (vectors[1:] @ vectors[0]).tolist()
        except Exception:
            pass  # Fall back to deterministic local TF-IDF when optional model is unavailable.
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        corpus = [query] + documents
        matrix = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True).fit_transform(corpus)
        return cosine_similarity(matrix[0:1], matrix[1:]).ravel().tolist()
    except Exception:
        q = tokenize(query)
        scores = []
        for doc in documents:
            d = tokenize(doc)
            scores.append(len(q & d) / max(1, len(q | d)))
        return scores


def retrieve_relevant_memories(message, category=None, limit=10):
    task = classify_task(message)
    category = category or task["category"]
    all_memories = get_preferences()
    candidates = [m for m in all_memories if m.get("category") in {category, "general"}]
    if not candidates:
        return []
    docs = [f"{m.get('memory_key','').replace('_', ' ')} {m.get('memory_value','')}" for m in candidates]
    similarities = _tfidf_scores(message, docs)
    ranked = []
    for memory, similarity in zip(candidates, similarities):
        conf = float(memory.get("confidence", 0.5) or 0.5)
        evidence = max(1, int(memory.get("evidence_count", 1) or 1))
        source = 0.25 if memory.get("source_type") == "user_confirmed" else 0
        category_boost = 0.22 if memory.get("category") == category else 0.08
        score = (float(similarity) * 0.65 + category_boost + source + min(0.12, math.log1p(evidence - 1) * 0.04) + _recency(memory.get("updated_at"))) * (0.6 + conf * 0.5)
        # A category match is useful even when the wording is different, but low-similarity records rank later.
        ranked.append((score, memory))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [memory for _, memory in ranked[:max(1, limit)]]


def build_memory_context(message, limit=10, task=None):
    task = task or classify_task(message)
    memories = retrieve_relevant_memories(message, task["category"], limit)
    if not memories:
        return {"task": task, "memories": [], "context": ""}
    lines = [
        "Relevant user memory and preference signals:",
        "Treat these as guidance, not as facts to announce. Current explicit instructions override older memories.",
    ]
    for item in memories:
        source = "user-confirmed" if item.get("source_type") == "user_confirmed" else "inferred"
        lines.append(f"- {item.get('memory_value','')} [{source}; confidence {float(item.get('confidence',0.5)):.0%}; evidence {item.get('evidence_count',1)}]")
    return {"task": task, "memories": memories, "context": "\n".join(lines)}



def capture_explicit_preferences(message, category=None, interaction_id=None):
    """Capture only clear first-person, durable preference statements.

    This is deliberately conservative: ordinary questions and one-off requests
    are not automatically converted into long-term profile memories.
    """
    text = re.sub(r"\s+", " ", str(message or "")).strip()
    patterns = [
        r"\b(i prefer|i usually prefer|my preference is|my preferred format is)\s+(.+)",
        r"\b(i don't like|i do not like|i dislike)\s+(.+)",
        r"\b(please remember that i prefer|remember that i prefer)\s+(.+)",
        r"\b(from now on,? (?:please )?(?:use|give me|show me|explain))\s+(.+)",
        r"\b(always (?:give me|show me|explain|include|use))\s+(.+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if not match:
            continue
        statement = text[match.start():].strip(" .!?")[:300]
        if len(statement.split()) < 3:
            return []
        resolved_category = category or classify_task(text)["category"]
        memory_id = save_preference(
            resolved_category,
            "preference_" + preference_topic(statement),
            statement,
            source_type="user_confirmed",
            confidence=0.94,
            source_interaction_id=interaction_id,
        )
        return [memory_id] if memory_id else []
    return []

def remember_user_feedback(category, reasons, custom_reason="", interaction_id=None):
    return save_confirmed_feedback(category, reasons, custom_reason, interaction_id)


def get_memory_overview():
    memories = get_preferences()
    summary = {}
    for memory in memories:
        category = memory.get("category", "other")
        summary.setdefault(category, {"count": 0, "confirmed": 0, "inferred": 0})
        summary[category]["count"] += 1
        source = memory.get("source_type", "inferred")
        summary[category]["confirmed" if source == "user_confirmed" else "inferred"] += 1
    return {"total_memories": len(memories), "categories": summary}

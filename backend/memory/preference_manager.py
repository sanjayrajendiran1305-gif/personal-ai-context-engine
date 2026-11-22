import re
import uuid
from backend.database.database import execute_query, fetch_one, fetch_all, current_time

CATEGORIES = {
    "programming", "education", "research", "general",
    "document_creation", "data_analysis", "creative", "career", "shopping", "health_wellness", "other",
}

ALIASES = {
    "coding": "programming", "software": "programming", "academic": "education",
    "study": "education", "documents": "document_creation", "writing": "document_creation",
    "analytics": "data_analysis", "presentation": "document_creation",
}


def normalize_category(category):
    value = re.sub(r"[^a-z_]+", "_", str(category or "general").lower()).strip("_")
    value = ALIASES.get(value, value)
    return value if value in CATEGORIES else "other"


def create_memory_key(value):
    key = re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")
    return key[:100] or "custom_preference"


def preference_topic(value):
    text = str(value or "").lower()
    topic_rules = [
        ("response_detail", r"\b(detail|detailed|concise|brief|short|long|comprehensive|depth)\b"),
        ("examples", r"\b(example|examples|practical|real.world|illustration)\b"),
        ("structure", r"\b(step.by.step|steps|organized|organised|table|tables|bullet|headings|format)\b"),
        ("code_examples", r"\b(code|coding|snippet|programming example)\b"),
        ("language_complexity", r"\b(simple|simpler|plain language|beginner|technical|jargon)\b"),
        ("sources", r"\b(source|sources|citation|citations|references)\b"),
        ("relevance", r"\b(relevant|specific|personalized|personalised|context)\b"),
    ]
    for topic, pattern in topic_rules:
        if re.search(pattern, text):
            return topic
    return create_memory_key(text)[:60] or "custom_preference"


def save_preference(category, memory_key, memory_value, source_type="user_confirmed",
                    confidence=0.9, source_interaction_id=None):
    category = normalize_category(category)
    key = create_memory_key(memory_key or memory_value)
    value = str(memory_value or "").strip()[:1000]
    if not value:
        return None
    if source_type not in {"user_confirmed", "inferred"}:
        raise ValueError("source_type must be user_confirmed or inferred")
    confidence = max(0.0, min(1.0, float(confidence)))
    now = current_time()
    existing = fetch_one("""
        SELECT id,confidence,evidence_count,source_type,memory_value
        FROM memories WHERE category=? AND memory_key=? AND is_active=1
        ORDER BY updated_at DESC LIMIT 1
    """, (category, key))

    if existing:
        old_source = existing["source_type"]
        old_confidence = float(existing["confidence"] or 0.5)
        if old_source == "user_confirmed" and source_type == "inferred":
            source_type = old_source
            confidence = old_confidence
            value = existing["memory_value"]
        elif source_type == "user_confirmed":
            confidence = min(0.99, max(old_confidence, confidence) + 0.01)
        else:
            confidence = max(old_confidence, confidence)
        execute_query("""
            UPDATE memories SET memory_value=?,source_type=?,confidence=?,
                evidence_count=evidence_count+1,
                source_interaction_id=COALESCE(?,source_interaction_id),updated_at=?
            WHERE id=?
        """, (value, source_type, confidence, source_interaction_id, now, existing["id"]))
        return existing["id"]

    memory_id = str(uuid.uuid4())
    execute_query("""
        INSERT INTO memories(id,category,memory_key,memory_value,source_type,
            confidence,evidence_count,source_interaction_id,is_active,created_at,updated_at)
        VALUES(?,?,?,?,?,?,1,?,1,?,?)
    """, (memory_id, category, key, value, source_type, confidence,
          source_interaction_id, now, now))
    return memory_id


def save_confirmed_feedback(category, reasons, custom_reason="", source_interaction_id=None):
    category = normalize_category(category)
    saved = []
    for reason in reasons or []:
        reason = str(reason).strip()[:200]
        if reason:
            saved.append(save_preference(category, "preference_" + preference_topic(reason), reason, "user_confirmed", 0.92, source_interaction_id))
    custom_reason = str(custom_reason or "").strip()[:500]
    if custom_reason:
        saved.append(save_preference(category, "preference_" + preference_topic(custom_reason), custom_reason, "user_confirmed", 0.92, source_interaction_id))
    return [item for item in saved if item]


def get_preferences(category=None, include_inactive=False):
    clauses = []
    params = []
    if category:
        clauses.append("category=?")
        params.append(normalize_category(category))
    if not include_inactive:
        clauses.append("is_active=1")
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    rows = fetch_all(
        "SELECT * FROM memories" + where + " ORDER BY confidence DESC,evidence_count DESC,updated_at DESC",
        tuple(params),
    )
    return [dict(row) for row in rows]


def build_preference_context(category=None, limit=12):
    rows = get_preferences(category)[:limit]
    if not rows:
        return ""
    grouped = {}
    for item in rows:
        grouped.setdefault(item["category"], []).append(item)
    lines = ["Relevant user-confirmed preferences and learned context:"]
    for group, items in grouped.items():
        lines.append(f"\n{group.replace('_', ' ').title()}:")
        for item in items:
            lines.append(
                f"- {item['memory_value']} (source={item['source_type']}; "
                f"confidence={float(item['confidence']):.0%}; evidence={item['evidence_count']})"
            )
    return "\n".join(lines)


def update_memory(memory_id, memory_value):
    value = str(memory_value or "").strip()[:1000]
    if not value:
        raise ValueError("Memory text cannot be empty")
    count = execute_query(
        "UPDATE memories SET memory_value=?,memory_key=?,source_type='user_confirmed',confidence=0.99,updated_at=? WHERE id=? AND is_active=1",
        (value, create_memory_key(value), current_time(), memory_id),
    )
    return count > 0


def deactivate_preference(memory_id):
    return execute_query(
        "UPDATE memories SET is_active=0,updated_at=? WHERE id=? AND is_active=1",
        (current_time(), memory_id),
    ) > 0


def clear_inferred_preferences(category=None):
    if category:
        return execute_query(
            "UPDATE memories SET is_active=0,updated_at=? WHERE category=? AND source_type='inferred' AND is_active=1",
            (current_time(), normalize_category(category)),
        )
    return execute_query(
        "UPDATE memories SET is_active=0,updated_at=? WHERE source_type='inferred' AND is_active=1",
        (current_time(),),
    )

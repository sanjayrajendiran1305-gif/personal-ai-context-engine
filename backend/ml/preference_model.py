"""Train a per-user response-style preference model from explicit A/B selections.

A pair contributes one positive row (the selected answer) and one negative row
(the alternative). No synthetic examples are created. A model is only trained
when there are enough real comparisons to make the result minimally useful.
"""
import json
import math
import re
import uuid
from pathlib import Path

from backend.database.database import fetch_all, fetch_one, execute_query, current_time

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "data" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
FEATURES = ["log_words", "headings", "bullets", "numbered_steps", "examples", "code_blocks", "tables", "paragraphs"]
MIN_PAIRS = 10


def response_features(text):
    text = str(text or "")
    words = re.findall(r"\b\w+\b", text)
    return {
        "log_words": math.log1p(len(words)),
        "headings": len(re.findall(r"(?m)^\s{0,3}#{1,6}\s", text)),
        "bullets": len(re.findall(r"(?m)^\s*[-*+]\s", text)),
        "numbered_steps": len(re.findall(r"(?m)^\s*\d+[.)]\s", text)),
        "examples": len(re.findall(r"\b(example|for instance|such as)\b", text, re.I)),
        "code_blocks": len(re.findall(r"```", text)) // 2,
        "tables": len(re.findall(r"(?m)^\s*\|", text)),
        "paragraphs": len([p for p in re.split(r"\n\s*\n", text) if p.strip()]),
    }


def record_pair(interaction_id, category, selected_provider, gemini_text, groq_text):
    if not gemini_text or not groq_text:
        return 0
    texts = {"gemini": gemini_text, "groq": groq_text}
    now = current_time()
    inserted = 0
    for provider, text in texts.items():
        label = 1 if provider == selected_provider else 0
        inserted += execute_query("""
            INSERT INTO preference_examples(
                id,interaction_id,category,provider,features_json,label,created_at
            ) VALUES(?,?,?,?,?,?,?)
            ON CONFLICT(interaction_id,provider) DO UPDATE SET
                category=excluded.category,
                features_json=excluded.features_json,
                label=excluded.label,
                created_at=excluded.created_at
        """, (str(uuid.uuid4()), interaction_id, category, provider,
              json.dumps(response_features(text)), label, now))
    return inserted


def _load_rows(category=None):
    if category and category != "all":
        rows = fetch_all("SELECT * FROM preference_examples WHERE category=? ORDER BY created_at", (category,))
    else:
        rows = fetch_all("SELECT * FROM preference_examples ORDER BY created_at")
    return [dict(row) for row in rows]


def training_status():
    rows = _load_rows()
    pairs = len({row["interaction_id"] for row in rows})
    latest = fetch_one("SELECT * FROM learning_runs ORDER BY created_at DESC LIMIT 1")
    models = []
    for path in sorted(MODEL_DIR.glob("*.joblib")):
        models.append(path.stem)
    by_category = {}
    for row in rows:
        by_category[row["category"]] = by_category.get(row["category"], 0) + 1
    return {
        "trained": bool(models),
        "model_categories": models,
        "comparison_pairs": pairs,
        "training_rows": len(rows),
        "minimum_pairs_required": MIN_PAIRS,
        "ready_to_train": pairs >= MIN_PAIRS,
        "categories": by_category,
        "latest_run": dict(latest) if latest else None,
        "message": (
            "Preference model trained from recorded user choices." if models
            else f"Collect {max(0, MIN_PAIRS-pairs)} more A/B selections. No artificial training data is used."
        ),
    }


def train_preference_models(force=False):
    try:
        import joblib
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import GroupShuffleSplit
        from sklearn.metrics import accuracy_score
    except ImportError as exc:
        raise RuntimeError("Install scikit-learn and joblib to train the preference model.") from exc

    results = []
    categories = ["all"] + sorted({row["category"] for row in _load_rows()})
    for category in categories:
        rows = _load_rows(None if category == "all" else category)
        pair_ids = {row["interaction_id"] for row in rows}
        labels = {int(row["label"]) for row in rows}
        if len(pair_ids) < MIN_PAIRS or labels != {0, 1}:
            continue
        X = np.array([[float(json.loads(row["features_json"]).get(name, 0.0)) for name in FEATURES] for row in rows], dtype=float)
        y = np.array([int(row["label"]) for row in rows], dtype=int)
        groups = np.array([row["interaction_id"] for row in rows])
        # Keep both answers from the same A/B comparison in the same split.
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
        train_idx, test_idx = next(splitter.split(X, y, groups))
        if len(set(y[train_idx])) < 2:
            continue
        model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
        model.fit(X[train_idx], y[train_idx])
        accuracy = None
        if len(set(y[test_idx])) == 2:
            accuracy = float(accuracy_score(y[test_idx], model.predict(X[test_idx])))
        path = MODEL_DIR / f"{category}.joblib"
        joblib.dump({"model": model, "features": FEATURES, "category": category, "pair_count": len(pair_ids)}, path)
        detail = f"Trained on {len(pair_ids)} real comparisons; held-out comparison-group accuracy: {accuracy if accuracy is not None else 'not available'}"
        try:
            display_path = str(path.relative_to(ROOT))
        except ValueError:
            display_path = str(path)
        execute_query("""
            INSERT INTO learning_runs(id,category,example_count,accuracy,model_path,status,details,created_at)
            VALUES(?,?,?,?,?,?,?,?)
        """, (str(uuid.uuid4()), category, len(rows), accuracy, display_path, "trained", detail, current_time()))
        results.append({"category": category, "pairs": len(pair_ids), "rows": len(rows), "accuracy": accuracy, "model": display_path})
    return {"trained_models": results, "status": training_status()}



def invalidate_models():
    """Remove cached classifiers after source comparison rows are deleted."""
    removed = 0
    for path in MODEL_DIR.glob("*.joblib"):
        try:
            path.unlink()
            removed += 1
        except OSError:
            pass
    return removed

def _load_model(category):
    try:
        import joblib
    except ImportError:
        return None
    # Keep style learning category-specific. The global model is used only for
    # the general category, not as a blanket preference across unrelated tasks.
    names = (category, "all") if category == "general" else (category,)
    for name in names:
        path = MODEL_DIR / f"{name}.joblib"
        if path.exists():
            try:
                return joblib.load(path)
            except Exception:
                continue
    return None


def learned_style_guidance(category):
    """Translate model coefficients into cautious style hints; never invent preferences."""
    saved = _load_model(category)
    if not saved:
        return ""
    model = saved["model"]
    coef = model.coef_[0]
    positive = [(FEATURES[i], float(coef[i])) for i in range(len(FEATURES)) if coef[i] > 0.18]
    positive.sort(key=lambda item: item[1], reverse=True)
    labels = {
        "log_words": "a fuller level of detail", "headings": "clear section headings",
        "bullets": "organized bullet points", "numbered_steps": "numbered step-by-step guidance",
        "examples": "practical examples", "code_blocks": "code examples where relevant",
        "tables": "comparison tables where useful", "paragraphs": "well-separated paragraphs",
    }
    hints = [labels[name] for name, _ in positive[:4] if name in labels]
    if not hints:
        return ""
    return "Learned response-style signals from the user's past explicit A/B selections suggest: " + ", ".join(hints) + ". These are tendencies, not strict rules; follow the current request first."

# Architecture notes

## Request lifecycle

1. The browser submits text and optional files to FastAPI.
2. The file service extracts document text locally and stores uploaded originals under `uploads/` with an attachment record.
3. The Personalization Agent classifies category, subtask and requested output format.
4. The memory manager retrieves relevant profile preferences using category filtering, TF-IDF similarity and confidence/evidence/recency weighting. Optional sentence-transformer embeddings can replace TF-IDF when explicitly enabled.
5. The agent adds a limited number of recent turns and matching memory records to one context prompt.
6. Selected AI providers receive the same relevant context in parallel. Provider errors are returned independently; one working provider is enough to save the turn.
7. The user may choose a preferred answer. When both answers exist, the system creates two real supervised examples: chosen answer = 1 and alternative = 0.
8. Optional reason feedback creates user-confirmed preference memories.
9. The local ML pipeline trains a Logistic Regression response-style model after at least ten distinct pairwise comparisons.

## Data stores

- `conversations`: chat titles and timestamps.
- `interactions`: user prompts, provider responses, selected provider, task labels and attachment metadata.
- `memories`: user-controlled category-specific memory and preferences.
- `preference_examples`: structural answer features and A/B labels used for model training.
- `learning_runs`: model training history and held-out evaluation metrics.
- `attachments`: uploaded original files associated with interactions.
- `artifacts`: generated exports and images.

## Model boundary

Gemini and Groq are replaceable provider adapters. Personalization and memory orchestration is local Python code. The preference classifier is a separate scikit-learn model trained on the individual user's own explicit comparisons; it is not a foundation model and does not learn semantic user facts by itself.

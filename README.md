# Personal AI Context Engine

A local-first multi-AI workspace. Gemini and Groq answer the user's requests; the Personalization Agent classifies tasks, retrieves category-relevant memory, composes a compact user context, and learns response-style signals from explicit A/B choices.

## What is included

- **Multi-provider chat:** ask Gemini, Groq, or both in parallel.
- **Conversation management:** create, search, continue, rename, and delete conversations.
- **Personalization Agent:** task/category classification, subtask and output-format detection, context assembly, relevant memory retrieval, and conversation continuity.
- **Memory controls:** inspect, edit, and forget individual memories. User-confirmed memories are protected from being overwritten by inferred signals.
- **Optional feedback:** select the response you preferred; optionally confirm suggested reasons or write your own.
- **Preference-learning pipeline:** selected-vs-alternative response features become labeled examples. A category/global Logistic Regression model is trained only after at least ten real A/B comparisons. No fake training data is generated. Clear first-person preference statements (for example, “I prefer…” or “from now on…”) can also be captured locally as user-confirmed memory without calling another AI model.
- **File inputs:** PDF, PPTX, DOCX, TXT, Markdown, CSV, XLSX, PNG, JPG, JPEG, and WEBP. Documents are text-extracted locally; image files are passed to providers that have vision capability configured.
- **Exports:** turn an AI response into a formatted PDF, editable DOCX, PPTX, TXT, or Markdown file.
- **Image Studio:** optional image generation through a separately configured Gemini image model.
- **SQLite storage:** conversation and profile data remain in the local project database by default.

## Important note about training

The uploaded database contained 16 historical interactions and four provider selections. A selection is useful preference evidence only when both provider responses exist. The migration backfills eligible historical comparisons; it does not invent missing alternatives or fabricate labels.

The project includes a reproducible training pipeline at `backend/ml/train_preferences.py`. It extracts structural answer features (length, headings, bullets, numbered steps, examples, code blocks, tables, paragraphs), keeps both answers from an A/B pair in the same evaluation split, and reports held-out comparison-group accuracy where possible. The model requires at least ten distinct A/B comparisons before training. Until enough real data exists, the application uses explicit confirmed memories and transparent rule-based context assembly instead of pretending an ML model has learned a reliable profile.

## Requirements

- Python 3.10 or newer (Python 3.12/3.13 recommended).
- API keys for Gemini and Groq.
- Internet access only for external model API calls and initial package/model installation.

## Setup on macOS

Open Terminal in this project folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

Open `.env` and add your own keys:

```dotenv
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=your_available_gemini_text_model
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=your_available_groq_chat_model
```

Use model IDs that are enabled for your own provider accounts. The defaults in `.env.example` are examples from the original project configuration and may not be available to every account.

Start the application:

```bash
uvicorn backend.main:app --reload
```

Open `http://127.0.0.1:8000`.

API documentation is at `http://127.0.0.1:8000/docs` and service status is at `http://127.0.0.1:8000/health`.

## Image understanding and generation

- Gemini image understanding uses the configured Gemini model and the provider's supported image input capability.
- Groq image understanding requires `GROQ_VISION_MODEL` to be set to a vision-capable model. The text-only `openai/gpt-oss-20b` model in the original setup is not treated as a vision model.
- Image generation requires a separate image-generation model. Set `GEMINI_IMAGE_MODEL` to an image model that your Gemini API account can access. A text model ID is not automatically an image model.

If these optional model capabilities are not configured, the UI reports the limitation instead of silently returning a fake image or ignoring an image attachment.

## Optional stronger semantic retrieval

The default memory retrieval uses local TF-IDF cosine similarity with category, confidence, evidence-count, source, and recency signals. It has no embedding API cost. For sentence-transformer embeddings instead:

```bash
pip install -r requirements-embeddings.txt
```

Then set in `.env`:

```dotenv
ENABLE_SEMANTIC_EMBEDDINGS=true
SENTENCE_TRANSFORMER_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

The model is downloaded the first time it is used. If the model cannot load, retrieval falls back to TF-IDF.

## Training the preference model

1. Ask both assistants questions where both return a response.
2. Select the answer you genuinely prefer. This records a positive example for the selected response and a negative example for the alternative.
3. Optionally select reasons or enter your own feedback. This stores user-confirmed, category-specific memories.
4. Open **Learning** in the app and press **Train / refresh model** after the minimum data threshold is reached. Alternatively:

```bash
python -m backend.ml.train_preferences
```

Training artifacts are stored in `data/models/`. They are local and ignored by Git. New selections can be used for subsequent training runs.

## Data and privacy

- `.env` contains provider secrets and is excluded from the project ZIP and Git.
- `memory.db` is the SQLite database. Back it up before major changes.
- Uploaded files are stored in `uploads/` and linked to their conversation interaction. Deleting a conversation also deletes its uploaded attachments.
- Generated exports are stored in `generated/` and can be removed manually when no longer needed.
- Selecting a response records a pairwise preference example. Saving optional reasons creates user-confirmed memories. Users can inspect, edit, or forget memories in the Memory view.
- Before context is sent to an external AI provider, it includes the current request, limited recent conversation context, relevant preference memories, and extracted attachment text. Avoid uploading sensitive files unless you are comfortable sending their contents to the selected provider.

## API overview

| Route | Purpose |
|---|---|
| `POST /ask` | Ask Gemini, Groq, or both |
| `POST /ask-with-files` | Ask with PDF/PPTX/DOCX/image and other supported attachments |
| `GET /conversations` | List/search conversations |
| `GET /conversations/{id}` | Read conversation and messages |
| `PATCH /conversations/{id}` | Rename a conversation |
| `DELETE /conversations/{id}` | Delete conversation, messages, and its attachments |
| `POST /preference` | Select preferred response and store pairwise signal |
| `POST /preference-feedback` | Save optional confirmed reasons |
| `GET /memory` | List memories |
| `PATCH /memory/{id}` | Edit a memory |
| `DELETE /memory/{id}` | Forget a memory |
| `POST /agent/analyze` | Inspect task classification and memory retrieval |
| `GET /learning/status` | Inspect available training data/models |
| `POST /learning/train` | Train eligible preference models |
| `POST /exports` | Export a response as PDF/DOCX/PPTX/TXT/MD |
| `POST /generate-image` | Generate an image with configured Gemini image model |

## Tests

```bash
pytest -q
python -m compileall -q backend
```

The test suite uses mocked model responses and does not require API keys or spend provider credits.

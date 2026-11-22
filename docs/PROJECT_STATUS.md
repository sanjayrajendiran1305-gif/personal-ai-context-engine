# Packaged project status

- Baseline source: user's uploaded `personal-ai-memory.zip`.
- Preserved local SQLite database: 5 conversations and 16 interactions at packaging time.
- Migrated user-confirmed memories from earlier optional feedback: 6 memories across general/shopping categories.
- Pairwise preference dataset: 3 eligible historical A/B comparisons (6 labeled response rows).
- Required minimum for a first real user-specific classifier: 10 distinct A/B comparisons.
- Trained production preference model: **not yet trained**, because the current user data is below the minimum threshold. The training implementation is included and tested with isolated temporary fixtures only; those fixtures are not in the production database.
- AI provider API calls: require the user to configure their own Gemini and Groq keys in `.env`.
- Image generation: requires a separately configured Gemini image model. The original text models are not assumed to generate images.
- Automated tests: 9 passing tests, including chat + preference lifecycle, conversation deletion, attachment handling, PDF/DOCX/PPTX exports, PDF/PPTX/DOCX/XLSX parsing, and the pairwise training pipeline.

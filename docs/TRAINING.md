# Preference model training protocol

The response-style model is trained only from actual user choices.

For each comparison where both Gemini and Groq responses are present and the user chooses one provider:

- Extract eight explainable features from each answer: log word count, headings, bullets, numbered steps, examples, code blocks, tables and paragraph count.
- Label the chosen provider's response as `1` and the alternative as `0`.
- Keep both responses tied to the same interaction ID.
- Train a Logistic Regression classifier when at least ten distinct comparison pairs exist.
- Use a group holdout split by interaction ID so the two answers from one comparison cannot leak across train and test sets.
- Record held-out accuracy when both labels are present in the test split.

The current uploaded database contains three eligible historic A/B comparisons (six labeled responses). That is below the ten-pair minimum, so this package does not claim to contain a trained user-specific model. It includes those historic comparisons, the training pipeline, and an interface that reports how many additional comparisons are needed.

This protocol is intentionally conservative: three pairs are not enough to support a robust personalization model. Collect more choices, then run `python -m backend.ml.train_preferences` or use the Learning page.

"""Ephemeral synthetic fixture verifies the trainer only; never enters the shipped user database."""
from backend.database import database
from backend.ml import preference_model


def test_pairwise_preference_model_trains_after_threshold(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "training_test.db")
    monkeypatch.setattr(preference_model, "MODEL_DIR", tmp_path / "models")
    preference_model.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    database.init_db()

    for index in range(10):
        if index % 2 == 0:
            gemini = "## Detailed guide\n\n" + ("A practical example with several clear steps.\n" * 12)
            groq = "Short answer."
            selected = "gemini"
        else:
            gemini = "Short answer."
            groq = "## Detailed guide\n\n" + ("A practical example with several clear steps.\n" * 12)
            selected = "groq"
        preference_model.record_pair(
            f"fixture-{index}", "programming", selected, gemini, groq
        )

    status = preference_model.training_status()
    assert status["ready_to_train"] is True
    result = preference_model.train_preference_models()
    assert any(model["category"] == "all" for model in result["trained_models"])
    assert any(model["category"] == "programming" for model in result["trained_models"])
    assert (preference_model.MODEL_DIR / "programming.joblib").exists()

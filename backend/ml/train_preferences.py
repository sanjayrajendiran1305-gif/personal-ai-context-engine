from backend.ml.preference_model import train_preference_models, training_status

if __name__ == "__main__":
    print("Current training data:")
    print(training_status())
    print("Training eligible preference models...")
    print(train_preference_models())

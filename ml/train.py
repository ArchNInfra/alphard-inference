from pathlib import Path

import joblib
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODELS_DIR / "model-latest.pkl"
N_FEATURES = 10


def main() -> None:
    X, y = make_classification(
        n_samples=1000,
        n_features=N_FEATURES,
        n_informative=8,
        n_redundant=2,
        random_state=42,
    )

    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)

    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()

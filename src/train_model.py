"""
Train an IsolationForest (unsupervised) on the credit-card dataset
and persist both model and scaler.
"""
import joblib
from pathlib import Path
from sklearn.ensemble import IsolationForest
from src.preprocess import load_data, train_test_split_scaled

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "model.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"

def main():
    df = load_data()
    X_train, _, _, _, scaler = train_test_split_scaled(df, test_size=0.2)

    model = IsolationForest(
        n_estimators=200,
        contamination=0.001,  # ≈ fraction of fraud in dataset
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"✅ Model saved to {MODEL_PATH}\n✅ Scaler saved to {SCALER_PATH}")


if __name__ == "__main__":
    main()

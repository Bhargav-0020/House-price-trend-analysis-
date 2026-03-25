import os

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


PROJECT_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(PROJECT_DIR, "dataset", "housing.csv")
MODEL_DIR = os.path.join(PROJECT_DIR, "model")


FEATURE_COLUMNS = [
    "area",
    "bedrooms",
    "bathrooms",
    "stories",
    "parking",
    "mainroad",
    "guestroom",
    "basement",
    "hotwaterheating",
    "airconditioning",
    "location",
]


def train_and_save_models():
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    # Match the preprocessing used in `train_model.ipynb`:
    # convert yes/no columns + location categories into numeric values.
    binary_cols = ["mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning"]
    for col in binary_cols:
        original = df[col]
        mapped = original.map({"yes": 1, "no": 0})
        df[col] = mapped.fillna(original).astype(int)

    df["location"] = df["location"].map({"Urban": 2, "Semi-Urban": 1, "Rural": 0}).astype(int)

    X = df[FEATURE_COLUMNS]
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Keep the same preprocessing approach as the original notebook + app.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "linear_model": LinearRegression(),
        "ridge_model": Ridge(alpha=1.0),
        "random_forest_model": RandomForestRegressor(
            n_estimators=300, random_state=42
        ),
        "gradient_boosting_model": GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05, random_state=42
        ),
    }

    results = {}
    for model_name, model in models.items():
        model.fit(X_train_scaled, y_train)
        pred = model.predict(X_test_scaled)

        results[model_name] = {
            "r2": float(r2_score(y_test, pred)),
            "mae": float(mean_absolute_error(y_test, pred)),
            "mse": float(mean_squared_error(y_test, pred)),
        }

        print(f"{model_name}: R2={results[model_name]['r2']:.4f}, MAE={results[model_name]['mae']:.2f}, MSE={results[model_name]['mse']:.2f}")

    # Persist scaler + all models with names expected by `app.py`.
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(models["linear_model"], os.path.join(MODEL_DIR, "model.pkl"))
    joblib.dump(models["ridge_model"], os.path.join(MODEL_DIR, "ridge_model.pkl"))
    joblib.dump(
        models["random_forest_model"], os.path.join(MODEL_DIR, "random_forest_model.pkl")
    )
    joblib.dump(
        models["gradient_boosting_model"],
        os.path.join(MODEL_DIR, "gradient_boosting_model.pkl"),
    )

    print("Saved: scaler.pkl + 4 regression models in `model/`.")


if __name__ == "__main__":
    train_and_save_models()



import pandas as pd
import joblib
from pathlib import Path

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_PATH = Path("data/processed/electricity_demand_features.csv")
MODEL_DIR = Path("models")
OUTPUT_DIR = Path("outputs/model_results")

MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"])

features = [
    "day_of_week", "month", "year", "day_of_year",
    "temperature", "humidity", "holiday",
    "hour", "day", "quarter", "is_weekend",
    "lag_1", "lag_24", "lag_168",
    "rolling_mean_24", "rolling_mean_168"
]

X = df[features]
y = df["demand"]

# Chronological split: first 80% train, last 20% test
split = int(len(df) * 0.8)

X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=100, max_depth=15,
        random_state=42, n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=100, learning_rate=0.05,
        max_depth=3, random_state=42
    )
}

results = []

for name, model in models.items():

    print("\nTraining:", name)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

    print("MAE:", round(mae, 3))
    print("RMSE:", round(rmse, 3))
    print("R2:", round(r2, 4))

    safe_name = name.lower().replace(" ", "_")
    joblib.dump(model, MODEL_DIR / f"{safe_name}.pkl")

results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

print("\nMODEL COMPARISON")
print(results_df.to_string(index=False))
print("\nMODELS SAVED SUCCESSFULLY")

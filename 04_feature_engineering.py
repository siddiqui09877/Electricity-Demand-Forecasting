
import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/processed/electricity_demand_cleaned.csv")
OUTPUT_PATH = Path("data/processed/electricity_demand_features.csv")

df = pd.read_csv(DATA_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"])

df = df.sort_values("timestamp").reset_index(drop=True)

# Time-based features
df["hour"] = df["timestamp"].dt.hour
df["day"] = df["timestamp"].dt.day
df["day_of_week"] = df["timestamp"].dt.dayofweek
df["month"] = df["timestamp"].dt.month
df["quarter"] = df["timestamp"].dt.quarter
df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

# Historical demand features
df["lag_1"] = df["demand"].shift(1)
df["lag_24"] = df["demand"].shift(24)
df["lag_168"] = df["demand"].shift(168)

# Rolling average features
df["rolling_mean_24"] = df["demand"].shift(1).rolling(24).mean()
df["rolling_mean_168"] = df["demand"].shift(1).rolling(168).mean()

# Remove rows with missing lag values
df = df.dropna().reset_index(drop=True)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)

print("FEATURE ENGINEERING COMPLETED")
print("Dataset shape:", df.shape)
print("New features:")
print(df.columns.tolist())
print("Saved to:", OUTPUT_PATH)

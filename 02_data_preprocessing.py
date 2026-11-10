
import pandas as pd
from pathlib import Path

print("=" * 60)
print("       ELECTRICITY DEMAND DATA PREPROCESSING")
print("=" * 60)

# Load original dataset
DATA_PATH = Path("data/electricity_demand_synthetic_2020_2024.csv")

df = pd.read_csv(DATA_PATH)

# 1. Convert timestamp into datetime
df["timestamp"] = pd.to_datetime(df["timestamp"])

# 2. Sort records chronologically
df = df.sort_values("timestamp").reset_index(drop=True)

# 3. Check date range
print("\n1. DATE RANGE")
print("Start:", df["timestamp"].min())
print("End:", df["timestamp"].max())

# 4. Check timestamp duplicates
print("\n2. DUPLICATE TIMESTAMPS")
print(df["timestamp"].duplicated().sum())

# 5. Check time interval
print("\n3. TIME INTERVAL")
print(df["timestamp"].diff().value_counts().head())

# 6. Check missing values
print("\n4. MISSING VALUES")
print(df.isnull().sum())

# 7. Check demand statistics
print("\n5. DEMAND STATISTICS")
print(df["demand"].describe())

# 8. Check negative demand
print("\n6. NEGATIVE DEMAND VALUES")
print((df["demand"] < 0).sum())

# 9. Create processed data folder
output_folder = Path("data/processed")
output_folder.mkdir(parents=True, exist_ok=True)

# 10. Save processed dataset
output_path = output_folder / "electricity_demand_cleaned.csv"
df.to_csv(output_path, index=False)

print("\n7. PROCESSED DATA SAVED")
print(output_path)

print("\n" + "=" * 60)
print("       PREPROCESSING COMPLETED")
print("=" * 60)

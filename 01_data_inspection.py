
import pandas as pd
from pathlib import Path

# Project dataset path
DATA_PATH = Path("data/electricity_demand_synthetic_2020_2024.csv")

print("=" * 60)
print("       ELECTRICITY DEMAND DATASET INSPECTION")
print("=" * 60)

# Load dataset
df = pd.read_csv(DATA_PATH)

# 1. Dataset dimensions
print("\n1. DATASET DIMENSIONS")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

# 2. Column names
print("\n2. COLUMN NAMES")
print(df.columns.tolist())

# 3. First five records
print("\n3. FIRST FIVE RECORDS")
print(df.head())

# 4. Data types
print("\n4. DATA TYPES")
print(df.dtypes)

# 5. Missing values
print("\n5. MISSING VALUES")
print(df.isnull().sum())

# 6. Duplicate records
print("\n6. DUPLICATE RECORDS")
print(df.duplicated().sum())

# 7. Statistical summary
print("\n7. STATISTICAL SUMMARY")
print(df.describe(include="all").T)

print("\n" + "=" * 60)
print("       DATASET INSPECTION COMPLETED")
print("=" * 60)

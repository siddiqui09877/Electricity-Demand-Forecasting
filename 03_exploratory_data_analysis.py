
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Project paths
DATA_PATH = Path("data/processed/electricity_demand_cleaned.csv")
OUTPUT_DIR = Path("outputs/eda")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"])

sns.set_theme(style="whitegrid")

print("=" * 60)
print("       EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# 1. Hourly demand pattern
df["hour"] = df["timestamp"].dt.hour
hourly = df.groupby("hour")["demand"].mean()

plt.figure(figsize=(12, 5))
hourly.plot(marker="o")
plt.title("Average Electricity Demand by Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Average Demand")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "hourly_demand.png", dpi=300)
plt.show()

# 2. Daily average demand
daily = df.set_index("timestamp")["demand"].resample("D").mean()

plt.figure(figsize=(14, 5))
daily.plot()
plt.title("Daily Average Electricity Demand")
plt.xlabel("Date")
plt.ylabel("Average Demand")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "daily_demand.png", dpi=300)
plt.show()

# 3. Monthly demand pattern
monthly = df.groupby("month")["demand"].mean()

plt.figure(figsize=(12, 5))
monthly.plot(kind="bar")
plt.title("Average Electricity Demand by Month")
plt.xlabel("Month")
plt.ylabel("Average Demand")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "monthly_demand.png", dpi=300)
plt.show()

# 4. Yearly demand pattern
yearly = df.groupby("year")["demand"].mean()

plt.figure(figsize=(10, 5))
yearly.plot(kind="bar")
plt.title("Average Electricity Demand by Year")
plt.xlabel("Year")
plt.ylabel("Average Demand")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "yearly_demand.png", dpi=300)
plt.show()

# 5. Demand distribution
plt.figure(figsize=(10, 5))
sns.histplot(df["demand"], kde=True)
plt.title("Electricity Demand Distribution")
plt.xlabel("Demand")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "demand_distribution.png", dpi=300)
plt.show()

# 6. Correlation heatmap
numeric_df = df.select_dtypes(include="number")

plt.figure(figsize=(10, 7))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "correlation_heatmap.png", dpi=300)
plt.show()

# 7. Print important statistics
print("\nHOURLY AVERAGE DEMAND")
print(hourly)

print("\nMONTHLY AVERAGE DEMAND")
print(monthly)

print("\nYEARLY AVERAGE DEMAND")
print(yearly)

print("\nHIGHEST DEMAND:", df["demand"].max())
print("LOWEST DEMAND:", df["demand"].min())
print("AVERAGE DEMAND:", df["demand"].mean())

print("\nEDA COMPLETED SUCCESSFULLY")
print("Graphs saved in:", OUTPUT_DIR)

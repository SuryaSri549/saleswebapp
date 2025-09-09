# synth_marketing_spend_v2.py
import numpy as np
import pandas as pd

SRC = "superstore_extended.csv"          # or "cleaned_superstore.csv" if that's your base
DST = "superstore_extended.csv"          # overwrite with better spend (ok)
np.random.seed(42)

# --- Load & basic clean ---
df = pd.read_csv(SRC)
need = {"Order Date", "Sales"}
if not need.issubset(df.columns):
    missing = ", ".join(sorted(need - set(df.columns)))
    raise ValueError(f"Missing required columns: {missing}")

df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce")
if "Discount" in df.columns:
    df["Discount"] = pd.to_numeric(df["Discount"], errors="coerce")
else:
    df["Discount"] = 0.0
df = df.dropna(subset=["Order Date", "Sales"]).copy()

have_region = "Region" in df.columns
have_subcat = "Sub-Category" in df.columns

# --- Keys for grouping ---
df["Month"] = df["Order Date"].dt.to_period("M").dt.to_timestamp()
grp_keys = ["Month"]
if have_region: grp_keys.append("Region")
if have_subcat: grp_keys.append("Sub-Category")

# --- Seasonality multipliers by month (customize if you like) ---
seasonality = {
    1:0.95,  2:0.95, 3:1.00, 4:1.00, 5:1.05, 6:1.05,
    7:1.00,  8:1.00, 9:1.08, 10:1.12, 11:1.22, 12:1.30  # stronger Q4 push
}

# --- Region multipliers (tweak to your reality) ---
region_mult = {"Central":0.98, "East":1.03, "South":1.00, "West":1.05}

# --- Base marketing intensity as % of sales (choose a band) ---
base_low, base_high = 0.05, 0.15  # 5%–15% of sales on average

# --- Build monthly group stats ---
g = df.groupby(grp_keys).agg(
    monthly_sales=("Sales", "sum"),
    avg_discount=("Discount", "mean"),
    rows=("Sales", "size")
).reset_index()

# Controls: higher discount → more spend (but cap)
# Example: promo_factor = 1 + 1.2 * avg_discount_fraction (where discount given as 0–1)
g["avg_discount"] = g["avg_discount"].clip(lower=0)
g["promo_factor"] = 1 + 1.2 * np.where(g["avg_discount"] <= 1, g["avg_discount"], g["avg_discount"] / 100.0)

# Seasonality & region factors
g["season_factor"] = g["Month"].dt.month.map(seasonality).fillna(1.0)
if have_region:
    g["region_factor"] = g["Region"].map(region_mult).fillna(1.0)
else:
    g["region_factor"] = 1.0

# Base share 5–15% with small noise
base_prop = np.random.uniform(base_low, base_high, size=len(g))
noise = np.random.normal(1.0, 0.06, size=len(g))  # mild noise

# Monthly group spend budget
g["group_spend"] = (
    g["monthly_sales"]
    * base_prop
    * g["season_factor"]
    * g["region_factor"]
    * g["promo_factor"]
    * noise
).clip(lower=0)

# --- Allocate group spend down to rows proportionally to their sales share ---
# Merge group budget back to rows
df = df.merge(g[grp_keys + ["monthly_sales", "group_spend"]], on=grp_keys, how="left")

# Avoid div by zero
df["share"] = np.where(df["monthly_sales"] > 0, df["Sales"] / df["monthly_sales"], 0.0)
df["Marketing Spend"] = (df["group_spend"] * df["share"]).fillna(0.0)

# Optional: jitter per-row spend slightly so duplicates aren't identical
df["Marketing Spend"] *= np.random.normal(1.0, 0.03, size=len(df))
df["Marketing Spend"] = df["Marketing Spend"].clip(lower=0)

# Tidy up
df.drop(columns=["Month", "monthly_sales", "group_spend", "share"], inplace=True, errors="ignore")
df.to_csv(DST, index=False)
print(f"✅ Wrote realistic Marketing Spend to {DST}")

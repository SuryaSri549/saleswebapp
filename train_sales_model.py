# train_sales_model.py
import pickle
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

SRC = "superstore_extended.csv"
OUT = "sales_model.pkl"

def main():
    # -------- Load --------
    try:
        df = pd.read_csv(SRC)
    except Exception as e:
        raise RuntimeError(f"Failed to read {SRC}: {e}")

    # -------- Validate columns --------
    required = {"Sales", "Marketing Spend"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {SRC}: {', '.join(sorted(missing))}")

    # Convert
    df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce")
    df["Marketing Spend"] = pd.to_numeric(df["Marketing Spend"], errors="coerce")

    # Discount handling (0–1 or 0–100). If not present, assume 0.
    if "Discount" in df.columns:
        df["Discount"] = pd.to_numeric(df["Discount"], errors="coerce").fillna(0)
        # If most discounts look like percentages > 1, convert to fraction
        if df["Discount"].quantile(0.95) > 1.5:
            df["DiscountFrac"] = (df["Discount"] / 100.0).clip(0, 1)
        else:
            df["DiscountFrac"] = df["Discount"].clip(0, 1)
    else:
        df["DiscountFrac"] = 0.0

    # Region one-hots in fixed order used by /predict
    regions = ["Central", "East", "South", "West"]
    for r in regions:
        df[f"Region_{r}"] = (df["Region"] == r).astype(int) if "Region" in df.columns else 0

    # Drop rows with missing essentials
    df = df.dropna(subset=["Sales", "Marketing Spend"]).copy()

    # -------- Features (EXACT order expected by /predict) --------
    X_cols = ["Marketing Spend", "DiscountFrac"] + [f"Region_{r}" for r in regions]
    X = df[X_cols].astype("float64").values
    y = df["Sales"].astype("float64").values

    # Optional: clip extreme outliers in y to stabilize training
    if len(y) > 50:
        y_hi = np.nanpercentile(y, 99.5)
        y = np.clip(y, 0, y_hi)

    # -------- Train/test split --------
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    # -------- Pipeline: scaling + ridge regression --------
    pipe = Pipeline(steps=[
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("model", Ridge(alpha=1.0, random_state=42))
    ])

    pipe.fit(X_tr, y_tr)

    # -------- Eval --------
    y_pred = pipe.predict(X_te)
    r2 = r2_score(y_te, y_pred) if len(y_te) else float("nan")
    mae = mean_absolute_error(y_te, y_pred) if len(y_te) else float("nan")

    print(f"R^2:  {r2:.3f}")
    print(f"MAE:  {mae:,.2f}")

    # -------- Save pipeline --------
    with open(OUT, "wb") as f:
        pickle.dump(pipe, f)

    print(f"\n✅ Sales prediction model trained and saved as {OUT}")
    print("   Features order:", X_cols)

if __name__ == "__main__":
    main()

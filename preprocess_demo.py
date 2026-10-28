"""Week 2: data collection simulation, cleaning and preprocessing for logistics data.
Uses DataCo-style column names. If the real CSV is missing, a messy synthetic sample is generated
so the pipeline can be tested end to end."""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# ---------- 1. Data collection (simulated) ----------
def load_raw(path="DataCoSupplyChainDataset.csv", n=5000, seed=7):
    try:
        return pd.read_csv(path, encoding="latin-1")
    except FileNotFoundError:
        rng = np.random.default_rng(seed)
        modes = ["Standard Class", "Second Class", "First Class", "Same Day"]
        sched = {"Standard Class": 4, "Second Class": 2, "First Class": 1, "Same Day": 0}
        df = pd.DataFrame({
            "Order Id": np.arange(1, n + 1),
            "Shipping Mode": rng.choice(modes, n, p=[.6, .2, .15, .05]),
            "Order Region": rng.choice(["Western Europe", "Central America", "South Asia", "Oceania"], n),
            "Order Item Quantity": rng.integers(1, 6, n).astype(float),
            "Sales": rng.gamma(4, 50, n),
            "Order Item Discount Rate": rng.uniform(0, .25, n),
            "order date (DateOrders)": pd.date_range("2024-01-01", periods=n, freq="2h").strftime("%m/%d/%Y %H:%M"),
        })
        df["Days for shipment (scheduled)"] = df["Shipping Mode"].map(sched).astype(float)
        df["Days for shipping (real)"] = (df["Days for shipment (scheduled)"] + rng.integers(-1, 3, n)).clip(lower=0)
        df["Order Profit Per Order"] = df["Sales"] * (0.25 - df["Order Item Discount Rate"]) + rng.normal(0, 10, n)
        # ---- inject realistic quality problems ----
        for col, frac in [("Sales", .04), ("Order Item Quantity", .03), ("Days for shipping (real)", .05),
                          ("Order Region", .02), ("Order Item Discount Rate", .03)]:
            df.loc[rng.choice(n, int(frac * n), replace=False), col] = np.nan
        idx = rng.choice(n, 60, replace=False)
        df.loc[idx, "Sales"] = df["Sales"].median() * rng.uniform(15, 40, 60)       # extreme outliers
        df.loc[rng.choice(n, 25, replace=False), "Order Item Quantity"] = -1         # invalid quantity
        df.loc[rng.choice(n, 30, replace=False), "Order Item Discount Rate"] = 1.8   # impossible rate
        m = rng.choice(n, 400, replace=False)
        df.loc[m[:200], "Shipping Mode"] = df.loc[m[:200], "Shipping Mode"].str.upper()   # inconsistent case
        df.loc[m[200:], "Shipping Mode"] = " " + df.loc[m[200:], "Shipping Mode"] + " "   # stray spaces
        df.loc[rng.choice(n, 40, replace=False), "order date (DateOrders)"] = "not available"
        df = pd.concat([df, df.sample(75, random_state=1)], ignore_index=True)       # duplicate rows
        return df

# ---------- 2. Data quality audit ----------
def audit(df, label):
    miss = df.isna().sum()
    print(f"--- Audit: {label} ---")
    print("rows:", len(df), "| duplicates:", int(df.duplicated().sum()), "| missing cells:", int(miss.sum()))
    print(miss[miss > 0].to_string())
    return miss

# ---------- 3. Cleaning ----------
def clean_text_and_types(df):
    df = df.copy()
    df["Shipping Mode"] = df["Shipping Mode"].str.strip().str.title()
    df["Order Region"] = df["Order Region"].str.strip().str.title()
    df["order date (DateOrders)"] = pd.to_datetime(df["order date (DateOrders)"], errors="coerce")
    return df

def drop_duplicates_and_invalid(df):
    n0 = len(df)
    df = df.drop_duplicates()
    removed_dup = n0 - len(df)
    bad_qty = df["Order Item Quantity"] <= 0
    bad_disc = (df["Order Item Discount Rate"] < 0) | (df["Order Item Discount Rate"] > 1)
    df.loc[bad_qty, "Order Item Quantity"] = np.nan          # impossible -> treat as missing
    df.loc[bad_disc, "Order Item Discount Rate"] = np.nan
    print(f"duplicates removed: {removed_dup} | invalid quantity set to NaN: {int(bad_qty.sum())} | invalid discount set to NaN: {int(bad_disc.sum())}")
    return df

def handle_missing(df):
    df = df.copy()
    n0 = len(df)
    df = df.dropna(subset=["order date (DateOrders)"])        # key field: cannot be guessed
    print("rows dropped (no valid order date):", n0 - len(df))
    for c in ["Sales", "Order Item Quantity", "Days for shipping (real)", "Order Item Discount Rate"]:
        df[c] = df[c].fillna(df[c].median())                   # median: robust to outliers
    df["Order Region"] = df["Order Region"].fillna(df["Order Region"].mode()[0])
    return df

def iqr_bounds(s, k=1.5):
    q1, q3 = s.quantile([.25, .75]); iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr

def handle_outliers(df, col="Sales"):
    lo, hi = iqr_bounds(df[col])
    n_out = int(((df[col] < lo) | (df[col] > hi)).sum())
    z = (df[col] - df[col].mean()) / df[col].std()
    print(f"{col}: IQR bounds = [{lo:.1f}, {hi:.1f}] | outliers (IQR) = {n_out} | outliers (|z|>3) = {int((z.abs() > 3).sum())}")
    df[col + "_capped"] = df[col].clip(lo, hi)                 # winsorise: keep the row, limit the extreme
    return df

# ---------- 4. Feature preparation ----------
def feature_prep(df):
    df = df.copy()
    df["delay_days"] = df["Days for shipping (real)"] - df["Days for shipment (scheduled)"]
    df["Late_delivery_risk"] = (df["delay_days"] > 0).astype(int)
    df["order_month"] = df["order date (DateOrders)"].dt.month
    df = pd.get_dummies(df, columns=["Shipping Mode", "Order Region"], drop_first=True)
    return df

def normalise(df, cols):
    df = df.copy()
    mm = MinMaxScaler().fit(df[cols])
    zs = StandardScaler().fit(df[cols])
    df[[c + "_minmax" for c in cols]] = mm.transform(df[cols])
    df[[c + "_z" for c in cols]] = zs.transform(df[cols])
    return df

# ---------- 5. Validation ----------
def validate(df):
    assert df["Order Id"].is_unique
    assert (df["Order Item Quantity"] > 0).all()
    assert df["Order Item Discount Rate"].between(0, 1).all()
    assert df.drop(columns=[]).isna().sum().sum() == 0
    print("validation passed: no duplicates, no missing values, all values in valid range")

if __name__ == "__main__":
    raw = load_raw()
    audit(raw, "raw")
    sales_before = raw["Sales"].describe().round(1)
    df = clean_text_and_types(raw)
    df = drop_duplicates_and_invalid(df)
    df = handle_missing(df)
    df = handle_outliers(df, "Sales")
    df = feature_prep(df)
    df = normalise(df, ["Sales_capped", "Order Item Quantity", "Days for shipping (real)"])
    audit(df, "clean")
    validate(df)
    print("Sales before:\n", sales_before.to_string())
    print("Sales after (capped):\n", df["Sales_capped"].describe().round(1).to_string())
    print("Min-max range:", df["Sales_capped_minmax"].min().round(2), "to", df["Sales_capped_minmax"].max().round(2),
          "| z-score mean/std:", df["Sales_capped_z"].mean().round(2), df["Sales_capped_z"].std().round(2))
    df.to_csv("clean_logistics_data.csv", index=False)
    fig, ax = plt.subplots(1, 2, figsize=(7, 3), sharey=False)
    ax[0].boxplot(raw["Sales"].dropna(), orientation="vertical"); ax[0].set_title("Sales: before"); ax[0].set_xticks([])
    ax[1].boxplot(df["Sales_capped"], orientation="vertical"); ax[1].set_title("Sales: after capping"); ax[1].set_xticks([])
    ax[0].set_ylabel("Sales"); plt.tight_layout(); plt.savefig("sales_before_after.png", dpi=170)

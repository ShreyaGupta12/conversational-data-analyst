# tools.py
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional
import difflib

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "startup_dataset.csv")
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots")

# alias map: user-friendly -> actual column name in CSV
ALIAS_MAP = {
    "funding": "Total Funding (USD; In Millions)",
    "total funding": "Total Funding (USD; In Millions)",
    "valuation": "Valuation (USD; In Millions)",
    "employees": "Employee Count",
    "employee count": "Employee Count",
    "founded": "Founded Year",
    "cohort": "Cohort",
    "country": "Country",
    "stage": "Stage",
    # add more aliases here if desired
}


def load_data() -> pd.DataFrame:
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Place startup_dataset.csv there.")
    # keep original column names, don't modify df in place
    return pd.read_csv(DATA_PATH)


def _normalize(s: str) -> str:
    return "".join(c for c in s.lower() if c.isalnum())


def find_column(user_name: str, df: pd.DataFrame) -> Optional[str]:
    """
    Try to match user_name to an actual column in df.
    Strategy:
     1) alias map (case-insensitive)
     2) exact name match
     3) case-insensitive match
     4) normalized match (remove non-alnum)
     5) difflib.get_close_matches on normalized names
     6) token containment
    """
    if user_name is None:
        return None

    user_name_str = user_name.strip()
    if not user_name_str:
        return None

    # 1) alias map
    lower_alias = user_name_str.lower()
    if lower_alias in ALIAS_MAP:
        target = ALIAS_MAP[lower_alias]
        if target in df.columns:
            return target

    cols = list(df.columns)

    # 2) exact
    if user_name_str in cols:
        return user_name_str

    # 3) case-insensitive
    lower_map = {c.lower(): c for c in cols}
    if user_name_str.lower() in lower_map:
        return lower_map[user_name_str.lower()]

    # 4) normalized match
    norm_map = {_normalize(c): c for c in cols}
    un = _normalize(user_name_str)
    if un in norm_map:
        return norm_map[un]

    # 5) fuzzy match on normalized names
    choices = list(norm_map.keys())
    matches = difflib.get_close_matches(un, choices, n=1, cutoff=0.6)
    if matches:
        return norm_map[matches[0]]

    # 6) token containment: if user gives 'employee' match 'Employee Count', etc.
    tokens = [t for t in user_name_str.replace("_", " ").split()]
    for t in tokens:
        for c in cols:
            if t.lower() in c.lower():
                return c

    return None


def clean_feature(feature_name: str, strategy: str = "mean") -> str:
    """
    Handle missing values in a column.
    strategy: "mean", "median", or "drop"
    Returns a short text summary of the action taken.
    """
    df = load_data()
    real_name = find_column(feature_name, df)
    if real_name is None:
        return f"Error: feature '{feature_name}' not found. Example columns: {', '.join(df.columns[:10])}"

    n_missing = df[real_name].isna().sum()
    if n_missing == 0:
        return f"No missing values found in '{real_name}'."

    if strategy.lower() == "drop":
        df_clean = df.dropna(subset=[real_name])
        df_clean.to_csv(DATA_PATH, index=False)
        return f"Dropped {n_missing} rows with missing '{real_name}'. Dataset overwritten."

    col = df[real_name]
    if pd.api.types.is_numeric_dtype(col):
        fill = col.median() if strategy.lower() == "median" else col.mean()
        df[real_name] = col.fillna(fill)
        df.to_csv(DATA_PATH, index=False)
        return f"Filled {n_missing} missing values in '{real_name}' with {strategy} ({fill}). Dataset overwritten."

    mode = col.mode()
    if len(mode) > 0:
        df[real_name] = col.fillna(mode[0])
        df.to_csv(DATA_PATH, index=False)
        return f"Filled {n_missing} missing values in '{real_name}' with mode '{mode[0]}'. Dataset overwritten."

    return "Unable to clean the feature with provided strategy."


def plot_histogram(feature_name: str, bins: int = 20, out_file: Optional[str] = None) -> str:
    """
    Save a histogram plot for a numeric column. Returns path or error message.
    """
    df = load_data()
    real_name = find_column(feature_name, df)
    if real_name is None:
        return f"Error: feature '{feature_name}' not found. Example columns: {', '.join(df.columns[:10])}"

    if not pd.api.types.is_numeric_dtype(df[real_name]):
        return f"Error: feature '{real_name}' is not numeric and cannot be histogrammed."

    os.makedirs(PLOTS_DIR, exist_ok=True)
    if out_file is None:
        safe_name = "".join(c if c.isalnum() or c in (" ", "_", "-") else "_" for c in real_name)
        out_file = os.path.join(PLOTS_DIR, f"hist_{safe_name}.png")

    plt.figure()
    df[real_name].dropna().hist(bins=bins)
    plt.xlabel(real_name)
    plt.ylabel("Count")
    plt.title(f"Histogram of {real_name}")
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()
    return f"Saved histogram to {out_file}"


def calculate_correlation(feature_a: str, feature_b: str) -> str:
    """
    Calculates Pearson correlation between two numeric features.
    Returns textual summary and numeric value.
    """
    df = load_data()
    real_a = find_column(feature_a, df)
    real_b = find_column(feature_b, df)
    if real_a is None:
        return f"Error: feature '{feature_a}' not found. Example columns: {', '.join(df.columns[:10])}"
    if real_b is None:
        return f"Error: feature '{feature_b}' not found. Example columns: {', '.join(df.columns[:10])}"

    if not (pd.api.types.is_numeric_dtype(df[real_a]) and pd.api.types.is_numeric_dtype(df[real_b])):
        return "Error: both features must be numeric to calculate correlation."

    clean_df = df[[real_a, real_b]].dropna()
    if clean_df.shape[0] < 2:
        return "Not enough data after dropping missing values."

    corr = clean_df[real_a].corr(clean_df[real_b])
    return f"Pearson correlation between '{real_a}' and '{real_b}' = {corr:.4f} (N={clean_df.shape[0]})"

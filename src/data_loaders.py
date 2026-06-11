"""
Data loaders.

Tier A  -> folktables / US Census ACS  (downloads automatically; has SEX, AGEP, RAC1P)
Tier B  -> OpenML small classification datasets (downloads automatically)

Every loader returns a dict:
    {
      "X": np.ndarray [n, d]  (numeric, preprocessed),
      "y": np.ndarray [n]     (int label-encoded 0..K-1),
      "n_classes": int,
      "sensitive": {axis_name: np.ndarray[n] of group labels (strings)},
    }

Both sources cache to ./data automatically. Nothing to place by hand.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _subsample(X, y, sensitive, n, seed):
    rng = np.random.default_rng(seed)
    if n is not None and len(y) > n:
        idx = rng.choice(len(y), size=n, replace=False)
        X = X[idx]
        y = y[idx]
        sensitive = {k: v[idx] for k, v in sensitive.items()}
    return X, y, sensitive


def _age_band(age):
    bins = [-np.inf, 25, 40, 55, 65, np.inf]
    names = ["<25", "25-39", "40-54", "55-64", "65+"]
    return np.array(names)[np.digitize(age, bins[1:-1])]


def _race_label(code):
    # ACS RAC1P codes -> coarse, readable groups
    mapping = {1: "White", 2: "Black", 6: "Asian"}
    return np.array([mapping.get(int(c), "Other") for c in code])


def _preprocess_frame(X_df):
    """Ordinal-encode categoricals, median-impute numerics -> numeric matrix."""
    X_df = X_df.copy()
    num_cols = X_df.select_dtypes(include=[np.number]).columns
    cat_cols = [c for c in X_df.columns if c not in num_cols]

    parts = []
    if len(num_cols):
        num = SimpleImputer(strategy="median").fit_transform(X_df[num_cols])
        parts.append(num)
    if len(cat_cols):
        cat = X_df[cat_cols].astype("object").where(pd.notnull(X_df[cat_cols]), "NA")
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        parts.append(enc.fit_transform(cat))
    return np.hstack(parts).astype(np.float32)


# --------------------------------------------------------------------------- #
# Tier A : folktables
# --------------------------------------------------------------------------- #
def load_folktables(task="ACSPublicCoverage", state="CA", year="2018",
                    n=8000, seed=0):
    """task in {ACSIncome, ACSPublicCoverage, ACSEmployment}."""
    from folktables import ACSDataSource
    import folktables as ft

    problem = {
        "ACSIncome": ft.ACSIncome,
        "ACSPublicCoverage": ft.ACSPublicCoverage,
        "ACSEmployment": ft.ACSEmployment,
    }[task]

    src = ACSDataSource(survey_year=year, horizon="1-Year", survey="person",
                        root_dir=os.path.join(DATA_DIR, "folktables"))
    acs = src.get_data(states=[state], download=True)
    X_df, y_df, _ = problem.df_to_pandas(acs)

    y = LabelEncoder().fit_transform(np.asarray(y_df).ravel().astype(int))
    sensitive = {}
    if "SEX" in X_df.columns:
        sensitive["sex"] = np.where(np.asarray(X_df["SEX"]) == 1, "Male", "Female")
    if "AGEP" in X_df.columns:
        sensitive["age"] = _age_band(np.asarray(X_df["AGEP"], dtype=float))
    if "RAC1P" in X_df.columns:
        sensitive["race"] = _race_label(np.asarray(X_df["RAC1P"]))

    X = _preprocess_frame(X_df)
    X, y, sensitive = _subsample(X, y, sensitive, n, seed)
    return {"X": X, "y": y, "n_classes": int(len(np.unique(y))),
            "sensitive": sensitive}


# --------------------------------------------------------------------------- #
# Tier B : OpenML
# --------------------------------------------------------------------------- #
# Stable data_ids of small (<=10k) classification datasets.
OPENML_DATASETS = {
    "credit-g":   31,
    "diabetes":   37,
    "breast-w":   15,
    "ilpd":       1480,
    "blood-transfusion": 1464,
    "banknote":   1462,
    "kc1":        1067,
    "phoneme":    1489,
    "adult":      1590,   # has sex + race -> extra fairness axes
}


def load_openml(name, n=8000, seed=0):
    from sklearn.datasets import fetch_openml
    data_id = OPENML_DATASETS[name]
    ds = fetch_openml(data_id=data_id, as_frame=True, cache=True,
                      data_home=os.path.join(DATA_DIR, "openml"))
    X_df, y_ser = ds.data, ds.target
    y = LabelEncoder().fit_transform(np.asarray(y_ser).astype(str))

    sensitive = {}
    # opportunistic sensitive axes if present
    for col, axis in [("sex", "sex"), ("Sex", "sex"),
                      ("race", "race"), ("Race", "race")]:
        if col in X_df.columns:
            sensitive[axis] = np.asarray(X_df[col].astype(str))
    for col in ["age", "Age", "AGEP"]:
        if col in X_df.columns:
            try:
                sensitive["age"] = _age_band(np.asarray(X_df[col], dtype=float))
            except Exception:
                pass
            break

    X = _preprocess_frame(X_df)
    X, y, sensitive = _subsample(X, y, sensitive, n, seed)
    return {"X": X, "y": y, "n_classes": int(len(np.unique(y))),
            "sensitive": sensitive}


# --------------------------------------------------------------------------- #
# registry used by the experiment runner
# --------------------------------------------------------------------------- #
def dataset_registry():
    """Maps a display name -> (loader callable, kwargs, tier)."""
    reg = {}
    # Tier A (fairness headline)
    for task in ["ACSPublicCoverage", "ACSIncome", "ACSEmployment"]:
        reg[f"{task}-CA"] = (load_folktables, dict(task=task, state="CA"), "A")
    # Tier B (breadth)
    for name in OPENML_DATASETS:
        reg[f"openml-{name}"] = (load_openml, dict(name=name), "B")
    return reg

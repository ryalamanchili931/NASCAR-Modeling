import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterSampler
from sklearn.metrics import roc_auc_score

from scipy.stats import loguniform

from Nascar_Pipeline import classwise_ece


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

N_BINS = 20
N_ITER = 25  # random search iterations


# ─────────────────────────────────────────────
# TEMPORAL SPLIT (3-way)
# ─────────────────────────────────────────────

def temporal_split_3way(df):

    train = df[df["year"].isin([2022, 2023])].copy()
    val   = df[df["year"].isin([2024])].copy()
    test  = df[df["year"].isin([2025])].copy()

    return train, val, test


# ─────────────────────────────────────────────
# HYPERPARAM SEARCH (ECE OBJECTIVE)
# ─────────────────────────────────────────────

def tune_svm(train_df, val_df, feature_cols):
    print("\n── Hyperparameter tuning (ECE objective) ──")

    X_train = train_df[feature_cols].values
    y_train = train_df["y"].values
    X_val   = val_df[feature_cols].values
    y_val   = val_df["y"].values

    # scale ONLY on train
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)

    param_dist = {
        "C": loguniform(0.1, 50),
        "kernel": ["linear", "rbf"], # poly, sigmoid
        "degree": [2, 3, 4],
        "gamma": loguniform(1e-4, 10),
    }

    sampler = list(ParameterSampler(param_dist, n_iter=N_ITER, random_state=42))

    best_score = float("inf")
    best_params = None
    best_model = None

    for i, params in enumerate(sampler, 1):
        kernel = params["kernel"]

        # degree only matters for poly
        degree = params["degree"] if kernel == "poly" else 3

        # gamma only matters for rbf, poly, sigmoid
        gamma = params["gamma"] if kernel in ["rbf", "poly", "sigmoid"] else "scale"

        base = SVC(
            C=params["C"],
            kernel=kernel,
            degree=degree,
            gamma=gamma,
        )

        model = CalibratedClassifierCV(base, method="sigmoid", cv=3)

        model.fit(X_train, y_train)

        probs = model.predict_proba(X_val)[:, 1]
        ece = classwise_ece(y_val, probs, min_non_empty_bins=0.8)

        print(f"{i:02d} | kernel={kernel:<7} C={params['C']:.3f} "
              f"deg={degree} | ECE={ece:.5f}")
        
        if ece == 1.0:
            continue

        if ece < best_score:
            best_score = ece
            best_params = params
            best_model = model

    print("\nBest params:", best_params)
    print("Best validation ECE:", best_score)

    return best_params, scaler


# ─────────────────────────────────────────────
# FINAL TRAIN + TEST
# ─────────────────────────────────────────────

def train_final(train_df, val_df, test_df, feature_cols, best_params, scaler):
    print("\n── Final training ──")

    # combine train + val
    train_full = pd.concat([train_df, val_df])

    X_train = train_full[feature_cols].values
    y_train = train_full["y"].values
    X_test  = test_df[feature_cols].values
    y_test  = test_df["y"].values

    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    kernel = best_params["kernel"]
    degree = best_params["degree"] if kernel == "poly" else 3

    base = SVC(
        C=best_params["C"],
        kernel=kernel,
        degree=degree,
    )

    model = CalibratedClassifierCV(base, method="isotonic", cv=5)
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]

    ece = classwise_ece(y_test, probs)
    auc = roc_auc_score(y_test, probs)

    print("\n── Final Test Performance ──")
    print(f"ECE : {ece:.5f}")
    print(f"AUC : {auc:.4f}")

    return model


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def run_svm_pipeline(path):
    df = pd.read_csv(path)

    feature_cols = [c for c in df.columns if c.startswith("diff_")]

    print("Using features:", feature_cols)

    # split
    train_df, val_df, test_df = temporal_split_3way(df)

    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # tune
    best_params, scaler = tune_svm(train_df, val_df, feature_cols)

    # final train
    model = train_final(train_df, val_df, test_df, feature_cols, best_params, scaler)

    return model


# ─────────────────────────────────────────────

if __name__ == "__main__":
    model = run_svm_pipeline("matchups1-Feature Subset.csv")
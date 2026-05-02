"""
NASCAR Head-to-Head Prediction: Feature Engineering Pipeline
============================================================

Per-driver feature vector (n_features + 1 wide):

  MOMENTUM (1 scalar)
    Unweighted mean of `rating` over the last 4 prior races.
    If fewer than 4 prior races exist, use however many are available.
    If no prior races at all: fall back to global mean of rating.

  ADJUSTED AVERAGE (n_features scalars)
    For each feature, a two-stage blend:

    Stage 1 — Baseline
      Last 20 prior races, exponential decay applied.
      weight_i = lambda ^ (age_i / 4)
      where age_i = 0 for most recent, 1/4 for one before, etc.
      Shrunk toward global mean:
        baseline_adj = (n_eff / (n_eff + k)) * baseline_wavg
                     + (k     / (n_eff + k)) * global_mean

    Stage 2 — Long-term track type
      All prior races of the same track_type, exponential decay applied.
      Ages measured from driver's most recent overall race for consistency.
      Shrunk toward global track-type mean:
        track_adj = (n_eff / (n_eff + k)) * track_wavg
                  + (k     / (n_eff + k)) * global_track_type_mean
      Fallback: if fewer than min_track_races prior races of this track type,
      use baseline_adj directly.

    Final adj_{feature} = track_adj  (or baseline_adj if track fallback triggered)

Edge cases (all documented):
  No prior races at all      → momentum = global rating mean,
                               adj_{feat} = global feature mean
  Momentum < 4 prior races   → use however many exist, no shrinkage
  Baseline < 20 prior races  → use however many exist, shrinkage still applied
  Track type below threshold → fall back to baseline_adj

Matchup vector = (driver_i − driver_j) across all n_features + 1 values.
Target y = 1 if driver_i finished ahead of driver_j (lower finish position = better).

Hyperparameters (tune via CV on training races):
  lam             : decay base per 4-race unit        (default 0.85)
  k               : shrinkage strength                (default 10.0)
  min_track_races : track-type fallback threshold     (default 3)
  momentum_window : races used for momentum scalar    (default 4)
  baseline_window : races used for baseline layer     (default 20)
"""

import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from itertools import combinations

# ── CONFIGURATION ─────────────────────────────────────────────────────────────

LAMBDA: float           = 0.85
K: float                = 10.0
MIN_TRACK_RACES: int    = 3
MOMENTUM_WINDOW: int    = 4
BASELINE_WINDOW: int    = 20
MIN_RACE_HISTORY: int   = 10

FEATURE_COLS: list[str] = [
    "finish",
    "start",
    "mid",
    "best",
    "worst",
    "avg",
    "green_flag_passing_diff",
    "green_flag_passes",
    "green_flag_times_passed",
    "quality_passes",
    "pct_quality_passes",
    "fastest_lap",
    "top_15_laps",
    "pct_top_15_laps",
    "laps_led",
    "pct_laps_led",
    "rating",
]


# ── STEP 0: LOAD & CLEAN ──────────────────────────────────────────────────────

def load_single(path: str) -> pd.DataFrame:
    """Load and normalise one CSV file."""
    df = pd.read_csv(path)
    # 2022-2024 use 'driver_name'; 2025 has unnamed first col + extra 'driver2'
    if df.columns[0].strip() == "":
        df = df.rename(columns={df.columns[0]: "driver_name"})
    if "driver2" in df.columns:
        df = df.drop(columns=["driver2"])
    df = df.rename(columns={"driver_name": "driver"})
    df["driver"]    = df["driver"].str.strip()
    df["race_date"] = pd.to_datetime(df["race_date"])
    return df


def load_data(paths: list[str]) -> pd.DataFrame:
    """
    Load and concatenate multiple yearly CSVs, sorted chronologically.
    Median-fills NaN values in feature columns across the combined dataset.
    """
    df = pd.concat([load_single(p) for p in paths], ignore_index=True)
    df = df.sort_values("race_date").reset_index(drop=True)
    for col in FEATURE_COLS:
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    return df


# ── STEP 1: ASSIGN RACE IDs ───────────────────────────────────────────────────

def assign_race_ids(df: pd.DataFrame) -> pd.DataFrame:
    """
    Each unique (race_date, track_name) pair → one race_id, numbered chronologically.
    """
    race_keys = (
        df[["race_date", "track_name"]]
        .drop_duplicates()
        .sort_values("race_date")
        .reset_index(drop=True)
    )
    race_keys["race_id"] = np.arange(len(race_keys))
    race_keys["year"] = race_keys["race_date"].dt.year
    return df.merge(race_keys, on=["race_date", "track_name"], how="left")


# ── STEP 2: HELPERS ───────────────────────────────────────────────────────────

def decay_weights(n: int, lam: float) -> np.ndarray:
    """
    Weights for n races ordered oldest → newest.
      most recent race  : age = 0.0        → weight = lam^0.0 = 1.0
      one race prior    : age = 1/4        → weight = lam^0.25
      i races prior     : age = i/4        → weight = lam^(i/4)
    """
    ages = np.arange(n - 1, -1, -1) / 4.0   # shape (n,), oldest first
    return lam ** ages


def weighted_avg_neff(
    values: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, float]:
    """
    Returns (weighted_average, effective_sample_size).
    n_eff = sum(w)^2 / sum(w^2)
    Returns (nan, 0.0) on empty input or zero weight sum.
    """
    w_sum = weights.sum()
    if w_sum == 0 or len(values) == 0:
        return np.nan, 0.0
    wavg  = (weights * values).sum() / w_sum
    n_eff = (w_sum ** 2) / (weights ** 2).sum()
    return wavg, n_eff


def shrink(wavg: float, n_eff: float, global_mean: float, k: float) -> float:
    """
    Shrinks weighted average toward global_mean:
      adjusted = (n_eff / (n_eff + k)) * wavg
               + (k     / (n_eff + k)) * global_mean
    """
    w = n_eff / (n_eff + k)
    return w * wavg + (1 - w) * global_mean


# ── STEP 3: COMPUTE FEATURES ──────────────────────────────────────────────────

def compute_driver_features(
    df: pd.DataFrame,
    feature_cols: list[str],
    lam: float           = LAMBDA,
    k: float             = K,
    min_track_races: int = MIN_TRACK_RACES,
    momentum_window: int = MOMENTUM_WINDOW,
    baseline_window: int = BASELINE_WINDOW,
) -> pd.DataFrame:
    """
    For every (driver, race) computes:
      momentum      — unweighted mean of rating over last momentum_window prior races
      adj_{feature} — shrinkage-adjusted blended average per feature

    Returns DataFrame with columns:
      driver, race_id, finish, momentum, adj_{feat} × n_features
    """
    # Global means — used as last-resort fallback (no prior races at all)
    global_means       = df[feature_cols].mean()
    global_track_means = df.groupby("track_type")[feature_cols].mean()

    df = df.sort_values(["driver", "race_date", "race_id"]).reset_index(drop=True)

    result_rows = []

    for (driver, team), grp in df.groupby(["driver", "team_name"], sort=False):
        grp = grp.sort_values("race_date").reset_index(drop=True)

        for idx in range(len(grp)):
            row        = grp.iloc[idx]
            prior      = grp.iloc[:idx]       # strictly prior races — no leakage
            if len(prior) < MIN_RACE_HISTORY:
                continue
            track_type = row["track_type"]

            record = {
                "driver":  driver,
                "team": team,
                "race_id": row["race_id"],
                "year": row["year"],
                "finish":  row["finish"],
            }

            # ── MOMENTUM ──────────────────────────────────────────────────────
            # Unweighted mean of rating over last momentum_window prior races.
            if len(prior) == 0:
                record["momentum"] = global_means["rating"]
            else:
                recent             = prior.iloc[-momentum_window:]
                record["momentum"] = recent["rating"].mean()

            # ── ADJUSTED AVERAGE ──────────────────────────────────────────────
            if len(prior) == 0:
                # No prior history — fall back to global means for all features
                for feat in feature_cols:
                    record[f"adj_{feat}"] = global_means[feat]

            else:
                # Stage 1: Baseline — last baseline_window races, decayed + shrunk
                baseline_prior   = prior.iloc[-baseline_window:]
                baseline_weights = decay_weights(len(baseline_prior), lam)

                # Stage 2: Track-type — all prior races of same type, decayed
                # Ages are measured from the driver's most recent overall race
                # so recency is on the same scale across both layers.
                track_mask    = (prior["track_type"] == track_type).values
                track_indices = np.where(track_mask)[0]   # positions in prior (oldest→newest)
                n_prior       = len(prior)

                # age of position pos in prior = (n_prior - 1 - pos) / 4
                track_ages    = (n_prior - 1 - track_indices) / 4.0
                track_weights = lam ** track_ages
                track_prior   = prior.iloc[track_indices]
                if len(track_prior) < min_track_races:
                    continue

                g_track_mean = (
                    global_track_means.loc[track_type]
                    if track_type in global_track_means.index
                    else global_means
                )

                for feat in feature_cols:
                    # Baseline: weighted avg of last 20, shrunk toward global mean
                    b_wavg, b_neff = weighted_avg_neff(
                        baseline_prior[feat].values, baseline_weights
                    )
                    baseline_adj = shrink(b_wavg, b_neff, global_means[feat], k)

                    # Track type: weighted avg of all same-type, shrunk toward
                    # global track-type mean. Fall back if too few races.
                    if len(track_prior) < min_track_races:
                        record[f"adj_{feat}"] = baseline_adj
                    else:
                        t_wavg, t_neff = weighted_avg_neff(
                            track_prior[feat].values, track_weights
                        )
                        record[f"adj_{feat}"] = shrink(
                            t_wavg, t_neff, g_track_mean[feat], k
                        )

            result_rows.append(record)

    return pd.DataFrame(result_rows)


# ── STEP 4: MATCHUP GENERATION ────────────────────────────────────────────────

def generate_matchups(
    driver_features: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    """
    For each race, generates all unordered (driver_i, driver_j) pairs with i < j.
    Matchup vector = driver_i values − driver_j values.
    Target y = 1 if driver_i finished ahead (lower finish position) of driver_j.
    """
    value_cols = ["momentum"] + [f"adj_{f}" for f in feature_cols]
    diff_cols  = ["diff_momentum"] + [f"diff_{f}" for f in feature_cols]

    matchup_rows = []

    for (race_id, year), race_grp in driver_features.groupby(["race_id", "year"]):
        drivers    = sorted(race_grp["driver"].tolist())
        feat_map   = race_grp.set_index("driver")[value_cols]
        finish_map = race_grp.set_index("driver")["finish"]

        for d_i, d_j in combinations(drivers, 2):
            if d_i not in feat_map.index or d_j not in feat_map.index:
                continue

            diffs = (feat_map.loc[d_i] - feat_map.loc[d_j]).values
            y     = 1 if finish_map[d_i] < finish_map[d_j] else 0

            matchup_rows.append({
                "race_id":  race_id,
                "driver_i": d_i,
                "driver_j": d_j,
                "y":        y,
                **dict(zip(diff_cols, diffs)),
            })

            matchup_rows.append({
                "race_id":  race_id,
                "year": year,
                "driver_i": d_j,
                "driver_j": d_i,
                "y":        1 - y,
                **dict(zip(diff_cols, -diffs)),
            })

    return pd.DataFrame(matchup_rows)


# ── STEP 5: FULL PIPELINE ─────────────────────────────────────────────────────

def time_based_split(matchups):
    train = matchups[matchups["year"].isin([2022, 2023])].copy()
    val   = matchups[matchups["year"] == 2024].copy()
    test  = matchups[matchups["year"] == 2025].copy()

    return train, val, test

def train_model(X, y):
    model = LogisticRegression(
        penalty="l2",
        solver="lbfgs",
        max_iter=1000,
    )
    model.fit(X, y)
    return model

# Only works for binary classification with doubled data (y=1 and y=0 rows are exact inverses of each other).
def classwise_ece(y_true, y_prob, n_bins=20):
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    non_empty_bins = 0

    for i in range(n_bins):
        lower, upper = bin_boundaries[i], bin_boundaries[i + 1]

        if i == n_bins - 1:
            in_bin = (y_prob >= lower) & (y_prob <= upper)
        else:
            in_bin = (y_prob >= lower) & (y_prob < upper)

        bin_size = np.sum(in_bin)

        print(f'Bin {i + 1}/{n_bins}: [{lower:.2f}, {upper:.2f}) - Size: {bin_size}')

        if bin_size > 0:
            non_empty_bins += 1
            accuracy = np.mean(y_true[in_bin])
            confidence = np.mean(y_prob[in_bin])
            ece += (bin_size / len(y_prob)) * abs(accuracy - confidence)
        
    if non_empty_bins < 0.8 * n_bins:
        print(f'Fewer than 80% of bins contain predictions.')
        return 1.0

    print(f'Classwise ECE: {ece:.4f} (calculated over {non_empty_bins}/{n_bins} non-empty bins)')
    return ece

def sfs(X_train, y_train, X_val, y_val, feature_names, n_bins=20):
    remaining = list(feature_names)
    selected = []

    best_ece = np.inf
    history = []

    while len(remaining) > 0:
        best_feature = None
        best_candidate_ece = np.inf

        for f in remaining:
            candidate_features = selected + [f]

            X_tr = X_train[candidate_features]
            X_vl = X_val[candidate_features]

            model = LogisticRegression(
                penalty="l2",
                solver="lbfgs",
                max_iter=1000,
            )
            model.fit(X_tr, y_train)

            probabilities = model.predict_proba(X_vl)[: ,1]
            ece = classwise_ece(y_val, probabilities, n_bins=n_bins)

            if ece < best_candidate_ece:
                best_candidate_ece = ece
                best_feature = f
        
        if best_candidate_ece >= best_ece:
            print(f'No improvement in ECE. Stopping selection.')
            break

        selected.append(best_feature)
        remaining.remove(best_feature)
        best_ece = best_candidate_ece

        history.append((list(selected), best_ece))

        print(f'Added {best_feature}, ECE: {best_ece:.5f}')

    return selected, history

def run_pipeline(
    paths: list[str],
    feature_cols: list[str] = FEATURE_COLS,
    lam: float              = LAMBDA,
    k: float                = K,
    min_track_races: int    = MIN_TRACK_RACES,
    momentum_window: int    = MOMENTUM_WINDOW,
    baseline_window: int    = BASELINE_WINDOW,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    End-to-end pipeline. Returns:
      driver_features : one row per (driver, race)
                        columns: driver, race_id, finish, momentum, adj_{feat}...
      matchups        : one row per (race, driver_i, driver_j)
                        columns: race_id, driver_i, driver_j, y,
                                 diff_momentum, diff_{feat}...
    """
    print("Loading data...")
    df = load_data(paths)

    print("Assigning race IDs...")
    df = assign_race_ids(df)

    available = [c for c in feature_cols if c in df.columns]
    missing   = set(feature_cols) - set(available)
    if missing:
        print(f"  Warning: columns not found, skipping: {missing}")

    print(
        f"Computing features for {df['driver'].nunique()} drivers "
        f"across {df['race_id'].nunique()} races..."
    )
    driver_features = compute_driver_features(
        df, available,
        lam=lam, k=k,
        min_track_races=min_track_races,
        momentum_window=momentum_window,
        baseline_window=baseline_window,
    )

    print("Generating matchups...")
    matchups = generate_matchups(driver_features, available)

    n_feat = len(available)
    print(f"\nDone.")
    print(f"  Feature vector width  : {n_feat} adj features + 1 momentum = {n_feat + 1}")
    print(f"  Driver features shape : {driver_features.shape}")
    print(f"  Matchups shape        : {matchups.shape}")
    print(f"  Label balance         : {matchups['y'].mean():.3f}  (ideal = 0.500)")

    train, val, test = time_based_split(matchups)

    calculated_feature_cols = [c for c in matchups.columns if c.startswith("diff_")]

    X_train, y_train = train[calculated_feature_cols], train["y"]
    X_val,   y_val   = val[calculated_feature_cols], val["y"]
    X_test,  y_test  = test[calculated_feature_cols], test["y"]

    # Use Spearman correlation to identify and drop highly correlated features (> 0.7).
    X_train_copy = X_train.copy()
    corr_matrix = X_train.corr(method="spearman").abs()
    target_corr = X_train_copy.apply(lambda col: col.corr(y_train, method="spearman")).abs()

    to_drop = set()
    columns = corr_matrix.columns

    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            f1, f2 = columns[i], columns[j]

            if corr_matrix.iloc[i, j] > 0.7:
                if target_corr[f1] < target_corr[f2]:
                    to_drop.add(f2)
                else:
                    to_drop.add(f1)

    print(f'Dropping features: {", ".join(list(to_drop))} due to high correlation (> 0.7)')
    X_train = X_train.drop(columns=list(to_drop))
    X_val = val[X_train.columns].copy()
    X_test = test[X_train.columns].copy()
    print("Features after filtering:", X_train.shape[1])

    # Use Kolmogorov-Smirnov test to identify and drop features with significantly different distributions between train and val sets (p < 0.01).
    keep_columns = []

    for column in X_train.columns:
        stat, p_value = ks_2samp(X_train[column], X_val[column])

        if p_value >= 0.01:
            keep_columns.append(column)
        else:
            print(f"Dropping {column} (p={p_value:.5f})")
    
    X_train = X_train[keep_columns]
    X_val = X_val[keep_columns]
    X_test = X_test[keep_columns]

    print("Features after KS test filtering:", X_train.shape[1])

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_val = pd.DataFrame(X_val_scaled, columns=X_val.columns, index=X_val.index)
    X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

    lr_model = train_model(X_train, y_train)

    probabilities = lr_model.predict_proba(X_val)[: ,1]
    ece = classwise_ece(y_val, probabilities)

    feature_subset, history = sfs(X_train, y_train, X_val, y_val, X_train.columns, n_bins=20)

    print(f"\nSelected Feature Subset: {", ".join(feature_subset)}")

    return driver_features, matchups


# ── ENTRYPOINT ────────────────────────────────────────────────────────────────

DATA_PATHS = [
    "2022 Loop Data.csv",
    "2023 Loop Data.csv",
    "2024 Loop Data.csv",
    "2025 Loop Data.csv",
]

if __name__ == "__main__":
    driver_features, matchups = run_pipeline(DATA_PATHS)

    driver_features.to_csv("./driver_features1.csv", index=False)
    matchups.to_csv("./matchups1.csv", index=False)
    print("Outputs saved to driver_features1.csv and matchups1.csv")
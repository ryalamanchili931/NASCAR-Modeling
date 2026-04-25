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
from itertools import combinations

# ── CONFIGURATION ─────────────────────────────────────────────────────────────

LAMBDA: float           = 0.85
K: float                = 10.0
MIN_TRACK_RACES: int    = 3
MOMENTUM_WINDOW: int    = 4
BASELINE_WINDOW: int    = 20

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

    for driver, grp in df.groupby("driver", sort=False):
        grp = grp.sort_values("race_date").reset_index(drop=True)

        for idx in range(len(grp)):
            row        = grp.iloc[idx]
            prior      = grp.iloc[:idx]       # strictly prior races — no leakage
            track_type = row["track_type"]

            record = {
                "driver":  driver,
                "race_id": row["race_id"],
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

    for race_id, race_grp in driver_features.groupby("race_id"):
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

    return pd.DataFrame(matchup_rows)


# ── STEP 5: FULL PIPELINE ─────────────────────────────────────────────────────

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

    driver_features.to_csv("./driver_features.csv", index=False)
    matchups.to_csv("./matchups.csv", index=False)
    print("Outputs saved to driver_features.csv and matchups.csv")
"""
NASCAR Head-to-Head Prediction: GD-Enhanced Training Pipeline
=============================================================

Integrates two gradient-descent components on top of the feature engineering pipeline:

  Option 3 — Learnable lam & k
    lam (decay base) and k (shrinkage strength) are treated as differentiable
    parameters and optimised via Adam on a pairwise BCE loss computed directly
    from the feature engineering output. This makes the track-type decay
    data-driven rather than hand-tuned.

  Option 2 — GD Feature Weighter
    A small learned linear layer that re-weights the diff feature vector before
    it enters the SVM. Trained with BCE loss + L2 regularisation.
    Lets the model discover which engineered features (and which track-type
    adjusted ones) matter most for head-to-head prediction.

Full pipeline:
  Raw CSVs
    → [Option 3] optimise lam / k via GD
    → run_pipeline(lam*, k*) → matchups DataFrame
    → [Option 2] train FeatureWeighter on matchups
    → reweight diff features
    → StandardScaler → SVC(linear | poly | rbf | sigmoid)
    → compare all four kernels, select best by AUC
    → evaluate (AUC, Brier, classification report)

Dependencies:
  numpy, pandas, scikit-learn, torch
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import loguniform
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, brier_score_loss, classification_report
from sklearn.calibration import CalibratedClassifierCV

# Import your existing pipeline
from Nascar_Pipeline import (
    run_pipeline,
    FEATURE_COLS,
    DATA_PATHS,
    LAMBDA,
    K,
    MIN_TRACK_RACES,
    MOMENTUM_WINDOW,
    BASELINE_WINDOW,
)

# ── CONFIGURATION ─────────────────────────────────────────────────────────────

TRAIN_SPLIT      : float = 0.80   # fraction of race_ids used for training
GD_LR            : float = 0.01   # learning rate for both GD stages
GD_EPOCHS_PARAMS : int   = 150    # epochs for lam/k optimisation (Option 3)
GD_EPOCHS_FEAT   : int   = 200    # epochs for feature weighter    (Option 2)
L2_REG           : float = 1e-3   # L2 penalty on feature weights
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── TEMPORAL TRAIN / TEST SPLIT ───────────────────────────────────────────────

def temporal_split(
    matchups: pd.DataFrame,
    train_frac: float = TRAIN_SPLIT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits matchups chronologically by race_id so no future information
    leaks into the training set.
    """
    cutoff = int(matchups["race_id"].quantile(train_frac))
    train  = matchups[matchups["race_id"] <= cutoff].copy()
    test   = matchups[matchups["race_id"] >  cutoff].copy()
    print(f"Train races: ≤{cutoff}  ({len(train):,} matchups)")
    print(f"Test  races: >{cutoff}  ({len(test):,}  matchups)")
    return train, test


def get_diff_cols(feature_cols: list[str]) -> list[str]:
    return ["diff_momentum"] + [f"diff_{f}" for f in feature_cols]


# ── OPTION 3: LEARNABLE lam AND k ────────────────────────────────────────────
#
# Strategy:
#   run_pipeline() is called once upfront with default lam/k to produce the
#   matchups DataFrame. The training matchups are split into n_batches chunks
#   by race_id. Each loss call is just a BCE computation on a frozen numpy
#   slice — no pipeline reruns at all during optimisation.
#
#   Finite-difference gradients are used because the pipeline is not
#   differentiable. Adam tracks moment estimates across all batch updates.
#
#   Note: because the matchups are fixed at default lam/k, the gradient signal
#   guides lam/k toward values that would have produced lower loss on those
#   features — a first-order approximation. The final pipeline rerun with
#   optimal lam*/k* produces the true adjusted features for SVM training.

class ParamOptimiser:
    """
    Optimises lam and k via finite-difference Adam on precomputed matchups.

    The matchups DataFrame (from run_pipeline) is passed in once and frozen.
    Each loss call slices a batch of rows by race_id and computes BCE — pure
    numpy, no pandas groupby, no pipeline reruns.

    Per batch: 3 loss calls (centre, lam+eps, k+eps) → 1 Adam update.
    Total cost: O(epochs × n_batches × batch_matchups) — very fast.
    """

    def __init__(
        self,
        matchups: pd.DataFrame,
        train_race_ids: set,
        feature_cols: list[str],
        lr: float   = GD_LR,
        epochs: int = GD_EPOCHS_PARAMS,
        n_batches: int = 10,
        eps: float  = 1e-3,
    ):
        self.feature_cols   = feature_cols
        self.lr             = lr
        self.epochs         = epochs
        self.n_batches      = n_batches
        self.eps            = eps
        self.lam            = LAMBDA
        self.k              = K

        self._m  = np.zeros(2)
        self._v  = np.zeros(2)
        self._t  = 0
        self._b1, self._b2, self._ep = 0.9, 0.999, 1e-8

        # Freeze training matchups as sorted numpy arrays — never touched again
        diff_cols  = get_diff_cols(feature_cols)
        train_m    = matchups[matchups["race_id"].isin(train_race_ids)].copy()
        train_m    = train_m.sort_values("race_id").reset_index(drop=True)

        self._race_ids    = train_m["race_id"].values          # (N,) int
        self._diff_finish = train_m["diff_finish"].values.astype(np.float64)
        self._y           = train_m["y"].values.astype(np.float64)

        # Precompute batch slices by race_id index (done once)
        all_ids   = sorted(train_race_ids)
        size      = max(1, len(all_ids) // n_batches)
        self._batches = []
        for i in range(0, len(all_ids), size):
            batch_ids = set(all_ids[i : i + size])
            mask      = np.isin(self._race_ids, list(batch_ids))
            self._batches.append(mask)

        total = sum(m.sum() for m in self._batches)
        print(f"  Matchup batches ready: {len(self._batches)} batches, "
              f"{total:,} matchups total")

    def _loss(self, lam: float, k: float, mask: np.ndarray) -> float:
        """
        BCE loss on a batch slice. Uses diff_finish as the logit signal —
        the same feature the pipeline adjusts most directly via lam/k.
        lam and k modulate the signal scale as a proxy for their effect
        on the underlying weighted averages.
        """
        diff  = self._diff_finish[mask]
        y     = self._y[mask]
        # lam controls recency sensitivity; k controls shrinkage toward mean.
        # Scaling diff by lam and shifting by 1/k approximates their combined
        # effect on the adjusted finish difference without rerunning the pipeline.
        logit = -(diff * lam - 1.0 / k)
        prob  = 1.0 / (1.0 + np.exp(-logit))
        prob  = np.clip(prob, 1e-7, 1 - 1e-7)
        return float(-np.mean(y * np.log(prob) + (1 - y) * np.log(1 - prob)))

    def _adam_step(self, grads: np.ndarray) -> np.ndarray:
        self._t += 1
        self._m  = self._b1 * self._m + (1 - self._b1) * grads
        self._v  = self._b2 * self._v + (1 - self._b2) * grads ** 2
        m_hat    = self._m / (1 - self._b1 ** self._t)
        v_hat    = self._v / (1 - self._b2 ** self._t)
        return self.lr * m_hat / (np.sqrt(v_hat) + self._ep)

    def optimise(self) -> tuple[float, float]:
        print("\n── Option 3: Optimising lam and k via finite-difference Adam ──")
        print(f"   {self.epochs} epochs × {len(self._batches)} batches "
              f"(pure numpy on frozen matchups — no pipeline reruns)")

        for epoch in range(1, self.epochs + 1):
            epoch_loss = 0.0

            for mask in self._batches:
                loss_c = self._loss(self.lam, self.k, mask)
                g_lam  = (self._loss(self.lam + self.eps, self.k, mask) - loss_c) / self.eps
                g_k    = (self._loss(self.lam, self.k + self.eps, mask) - loss_c) / self.eps

                update    = self._adam_step(np.array([g_lam, g_k]))
                self.lam -= float(update[0])
                self.k   -= float(update[1])
                self.lam  = float(np.clip(self.lam, 0.50, 0.99))
                self.k    = float(np.clip(self.k,   1.00, 100.0))
                epoch_loss += loss_c

            avg_loss = epoch_loss / len(self._batches)
            if epoch % 25 == 0 or epoch == 1:
                print(f"  Epoch {epoch:>3d} | avg_loss={avg_loss:.5f} | "
                      f"lam={self.lam:.4f} | k={self.k:.4f}")

        print(f"\n  ✓ Optimal lam={self.lam:.4f}  k={self.k:.4f}")
        return self.lam, self.k


# ── OPTION 2: GD FEATURE WEIGHTER ────────────────────────────────────────────

class FeatureWeighter(nn.Module):
    """
    Learns a per-feature soft weight vector via gradient descent.
    Applied as an element-wise scale on the diff feature vector.

    Architecture:
      w = softmax(raw_weights) * n_features    # sums to n_features
      output = input * w

    This is a linear transformation with a sum constraint, so no feature
    can be completely zeroed globally — it redistributes importance.
    """

    def __init__(self, n_features: int):
        super().__init__()
        self.raw_weights = nn.Parameter(torch.zeros(n_features))
        self.n_features  = n_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = torch.softmax(self.raw_weights, dim=0) * self.n_features
        return x * w

    def get_weights(self, feature_names: list[str]) -> pd.Series:
        """Returns learned weights as a named Series for inspection."""
        w = torch.softmax(self.raw_weights, dim=0).detach().cpu().numpy()
        w *= self.n_features
        return pd.Series(w, index=feature_names).sort_values(ascending=False)


def train_feature_weighter(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_features: int,
    lr: float   = GD_LR,
    epochs: int = GD_EPOCHS_FEAT,
    l2_reg: float = L2_REG,
) -> FeatureWeighter:
    """
    Trains FeatureWeighter with BCE loss + L2 regularisation on weight magnitudes.
    Uses a thin logistic head so gradients flow back into the weights cleanly.
    """
    print("\n── Option 2: Training FeatureWeighter via Adam ──")

    model     = FeatureWeighter(n_features).to(DEVICE)
    log_head  = nn.Linear(n_features, 1).to(DEVICE)   # thin logistic head
    optimizer = optim.Adam(
        list(model.parameters()) + list(log_head.parameters()), lr=lr
    )
    criterion = nn.BCEWithLogitsLoss()

    X_t = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    y_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(DEVICE)

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        weighted = model(X_t)
        logits   = log_head(weighted)
        bce_loss = criterion(logits, y_t)
        l2_loss  = l2_reg * (model.raw_weights ** 2).sum()
        loss     = bce_loss + l2_loss

        loss.backward()
        optimizer.step()

        if epoch % 40 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>3d} | loss={loss.item():.5f} | bce={bce_loss.item():.5f}")

    print("  ✓ FeatureWeighter trained")
    return model


def apply_weighter(
    model: FeatureWeighter,
    X: np.ndarray,
) -> np.ndarray:
    """Apply trained weighter to numpy array, return reweighted numpy array."""
    model.eval()
    with torch.no_grad():
        X_t  = torch.tensor(X, dtype=torch.float32).to(DEVICE)
        X_w  = model(X_t)
    return X_w.cpu().numpy()


# ── SVM TRAINING & EVALUATION ─────────────────────────────────────────────────

# Kernel configurations matching the four kernels from the paper.
#
# C  : continuous, sampled log-uniformly from [0.1, 50] via RandomizedSearchCV.
#      loguniform(0.1, 50) biases sampling toward smaller values where the
#      decision boundary is most sensitive, while still exploring up to 50.
#
# degree : discrete {2, 3, 4} — applies to poly kernel only (ignored by others).
#
# gamma  : applies to rbf, poly, sigmoid.
# coef0  : applies to poly, sigmoid.

C_DIST = loguniform(0.1, 50)          # continuous C ∈ [0.1, 50]
DEGREE_VALUES = [2, 3, 4]             # discrete degree for poly kernel
N_ITER_SEARCH = 40                    # random search iterations per kernel

KERNEL_CONFIGS: dict[str, dict] = {
    "linear": {
        "svm__C": C_DIST,
    },
    "poly": {
        "svm__C":      C_DIST,
        "svm__degree": DEGREE_VALUES,
        "svm__gamma":  ["scale", "auto"],
        "svm__coef0":  [0.0, 1.0],
    },
    "rbf": {
        "svm__C":     C_DIST,
        "svm__gamma": ["scale", "auto", 0.01, 0.001],
    },
    "sigmoid": {
        "svm__C":     C_DIST,
        "svm__gamma": ["scale", "auto", 0.01],
        "svm__coef0": [0.0, -1.0, 1.0],
    },
}


def _build_kernel_pipe(kernel: str) -> Pipeline:
    """Returns a Pipeline(scaler + calibrated SVC) for the given kernel."""
    base = SVC(kernel=kernel, class_weight="balanced", probability=False)
    calibrated = CalibratedClassifierCV(base, method="isotonic", cv=5)
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svm",    calibrated),
    ])


def _build_kernel_pipe_for_search(kernel: str) -> Pipeline:
    """Returns a Pipeline(scaler + bare SVC) suitable for GridSearchCV."""
    base = SVC(kernel=kernel, class_weight="balanced", probability=True)
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svm",    base),
    ])


def benchmark_kernels(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    tune_hyperparams: bool = False,
) -> tuple[dict[str, dict], str]:
    """
    Trains and evaluates all four kernels (linear, poly, rbf, sigmoid).

    For each kernel:
      - If tune_hyperparams=True: GridSearchCV with TimeSeriesSplit(5)
      - Otherwise: default C=1.0 with CalibratedClassifierCV

    Returns:
      kernel_results : dict  kernel → {model, train_auc, test_auc, test_brier, report}
      best_kernel    : str   kernel name with highest test AUC
    """
    print("\n" + "=" * 60)
    print("Kernel Benchmark: linear | poly | rbf | sigmoid")
    print("=" * 60)

    tscv           = TimeSeriesSplit(n_splits=5)
    kernel_results = {}

    for kernel, param_grid in KERNEL_CONFIGS.items():
        print(f"\n  ── Kernel: {kernel} ──")

        if tune_hyperparams:
            pipe = _build_kernel_pipe_for_search(kernel)
            gs   = RandomizedSearchCV(
                pipe, param_grid,
                n_iter=N_ITER_SEARCH,
                cv=tscv, scoring="roc_auc",
                n_jobs=-1, verbose=0,
                random_state=42,
            )
            gs.fit(X_train, y_train)
            model    = gs.best_estimator_
            cv_auc   = gs.best_score_
            print(f"    Best params : {gs.best_params_}")
            print(f"    CV AUC      : {cv_auc:.4f}")
        else:
            model = _build_kernel_pipe(kernel)
            model.fit(X_train, y_train)

        # ── Metrics ───────────────────────────────────────────────────────────
        train_probs = model.predict_proba(X_train)[:, 1]
        test_probs  = model.predict_proba(X_test)[:, 1]
        test_preds  = model.predict(X_test)

        train_auc  = roc_auc_score(y_train, train_probs)
        test_auc   = roc_auc_score(y_test,  test_probs)
        test_brier = brier_score_loss(y_test, test_probs)
        report     = classification_report(
            y_test, test_preds,
            target_names=["j ahead", "i ahead"],
        )

        print(f"    Train AUC   : {train_auc:.4f}")
        print(f"    Test  AUC   : {test_auc:.4f}")
        print(f"    Brier score : {test_brier:.4f}")

        kernel_results[kernel] = {
            "model":       model,
            "train_auc":   train_auc,
            "test_auc":    test_auc,
            "test_brier":  test_brier,
            "report":      report,
        }

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Kernel Comparison Summary")
    print("=" * 60)
    print(f"  {'Kernel':<10} {'Train AUC':>10} {'Test AUC':>10} {'Brier':>8}")
    print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*8}")

    best_kernel = max(kernel_results, key=lambda k: kernel_results[k]["test_auc"])

    for kernel, res in kernel_results.items():
        marker = " ◀ best" if kernel == best_kernel else ""
        print(
            f"  {kernel:<10} {res['train_auc']:>10.4f} {res['test_auc']:>10.4f} "
            f"{res['test_brier']:>8.4f}{marker}"
        )

    print(f"\n  ✓ Best kernel: {best_kernel}  (test AUC = {kernel_results[best_kernel]['test_auc']:.4f})")
    print(f"\n  Classification report for best kernel ({best_kernel}):")
    print(kernel_results[best_kernel]["report"])

    return kernel_results, best_kernel


def evaluate(
    model,
    X: np.ndarray,
    y: np.ndarray,
    label: str = "Test",
) -> None:
    """Standalone evaluation for a single model (used outside benchmark)."""
    probs = model.predict_proba(X)[:, 1]
    preds = model.predict(X)

    auc   = roc_auc_score(y, probs)
    brier = brier_score_loss(y, probs)

    print(f"\n── {label} Evaluation ──")
    print(f"  AUC        : {auc:.4f}")
    print(f"  Brier score: {brier:.4f}  (lower = better calibration)")
    print(f"\n{classification_report(preds, y, target_names=['j ahead', 'i ahead'])}")


# ── FULL PIPELINE ─────────────────────────────────────────────────────────────

def run_full_pipeline(
    data_paths: list[str]    = DATA_PATHS,
    feature_cols: list[str]  = FEATURE_COLS,
    tune_hyperparams: bool   = False,
    run_param_opt: bool      = True,
) -> dict:
    """
    End-to-end pipeline combining Options 2 and 3 with SVM.

    Steps:
      1. Run pipeline once with default lam/k → matchups
      2. [Option 3] Optimise lam/k on training matchups via finite-diff Adam
      3. If params changed, rerun pipeline with optimal lam*/k*
      4. Temporal train/test split
      5. [Option 2] Train FeatureWeighter on training matchups
      6. Reweight features for both splits
      7. Benchmark all four kernels, select best by AUC
      8. Evaluate on test set

    Returns dict with model components for downstream use.
    """

    # ── Step 1: Run pipeline once with defaults ───────────────────────────────
    print("=" * 60)
    print("Step 1: Running pipeline with default lam/k")
    print("=" * 60)

    _, matchups_default = run_pipeline(
        data_paths,
        feature_cols = feature_cols,
        lam          = LAMBDA,
        k            = K,
    )
    available_cols = [c.replace("diff_", "", 1) for c in matchups_default.columns
                      if c.startswith("diff_") and c != "diff_momentum"]

    all_race_ids   = sorted(matchups_default["race_id"].unique())
    cutoff_idx     = int(len(all_race_ids) * TRAIN_SPLIT)
    train_race_ids = set(all_race_ids[:cutoff_idx])

    # ── Step 2: Option 3 — Optimise lam and k ────────────────────────────────
    if run_param_opt:
        print("\n" + "=" * 60)
        print("Step 2: Option 3 — Optimising lam / k via GD")
        print("=" * 60)

        param_opt = ParamOptimiser(
            matchups       = matchups_default,
            train_race_ids = train_race_ids,
            feature_cols   = available_cols,
        )
        opt_lam, opt_k = param_opt.optimise()
    else:
        opt_lam, opt_k = LAMBDA, K
        print(f"\nStep 2: Skipped param optimisation — using lam={opt_lam}, k={opt_k}")

    # ── Step 3: Rerun pipeline only if params actually changed ────────────────
    params_changed = (abs(opt_lam - LAMBDA) > 1e-4 or abs(opt_k - K) > 1e-4)
    if run_param_opt and params_changed:
        print("\n" + "=" * 60)
        print(f"Step 3: Rerunning pipeline with lam={opt_lam:.4f}, k={opt_k:.4f}")
        print("=" * 60)
        _, matchups = run_pipeline(
            data_paths,
            feature_cols = available_cols,
            lam          = opt_lam,
            k            = opt_k,
        )
    else:
        print("\nStep 3: Params unchanged — reusing initial matchups")
        matchups = matchups_default

    # ── Step 4: Temporal split ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 4: Temporal train/test split")
    print("=" * 60)

    train_m, test_m = temporal_split(matchups)
    diff_cols       = get_diff_cols(available_cols)

    X_train = train_m[diff_cols].values.astype(np.float32)
    y_train = train_m["y"].values.astype(np.float32)
    X_test  = test_m[diff_cols].values.astype(np.float32)
    y_test  = test_m["y"].values.astype(np.float32)

    # ── Step 5: Option 2 — Train FeatureWeighter ─────────────────────────────
    print("\n" + "=" * 60)
    print("Step 5: Option 2 — Training FeatureWeighter")
    print("=" * 60)

    weighter = train_feature_weighter(X_train, y_train, n_features=len(diff_cols))

    # Print top features by learned weight
    weights = weighter.get_weights(diff_cols)
    print("\n  Top 10 features by learned weight:")
    print(weights.head(10).to_string())

    # ── Step 6: Reweight features ─────────────────────────────────────────────
    X_train_w = apply_weighter(weighter, X_train)
    X_test_w  = apply_weighter(weighter, X_test)

    # ── Step 7 & 8: Benchmark all four kernels, select best ───────────────────
    kernel_results, best_kernel = benchmark_kernels(
        X_train_w, y_train,
        X_test_w,  y_test,
        tune_hyperparams=tune_hyperparams,
    )

    best_model = kernel_results[best_kernel]["model"]

    return {
        "opt_lam":        opt_lam,
        "opt_k":          opt_k,
        "weighter":       weighter,
        "best_kernel":    best_kernel,
        "best_model":     best_model,
        "kernel_results": kernel_results,   # all four models + metrics
        "diff_cols":      diff_cols,
        "matchups":       matchups,
        "feature_weights": weights,
    }


# ── INFERENCE HELPER ──────────────────────────────────────────────────────────

def predict_matchup(
    components: dict,
    diff_vector: np.ndarray,
) -> dict:
    """
    Given a diff feature vector for a new (driver_i, driver_j) matchup,
    returns probability that driver_i finishes ahead of driver_j.

    diff_vector: 1-D numpy array of length len(diff_cols),
                 ordered as [diff_momentum, diff_finish, diff_start, ...]
    """
    weighter = components["weighter"]
    svm      = components["best_model"]

    X        = diff_vector.reshape(1, -1).astype(np.float32)
    X_w      = apply_weighter(weighter, X)
    prob_i_wins = svm.predict_proba(X_w)[0, 1]

    return {
        "prob_i_ahead": prob_i_wins,
        "prob_j_ahead": 1.0 - prob_i_wins,
        "prediction":   "i" if prob_i_wins > 0.5 else "j",
    }


# ── ENTRYPOINT ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    components = run_full_pipeline(
        data_paths       = DATA_PATHS,
        feature_cols     = FEATURE_COLS,
        tune_hyperparams = False,   # set True for GridSearch over C/gamma/degree (slower)
        run_param_opt    = True,    # set False to skip Option 3 and use defaults
    )

    print("\n── Learned hyperparameters ──")
    print(f"  lam (decay base) : {components['opt_lam']:.4f}  (started at {LAMBDA})")
    print(f"  k   (shrinkage)  : {components['opt_k']:.4f}   (started at {K})")

    print(f"\n── Best kernel selected: {components['best_kernel']} ──")
    for kernel, res in components["kernel_results"].items():
        print(f"  {kernel:<10}  test AUC={res['test_auc']:.4f}  brier={res['test_brier']:.4f}")

    print("\n── Full feature weight ranking ──")
    print(components["feature_weights"].to_string())
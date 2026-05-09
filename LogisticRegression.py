import pandas as pd
from Nascar_Pipeline import classwise_ece
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Load data
data = pd.read_csv("matchups1-Feature Subset.csv")

feature_cols = [c for c in data.columns if c.startswith("diff_")] # highly engineered
#feature_cols = [c for c in data.columns if c in ["diff_worst_track", "diff_green_flag_passing_diff_baseline", "diff_finish_track", "diff_pct_laps_led_track", "diff_green_flag_passing_diff_track"]] # basic
#feature_cols = [c for c in data.columns if c in ["diff_green_flag_passing_diff_ytd", "diff_pct_laps_led_ytd", "diff_green_flag_passes_ytd"] # paper

#Training and Testing
train_mask = data["year"].isin([2022, 2023, 2024])
test_mask  = data["year"] == 2025

X_train = data.loc[train_mask, feature_cols]
y_train = data.loc[train_mask, "y"]

X_test  = data.loc[test_mask, feature_cols]
y_test  = data.loc[test_mask, "y"]

print("Train set size:", len(X_train))
print("Test set size:", len(X_test))

#Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

#Model
model = LogisticRegression(
    penalty="l2",
    solver="lbfgs",
    max_iter=1000,
)

model.fit(X_train_scaled, y_train)

#Probabilities
y_prob = model.predict_proba(X_test_scaled)[:, 1]
print("Sum of Probabilities:", y_prob.sum())
print ("Actual Positives in Test Set:", y_test.sum())

ece = classwise_ece(y_test, y_prob, min_non_empty_bins=0.80)
print("ECE:", ece)

#Test on Non-Training Data
race_id = 134

driver_i = "Tyler Reddick"
driver_j = "Kyle Larson"

test_mask = (
    (data["race_id"] == race_id) &
    (data["driver_i"] == driver_i) &
    (data["driver_j"] == driver_j)
)

X_test  = data.loc[test_mask, feature_cols]
y_test  = data.loc[test_mask, "y"]
X_test_scaled  = scaler.transform(X_test)
y_prob = model.predict_proba(X_test_scaled)[:, 1]
print(f"Predicted probability of {driver_i} beating {driver_j} in race {race_id}: {y_prob[0]:.5f}")
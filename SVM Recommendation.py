import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# Example: Replace with your actual data
# X shape: (100000, 22)
# y shape: (100000,)
X = np.random.randn(100000, 22)
y = np.random.randint(0, 2, size=100000)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Pipeline: scaling + SVM
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC())
])

# Hyperparameter search space
param_dist = {
    "svm__C": np.linspace(0.1, 50, 100),  # continuous range sampled
    "svm__kernel": ["linear", "poly", "rbf", "sigmoid"],
    "svm__degree": [2, 3, 4]  # only used for 'poly'
}

# Randomized search (faster than grid for large data)
search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_dist,
    n_iter=25,              # increase for better tuning
    cv=3,                   # 3-fold CV
    verbose=2,
    n_jobs=-1,              # use all CPU cores
    random_state=42
)

# Train
search.fit(X_train, y_train)

# Best model
best_model = search.best_estimator_
print("Best params:", search.best_params_)

# Evaluate
y_pred = best_model.predict(X_test)
print("Test accuracy:", accuracy_score(y_test, y_pred))
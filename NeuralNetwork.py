import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import StandardScaler
from Nascar_Pipeline import classwise_ece

class HybridLoss(nn.Module):
    def __init__(self, alpha=0.7):
        super().__init__()
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)

        probs = torch.sigmoid(logits)

        brier_loss = torch.mean((probs - targets) ** 2)

        return self.alpha * bce_loss + (1 - self.alpha) * brier_loss

data = pd.read_csv("matchups1-Feature Subset.csv")
feature_cols = [c for c in data.columns if c.startswith("diff_")]

train_mask = data["year"].isin([2022, 2023, 2024])
test_mask  = data["year"] == 2025

X_train = data.loc[train_mask, feature_cols]
y_train = data.loc[train_mask, "y"]

X_test  = data.loc[test_mask, feature_cols]
y_test  = data.loc[test_mask, "y"]

X_train = X_train.dropna()
y_train = y_train.loc[X_train.index]

X_test = X_test.dropna()
y_test = y_test.loc[X_test.index]

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)

X_test_t  = torch.tensor(X_test, dtype=torch.float32)
y_test_t  = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        return self.net(x)

model = SimpleNN(input_dim=X_train.shape[1])

criterion = HybridLoss(alpha=0.8)
optimizer = optim.Adam(model.parameters(), lr=0.001)

EPOCHS = 50

for epoch in range(EPOCHS):
    model.train()

    logits = model(X_train_t)
    loss = criterion(logits, y_train_t)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item():.5f}")

model.eval()
with torch.no_grad():
    logits = model(X_test_t)
    probs  = torch.sigmoid(logits).numpy().flatten()

ece = classwise_ece(y_test.values, probs)

print("\n── Neural Network Results ──")
print("ECE:", ece)
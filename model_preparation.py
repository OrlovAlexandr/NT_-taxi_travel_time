import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

# Import data
X_train_scaled_k = np.loadtxt("artifacts/X_train_scaled_k.csv", delimiter=",")
X_valid_scaled_k = np.loadtxt("artifacts/X_valid_scaled_k.csv", delimiter=",")
y_train_log = pd.read_csv("artifacts/y_train_log.csv")
y_valid_log = pd.read_csv("artifacts/y_valid_log.csv")

# Create the model
gb = GradientBoostingRegressor(
    learning_rate=0.5,
    n_estimators=100,
    max_depth=6,
    min_samples_split=30,
    random_state=42,
    verbose=True
)

# Train the model
gb.fit(X_train_scaled_k, y_train_log)

# Save the model to a file
with open("gb_k_model.pkl", "wb") as f:
    pickle.dump(gb, f)

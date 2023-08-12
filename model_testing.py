import pickle

import numpy as np
import pandas as pd
from sklearn import metrics

# Import data
X_train_scaled_k = np.loadtxt("artifacts/X_train_scaled_k.csv", delimiter=",")
X_valid_scaled_k = np.loadtxt("artifacts/X_valid_scaled_k.csv", delimiter=",")
y_train_log = pd.read_csv("artifacts/y_train_log.csv")
y_valid_log = pd.read_csv("artifacts/y_valid_log.csv")

# Load the trained model from the file
with open("artifacts/gb_k_model.pkl", "rb") as f:
    gb = pickle.load(f)

# Predict
y_train_pred_k = gb.predict(X_train_scaled_k)
y_valid_pred_k = gb.predict(X_valid_scaled_k)

# Display metrics
print('RMSLE on the training set:',
      round(np.sqrt(metrics.mean_squared_error(y_train_log, y_train_pred_k)),
            2))
print('RMSLE on the validation set:',
      round(np.sqrt(metrics.mean_squared_error(y_valid_log, y_valid_pred_k)),
            2))

print('-' * 79 + '\n')
print('Model testing is finished!')

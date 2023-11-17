import os
import pickle
import sys
import json

import numpy as np
import yaml
from sklearn import metrics

# Load hyperparameters for training from the YAML configuration file
params = yaml.safe_load(open("params.yaml"))["train"]

# Check if the correct number of command-line arguments is provided
if len(sys.argv) != 4:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython3 evaluate.py data-file\n")
    sys.exit(1)

# Set the output path for the evaluation scores
f_output = os.path.join("evaluate", "score_gbr.json")
os.makedirs(os.path.join("evaluate"), exist_ok=True)

# Set the input path
train_path = sys.argv[1]
test_path = sys.argv[2]
print('train path:', train_path)
print('test path:', test_path)

# Import the training and testing data from the specified files
with open(train_path, 'rb') as f:
    train_set = np.load(f)
    X_train = train_set['X_train']
    y_train = train_set['y_train']
    
with open(test_path, 'rb') as f:
    test_set = np.load(f)
    X_test = test_set['X_test']
    y_test = test_set['y_test']


# Load the trained model from the specified file
with open(sys.argv[3], "rb") as f:
    model = pickle.load(f)

# Predict using the trained model
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Calculate evaluation metrics
score = round(model.score(X_test, y_test), 4)
train_rmsle = round(
    np.sqrt(
        metrics.mean_squared_error(y_train, y_train_pred)
    ),
    4)

test_rmsle = round(
    np.sqrt(
        metrics.mean_squared_error(y_test, y_test_pred)
    ),
    4)

# Display and save evaluation metrics
print('score:', score)
print('RMSLE on the train set:', train_rmsle)
print('RMSLE on the test set:', test_rmsle)

with open(f_output, "w") as fd:
    json.dump({"score": score, "RMSLE_train": train_rmsle, "RMSLE_test": test_rmsle}, fd)
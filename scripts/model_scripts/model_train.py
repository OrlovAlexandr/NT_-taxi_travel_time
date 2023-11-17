import os
import pickle
import sys

import numpy as np
import yaml
from sklearn.ensemble import GradientBoostingRegressor

# Load hyperparameters for training from the YAML configuration file
params = yaml.safe_load(open("params.yaml"))["train"]

# Check if the correct number of command-line arguments is provided
if len(sys.argv) != 3:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython3 model_train.py data-file\n")
    sys.exit(1)

# Set the output path for the trained model
f_output = os.path.join("models", sys.argv[2])
os.makedirs(os.path.join("models"), exist_ok=True)

# Set the input path
train_path = sys.argv[1]
print('train path:', train_path)
print('model path:', f_output)

# Import the training data from the specified file
with open(train_path, 'rb') as f:
    train_set = np.load(f)
    X_train = train_set['X_train']
    y_train = train_set['y_train']

# Create the Gradient Boosting Regressor model with specified hyperparameters
learning_rate = params['rate']
iterations = params['iterations']
max_depth = params['max_depth']
random_state = params['seed']

model = GradientBoostingRegressor(
    learning_rate=learning_rate,
    n_estimators=iterations,
    max_depth=max_depth,
    random_state=random_state,
    min_samples_split=30,
    verbose=True
)

# Train the model on the provided training set
model.fit(X_train, y_train)

# Save the trained model to a file using Pickle serialization
with open(f_output, "wb") as f:
    pickle.dump(model, f)

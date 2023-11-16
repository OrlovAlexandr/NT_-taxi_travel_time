import os
import pickle
import sys

import numpy as np
import yaml
from sklearn.ensemble import GradientBoostingRegressor

params = yaml.safe_load(open("params.yaml"))["train"]

if len(sys.argv) != 3:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython3 model_train.py data-file\n")
    sys.exit(1)

f_output = os.path.join("models", sys.argv[2])
os.makedirs(os.path.join("models"), exist_ok=True)

# Import data
train_path = sys.argv[1]
print('train path:', train_path)
print('model path:', f_output)

with open(train_path, 'rb') as f:
    train_set = np.load(f)
    X_train = train_set['X_train']
    y_train = train_set['y_train']

# Create the model
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

# Train the model
model.fit(X_train, y_train)

# Save the model to a file
with open(f_output, "wb") as f:
    pickle.dump(model, f)

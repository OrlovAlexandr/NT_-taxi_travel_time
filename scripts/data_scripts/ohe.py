import os
import sys

import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Check if the correct number of command-line arguments is provided
if len(sys.argv) != 2:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython3 ohe.py data-file\n")
    sys.exit(1)

# Set the path to the input data
data_path = sys.argv[1]
print('data_path:', data_path)

# Set the output path for the dataset with one-hot encoded features
f_output = os.path.join("data", "stage3", "dataset_ohe.ftr")
os.makedirs(os.path.join("data", "stage3"), exist_ok=True)

# Read the dataset from the provided feather file
data = pd.read_feather(data_path)

# Encoding 'vendor_id', 'store_and_fwd_flag' binary features to 0-1 range
data['vendor_id'] = data['vendor_id'].map({1: 0, 2: 1})
data['store_and_fwd_flag'] = data['store_and_fwd_flag'].map(
    {'N': 0, 'Y': 1})

# Columns to encode using One-Hot Encoding
columns_to_change = ['pickup_day_of_week', 'geo_cluster', 'events']

# One-Hot Encode the specified columns
one_hot_encoder = OneHotEncoder(drop='first')
data_onehot = one_hot_encoder.fit_transform(data[columns_to_change])
column_names = one_hot_encoder.get_feature_names_out()

# Create a DataFrame from the encoded features
data_onehot = pd.DataFrame(data_onehot.toarray(), columns=column_names)

# Add the encoded table to the original dataset
data = pd.concat(
    [data.reset_index(drop=True).drop(columns_to_change, axis=1),
     data_onehot],
    axis=1
)

# Save the DataFrame with one-hot encoded features to a new feather file
data.to_feather(f_output)

import os
import sys

import pandas as pd

# Check if the correct number of command-line arguments is provided
if len(sys.argv) != 2:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython3 clean_features.py data-file\n")
    sys.exit(1)

# Set the path to the input data
data_path = sys.argv[1]
print('data_path:', data_path)

# Set the output path for the cleaned dataset
f_output = os.path.join("data", "stage2", "dataset_cleaned.ftr")
os.makedirs(os.path.join("data", "stage2"), exist_ok=True)

# Read the dataset from the provided feather file
data = pd.read_feather(data_path)

# Drop columns related to trip end information
# as we are predicting the end time
drop_columns = ['id', 'dropoff_datetime']
data = data.drop(drop_columns, axis=1)
print('Columns left after cleaning:', data.shape[1])

# Drop columns related to the pickup date,
# as necessary information has already been extracted
drop_columns = ['pickup_datetime', 'pickup_date']
data = data.drop(drop_columns, axis=1)
print('Shape of data:  {}'.format(data.shape))

# Save the cleaned dataframe to a new feather file
data.to_feather(f_output)

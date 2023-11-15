import os
import sys

import pandas as pd

if len(sys.argv) != 2:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython3 clean_features.py data-file\n")
    sys.exit(1)

data_path = sys.argv[1]
print('data_path:', data_path)

f_output = os.path.join("data", "stage2", "dataset_cleaned.ftr")
os.makedirs(os.path.join("data", "stage2"), exist_ok=True)

# Read dataset
data = pd.read_feather(data_path)

# Starting a trip, we can never know exactly when it will end, as we are
# trying to predict the end time of the trip.
drop_columns = ['id', 'dropoff_datetime']
data = data.drop(drop_columns, axis=1)
print('Columns left after cleaning:', data.shape[1])

# Earlier, we extracted all the necessary information from the date of
# the trip
drop_columns = ['pickup_datetime', 'pickup_date']
data = data.drop(drop_columns, axis=1)
print('Shape of data:  {}'.format(data.shape))

# Saving dataframe to file
data.to_feather(f_output)

import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

# import forward_feature_selection as ffs

# Read dataset
taxi_data = pd.read_csv("artifacts/taxi_data_df.csv")

# Take the logarithm of the trip duration feature
taxi_data['trip_duration_log'] = np.log(taxi_data['trip_duration'] + 1)

# Make a copy of the original travel table
train_data = taxi_data.copy()

# Starting a trip, we can never know exactly when it will end, as we are
# trying to predict the end time of the trip.
drop_columns = ['id', 'dropoff_datetime']
train_data = train_data.drop(drop_columns, axis=1)
print('Columns left after cleaning:', train_data.shape[1])

# Earlier, we extracted all the necessary information from the date of
# the trip
drop_columns = ['pickup_datetime', 'pickup_date']
train_data = train_data.drop(drop_columns, axis=1)
print('Shape of data:  {}'.format(train_data.shape))

# Encoding 'vendor_id', 'store_and_fwd_flag' features
train_data['vendor_id'] = train_data['vendor_id'].map({1: 0, 2: 1})
train_data['store_and_fwd_flag'] = train_data['store_and_fwd_flag'].map(
    {'N': 0, 'Y': 1})

# Columns to encode
columns_to_change = ['pickup_day_of_week', 'geo_cluster', 'events']

# OHE
one_hot_encoder = OneHotEncoder(drop='first')
data_onehot = one_hot_encoder.fit_transform(train_data[columns_to_change])
column_names = one_hot_encoder.get_feature_names_out()

# Dataframe from encoded features
data_onehot = pd.DataFrame(data_onehot.toarray(), columns=column_names)

# Add encoded table to train data
train_data = pd.concat(
    [train_data.reset_index(drop=True).drop(columns_to_change, axis=1),
     data_onehot],
    axis=1
)

# Form the observation matrix X, the target variable vector y, and its
# logarithm y_log
X = train_data.drop(['trip_duration', 'trip_duration_log'], axis=1)
y = train_data['trip_duration']
y_log = train_data['trip_duration_log']

# Split the dataset into training and validation sets in a ratio of 67/33.
# For training, the logarithmic version was chosen.
X_train, X_valid, y_train_log, y_valid_log = model_selection.train_test_split(
    X, y_log,
    test_size=0.33,
    random_state=42
)

# Choosing the 25 best features (KBest model)
k_best = SelectKBest(score_func=f_regression, k=25)
k_best.fit_transform(X_train, y_train_log)

# List of chosen features
best_features = k_best.get_feature_names_out()

X_train_k = X_train[best_features]
X_valid_k = X_valid[best_features]

print('KBest features:', best_features)
print('Not included features',
      set(best_features).symmetric_difference(set(X.columns)))
'''
# Choosing the 25 best features (Forward feature selection model)
ffs_best_features = ffs.forward_feature_selection(X, y_log,
                                                  verbose=0, number=25)

print('FFS best features:', ffs_best_features)
print('Not included features',
      set(ffs_best_features).symmetric_difference(set(X.columns)))

X_train_ffs = X_train[ffs_best_features]
X_valid_ffs = X_valid[ffs_best_features]
'''
# Scaling feature values
scaler = MinMaxScaler()
X_train_scaled_k = scaler.fit_transform(X_train_k)
X_valid_scaled_k = scaler.transform(X_valid_k)

# X_train_scaled_ffs = scaler.fit_transform(X_train_ffs)
# X_valid_scaled_ffs = scaler.transform(X_valid_ffs)

# Save data
np.savetxt('artifacts/X_train_scaled_k.csv', X_train_scaled_k, delimiter=",")
np.savetxt('artifacts/X_valid_scaled_k.csv', X_valid_scaled_k, delimiter=",")
# np.savetxt('artifacts/X_train_scaled_ffs.csv',
#            X_train_scaled_ffs, delimiter=",")
# np.savetxt('artifacts/X_valid_scaled_ffs.csv',
#            X_valid_scaled_ffs, delimiter=",")
y_train_log.to_csv('artifacts/y_train_log.csv', index=False)
y_valid_log.to_csv('artifacts/y_valid_log.csv', index=False)

print('-' * 79 + '\n')
print('Data preprocessing is finished!')

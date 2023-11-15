import os
import sys

import pandas as pd

import add_features as af

if len(sys.argv) != 5:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython3 get_features.py data-file\n")
    sys.exit(1)

train_path = sys.argv[1] #osrm
holiday_path = sys.argv[2] #train
osrm_path = sys.argv[3] #holiday
weather_path = sys.argv[4] #weather
print('train_path', train_path)
print('holiday_path', holiday_path)
print('osrm_path', osrm_path)
print('weather_path', weather_path)


f_output = os.path.join("data", "stage1", "taxi_dataset.csv")
os.makedirs(os.path.join("data", "stage1"), exist_ok=True)

taxi_data = pd.read_csv(train_path)
print('Train data shape: {}'.format(taxi_data.shape))

# Convert to datetime
taxi_data['pickup_datetime'] = pd.to_datetime(taxi_data['pickup_datetime'],
                                              format='%Y-%m-%d %H:%M:%S')
print(f'\nMissing data in train dataframe: {taxi_data.isna().sum().sum()}')

# Adding datetime features
taxi_data = af.add_datetime_features(taxi_data)

# Adding holiday features
holiday_data = pd.read_csv(holiday_path, sep=';')
holiday_data['date'] = pd.to_datetime(holiday_data['date'],
                                      format='%Y-%m-%d').dt.date

taxi_data = af.add_holiday_features(taxi_data, holiday_data)

# Adding OSRM features
osrm_data = pd.read_csv(osrm_path,
                        usecols=['id', 'total_distance', 'total_travel_time',
                                 'number_of_steps']
                        )

taxi_data = af.add_osrm_features(taxi_data, osrm_data)
print(f"Missing data after adding OSRM: {taxi_data['total_distance'].isna().sum()}")

# Adding geographical features
af.add_geographical_features(taxi_data)

# Adding cluster features
af.add_cluster_features(taxi_data)

# Adding weather features
weather_data = pd.read_csv(weather_path,
                           usecols=['date', 'hour',
                                    'temperature', 'visibility',
                                    'wind speed', 'precip', 'events']
                           )

weather_data['pickup_date'] = pd.to_datetime(weather_data['date'],
                                             format='%Y-%m-%d').dt.date
weather_data['pickup_hour'] = weather_data['hour'].astype('int64')
weather_data = weather_data.drop(columns=['date', 'hour'])

taxi_data = af.add_weather_features(taxi_data, weather_data)
print(f'\nMissing data after adding weather features: {taxi_data.isna().sum().sum()}')

# Filling missing data in weather features
af.fill_null_weather_data(taxi_data)
print(f'\nMissing data after filling null: {taxi_data.isna().sum().sum()}')

# Remove extremely long trips
taxi_data = taxi_data[taxi_data['trip_duration'] <= 24 * 3600]

# Remove extremely fast trips
avg_speed = taxi_data['total_distance'] / taxi_data['trip_duration'] * 3.6
taxi_data = taxi_data[avg_speed <= 300]

# Saving dataframe to file
taxi_data.to_csv(f_output, index=False)

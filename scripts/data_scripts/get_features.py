import os
import sys

import pandas as pd

import add_features as af

# Check if the correct number of command-line arguments is provided
if len(sys.argv) != 5:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython3 get_features.py data-file\n")
    sys.exit(1)

# Set input paths from command-line arguments
train_path = sys.argv[1]
holiday_path = sys.argv[2]
osrm_path = sys.argv[3]
weather_path = sys.argv[4]
print('train_path:', train_path)
print('holiday_path:', holiday_path)
print('osrm_path:', osrm_path)
print('weather_path:', weather_path)

# Set output path for the final dataset
f_output = os.path.join("data", "stage1", "taxi_dataset.ftr")
os.makedirs(os.path.join("data", "stage1"), exist_ok=True)

# Read the main dataset
taxi_data = pd.read_csv(train_path)

# Convert the 'pickup_datetime' column to datetime format
taxi_data['pickup_datetime'] = pd.to_datetime(taxi_data['pickup_datetime'],
                                              format='%Y-%m-%d %H:%M:%S')
print(f'\nMissing data in train dataframe: {taxi_data.isna().sum().sum()}')

# Add datetime features
taxi_data = af.add_datetime_features(taxi_data)

# Add holiday features
holiday_data = pd.read_csv(holiday_path, sep=';')
holiday_data['date'] = pd.to_datetime(holiday_data['date'],
                                      format='%Y-%m-%d').dt.date

taxi_data = af.add_holiday_features(taxi_data, holiday_data)

# Add OSRM features
osrm_data = pd.read_csv(osrm_path,
                        usecols=['id', 'total_distance', 'total_travel_time',
                                 'number_of_steps']
                        )

taxi_data = af.add_osrm_features(taxi_data, osrm_data)
print(f"Missing data after adding OSRM: {taxi_data['total_distance'].isna().sum()}")

# Add geographical features
af.add_geographical_features(taxi_data)

# Add cluster features
af.add_cluster_features(taxi_data)

# Add weather features
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

# Fill missing data in weather features
af.fill_null_weather_data(taxi_data)
print(f'\nMissing data after filling null: {taxi_data.isna().sum().sum()}')

# Remove extremely long trips
taxi_data = taxi_data[taxi_data['trip_duration'] <= 24 * 3600]

# Remove extremely fast trips
avg_speed = taxi_data['total_distance'] / taxi_data['trip_duration'] * 3.6
taxi_data = taxi_data[avg_speed <= 300]

# Save the processed dataframe to a file
taxi_data.reset_index().iloc[:, 1:].to_feather(f_output)

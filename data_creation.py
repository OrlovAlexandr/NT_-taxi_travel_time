import pandas as pd

import add_features as af

taxi_data = pd.read_csv("datasets/train.csv")

print('Train data shape: {}'.format(taxi_data.shape))

# Convert to datetime
taxi_data['pickup_datetime'] = pd.to_datetime(taxi_data['pickup_datetime'],
                                              format='%Y-%m-%d %H:%M:%S')

# Missing data
print(f'\nMissing data: {taxi_data.isna().sum().sum()}')

# Adding datetime features
taxi_data = af.add_datetime_features(taxi_data)

# Adding holiday features
holiday_data = pd.read_csv('datasets/holiday_data.csv', sep=';')
holiday_data['date'] = pd.to_datetime(holiday_data['date'],
                                      format='%Y-%m-%d').dt.date

taxi_data = af.add_holiday_features(taxi_data, holiday_data)

# Adding OSRM features
osrm_data = pd.read_csv('datasets/osrm_data_train.zip',
                        usecols=['id', 'total_distance', 'total_travel_time',
                                 'number_of_steps']
                        )

taxi_data = af.add_osrm_features(taxi_data, osrm_data)

# Missing data
print(f"Missing data: {taxi_data['total_distance'].isna().sum()}")

# Adding geographical features
af.add_geographical_features(taxi_data)

# Adding cluster features
af.add_cluster_features(taxi_data)

# Adding weather features
weather_data = pd.read_csv('datasets/weather_data.zip',
                           usecols=['date', 'hour',
                                    'temperature', 'visibility',
                                    'wind speed', 'precip', 'events']
                           )
weather_data['pickup_date'] = pd.to_datetime(weather_data['date'],
                                             format='%Y-%m-%d').dt.date
weather_data['pickup_hour'] = weather_data['hour'].astype('int64')
weather_data = weather_data.drop(columns=['date', 'hour'])

taxi_data = af.add_weather_features(taxi_data, weather_data)

# Missing data
print(f'\nMissing data: {taxi_data.isna().sum().sum()}')

# Filling missing data in weather features
af.fill_null_weather_data(taxi_data)
print(taxi_data.loc[1:2, :].transpose())

# Missing data
print(f'\nMissing data: {taxi_data.isna().sum().sum()}')

# Remove extremely long trips
taxi_data = taxi_data[taxi_data['trip_duration'] <= 24 * 3600]

# Remove extremely fast trips
avg_speed = taxi_data['total_distance'] / taxi_data['trip_duration'] * 3.6
taxi_data = taxi_data[avg_speed <= 300]

# Saving dataframe to file
taxi_data.to_csv('artifacts/taxi_data_df.csv', index=False)

print('-' * 79 + '\n')
print('Data creation is finished!')

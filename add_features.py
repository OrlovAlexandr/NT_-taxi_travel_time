import numpy as np
from sklearn import cluster

import calc_distance as cd


def add_datetime_features(df):
    """This function adds date, hour and day of week number features to the
    dataframe

    Args:
        df (DataFrame): source dataframe

    Returns:
        DataFrame with new datetime features
    """
    df['pickup_date'] = df['pickup_datetime'].dt.date
    df['pickup_hour'] = df['pickup_datetime'].dt.hour
    df['pickup_day_of_week'] = df['pickup_datetime'].dt.dayofweek
    return df


def add_holiday_features(df, holiday_df):
    """This function adds a feature about holidays to the dataframe

    Args:
        df (DataFrame): source dataframe
        holiday_df (DataFrame): dataframe with holiday features

    Returns:
        DataFrame with binary feature whether it is a holiday or not
    """
    df['pickup_holiday'] = \
        df['pickup_date'].isin(holiday_df['date']).astype('int')

    return df


def add_osrm_features(df, df_osrm):
    """This function adds OSRM data, which represents the shortest route based
    on geographical data

    Args:
        df (DataFrame): source dataframe
        df_osrm (DataFrame): dataframe with OSR  data

    Returns:
        Updated travel data dataframe with 3 columns added:
        'total_distance', 'total_travel_time', 'number_of_steps'.
    """
    df = df.merge(df_osrm, on='id', how='left')

    return df


def add_geographical_features(df):
    """This function adds geographic features such as haversine distance and
    direction

    Args:
        df (DataFrame): source dataframe

    Returns:
        Updated travel data dataframe with 2 columns added:
        'haversine_distance', 'direction'.
    """
    df['haversine_distance'] = cd.get_haversine_distance(
        df['pickup_latitude'],
        df['pickup_longitude'],
        df['dropoff_latitude'],
        df['dropoff_longitude'])
    df['direction'] = cd.get_angle_direction(
        df['pickup_latitude'],
        df['pickup_longitude'],
        df['dropoff_latitude'],
        df['dropoff_longitude'])

    return df


def add_cluster_features(df):
    """
    Clustering based on geographical coordinates.
    Group all trips based on the coordinates of the start and end points of the
    trip using clustering methods, thereby adding information about the areas
    where the starting and ending points of the trips are located.
    Args:
        df (DataFrame): source dataframe

    Returns:
        DataFrame with geo_cluster feature
    """

    # Creating a training set from the geographical coordinates of all points
    coords = np.hstack((df[['pickup_latitude', 'pickup_longitude']],
                        df[['dropoff_latitude', 'dropoff_longitude']]))

    # Training the clustering algorithm
    kmeans = cluster.KMeans(n_clusters=10, random_state=42, n_init=10)
    kmeans.fit(coords)

    # Predicting
    df['geo_cluster'] = kmeans.predict(coords)

    return df


def add_weather_features(df, weather_df):
    """This function adds weather features such as temperature, visibility,
    wind speed, precipitation, events

    Args:
        df (DataFrame): source dataframe
        weather_df (DataFrame): weather features dataframe

    Returns:
        DataFrame with weather features
    """
    df = df.merge(weather_df, how='left', on=['pickup_date', 'pickup_hour'])

    return df


def fill_null_weather_data(df):
    """This function fills missing data in weather and OSRM features

    Args:
        df (DataFrame): source dataframe

    Returns:
        DataFrame with filled data in weather and OSRM features
    """
    for column in ['temperature', 'visibility', 'wind speed', 'precip']:
        df[column] = df[column].fillna(df.groupby('pickup_date')[column].
                                       transform('median'))

    df['events'] = df['events'].fillna('None')

    for column in ['total_distance', 'total_travel_time', 'number_of_steps']:
        df[column] = df[column].fillna(df[column].median())

    return df

import numpy as np


def get_haversine_distance(lat1, lng1, lat2, lng2):
    """Calculating the shortest distance by the Haversine formula

    Args:
        lat1 (Series): Start latitude
        lng1 (Series): Start longitude
        lat2 (Series): End latitude
        lng2 (Series): End longitude

    Returns:
        float: Haversine distance in km

    """
    # convert angles to radians
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    # radius of the Earth in kilometers
    EARTH_RADIUS = 6371
    # calculate the shortest distance h using the Haversin formula
    lat_delta = lat2 - lat1
    lng_delta = lng2 - lng1
    d = np.sin(lat_delta * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * \
        np.sin(lng_delta * 0.5) ** 2
    h = 2 * EARTH_RADIUS * np.arcsin(np.sqrt(d))
    return h


def get_angle_direction(lat1, lng1, lat2, lng2):
    """Calculating the direction angle alpha using the formula of bearing
    angle

    Args:
        lat1 (Series): Start latitude
        lng1 (Series): Start longitude
        lat2 (Series): End latitude
        lng2 (Series): End longitude

    Returns:
        float: direction angle
    """
    # convert angles to radians
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    # calculating the direction angle alpha using the formula of bearing
    lng_delta_rad = lng2 - lng1
    y = np.sin(lng_delta_rad) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * \
        np.cos(lng_delta_rad)
    alpha = np.degrees(np.arctan2(y, x))
    return alpha

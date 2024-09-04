"""
Collection of tools for processing WATLAS data.

"""
import sys

import polars as pl
from datetime import datetime, timezone


def get_watlas_data(tags, tracking_time_start, tracking_time_end,
                    sqlite_file='sqlite://../data/SQLite/watlas-2023.sqlite'):
    """
    Get watlas data from a local SQLite database file and return it as a polars dataframe.
    Results will be filtered to fit between start.
    Args:
        tags (list): list of WATLAS tags
        tracking_time_start (str): start of tracking time
        tracking_time_end (str): end of tracking time
        sqlite_file:

    Returns:
        raw_watlas_df (pd.DataFrame): a polars dataframe containing WATLAS data

    """

    # format tag to server tag numbers
    server_tags = ', '.join(f'3100100{str(t)}' for t in tags)

    # TODO implement timezone conversion
    # convert sting to datetime object in utc time zone
    tracking_time_start = datetime.strptime(tracking_time_start, '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)
    tracking_time_end = datetime.strptime(tracking_time_end, '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)

    # format time to unix timestamp in milliseconds
    tracking_time_start = int(tracking_time_start.timestamp() * 1000)
    tracking_time_end = int(tracking_time_end.timestamp() * 1000)

    # set query for sqlite database
    query = f"""
    SELECT TAG, TIME, X, Y, NBS, VARX, VARY, COVXY
    FROM LOCALIZATIONS
    WHERE TAG IN ({server_tags})
      AND TIME > {tracking_time_start}
      AND TIME < {tracking_time_end}
    ORDER BY TIME ASC;
    """

    # run query on sqlite file and get polars dataframe
    raw_watlas_df = pl.read_database_uri(query, sqlite_file)

    # check if dataframe is empty
    if raw_watlas_df.shape[0] == 0:
        # let user know and exit
        print("No data found in SQLite file for given tags and times. No data to process.")
        sys.exit(1)

    return raw_watlas_df


# TODO get speed
def get_simple_distance(watlas_df):
    """
    Gets the Euclidean distance between consecutive localization in a coordinate

    Args:
        watlas_df (pl.DataFrame): a polars dataframe containing WATLAS data:

    Returns:
        dist_series: a polars series with euclidian distances.

    """

    dist_series = (
        (
            # get the difference between the current coordinate and the next coordinate using shift
            # multiply by the power of 2
            (watlas_df["X"] - watlas_df["X"].shift(1)) ** 2 +
            (watlas_df["Y"] - watlas_df["Y"].shift(1)) ** 2
        ) ** 0.5  # multiply by power of 0.5 to get square root, to get euclidian distance
    )

    return dist_series


def get_speed(watlas_df):
    ...


# TODO get turn angle

# TODO thin data

# TODO filter data

if __name__ == '__main__':
    data = get_watlas_data([3001],
                           tracking_time_start="2023-08-21 09:00:00",
                           tracking_time_end="2023-08-21 10:20:00")
    print("watlas data found: ")
    print(data)

    get_simple_distance(data)

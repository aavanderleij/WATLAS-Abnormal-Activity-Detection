"""
Collection of tools for processing WATLAS data.

"""

import polars as pl
from datetime import datetime, timezone


# TODO read SQLite file

def get_watlas_data(tags, tracking_time_start, tracking_time_end,
                    sqlite_file='sqlite://../data/SQLite/watlas-2023.sqlite'):
    """
    Get watlas data from a local SQLite database file and return it as a polars dataframe.
    Results will be filtered to fit between start and end time and on given tag list.
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

    # check if time is the same as input
    start_time = datetime.utcfromtimestamp(tracking_time_start / 1000)
    end_time = datetime.utcfromtimestamp(tracking_time_end / 1000)

    return raw_watlas_df


# TODO get speed

# TODO get turn angle

# TODO thin data

# TODO filter data

if __name__ == '__main__':
    data = get_watlas_data([3001],
                           tracking_time_start="2023-08-21 09:00:00",
                           tracking_time_end="2023-08-21 10:20:00")
    print("watlas data found: ")
    print(data)

"""
Collection of tools for processing WATLAS data.

"""
import sys

import numpy as np
import polars as pl
from datetime import datetime, timezone
import math
import timeit


def get_watlas_data(tags, tracking_time_start, tracking_time_end,
                    sqlite_file='sqlite://../data/SQLite/watlas-2023.sqlite'):
    """
    Get watlas data from a local SQLite database file and return it as a polars dataframe.
    Results will be of the specified tags and filtered to fit between start time and end time.

    Dataframe legend:
    TAG 	=	11 digit WATLAS tag ID
    TIME	=	UNIX time (milliseconds)
    X		=	X-ccordinates in meters (utm 31 N)
    Y		=	Y-ccordinates in meters (utm 31 N)
    NBS		=	Number of Base Stations used in calculating coordinates
    VARX	=	Variance in estimating X-coordinates
    VARY	=	Variance in estimating Y-coordinates
    COVXY	=	Co-variance between X- and Y-coordinates

    Args:
        tags (list): list of WATLAS tags
        tracking_time_start (str): start of tracking time
        tracking_time_end (str): end of tracking time
        sqlite_file:

    Returns:
        raw_watlas_df (pd.DataFrame): a polars dataframe with localizations of the specified tag,
         filtered between the start and end times.

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

    raw_watlas_df = get_datetime(raw_watlas_df)

    return raw_watlas_df


def get_datetime(watlas_df):
    watlas_df = watlas_df.with_columns(
        pl.from_epoch(pl.col("TIME"), time_unit="ms").alias("timestamp")
    )
    return watlas_df


def get_simple_distance(watlas_df):
    """
    Gets the Euclidean distance in meters between consecutive localization in a coordinate.

    Args:
        watlas_df (pl.DataFrame): a polars dataframe containing WATLAS data

    Returns:
        dist_series: a polars series with euclidian distances.

    """

    dist_series = (
            (
                # get the difference between the current coordinate and the next coordinate in the X and Y columns
                # multiply by the power of 2
                    (watlas_df["X"] - watlas_df["X"].shift(1)) ** 2 +
                    (watlas_df["Y"] - watlas_df["Y"].shift(1)) ** 2
            ) ** 0.5  # multiply by power of 0.5 to get square root, to get euclidian distance
    )

    return dist_series


def get_speed(watlas_df):
    """
    Calculate speed in meters per second for a watlas dataframe.

    Args:
        watlas_df (pl.DataFrame): a polars dataframe containing WATLAS data

    Returns:
        dist_series: a polars series with speed in meters per second.

    """
    # get distance
    distance = get_simple_distance(watlas_df)
    # get the time interval between rows in the "TIME" column
    time = (watlas_df["TIME"] - watlas_df["TIME"].shift(1)) / 1000
    # calculate speed
    speed = distance / time

    return speed


def get_turn_angle(watlas_df):
    """
    Calculate turn angle in degrees for a watlas dataframe.
    Using the law of cosines this function returns the turning angle in degrees based on the x an y coordinates.
    Negative  degrees indicate left turns (counter-clockwise)

    Args:
        watlas_df (pl.DataFrame): a polars dataframe containing WATLAS data

    Returns:
        angle: a numpy array with turning angle in degrees.

    """
    # Create lagged versions of X
    x1 = watlas_df["X"][:-2]
    x2 = watlas_df["X"][1:-1]
    x3 = watlas_df["X"][2:]

    # Create lagged version of Y
    y1 = watlas_df["Y"][:-2]
    y2 = watlas_df["Y"][1:-1]
    y3 = watlas_df["Y"][2:]

    dist_x1_x2 = np.sqrt(((x2 - x1) ** 2) + ((y2 - y1) ** 2))
    dist_x2_x3 = np.sqrt(((x3 - x2) ** 2) + ((y3 - y2) ** 2))
    dist_x3_x1 = np.sqrt(((x3 - x1) ** 2) + ((y3 - y1) ** 2))

    angle = np.acos((
                            (dist_x1_x2 ** 2) +
                            (dist_x2_x3 ** 2) -
                            (dist_x3_x1 ** 2)
                    ) /
                    (2 * dist_x1_x2 * dist_x2_x3)
                    )

    # convert to degrees
    angle = angle * 180 / math.pi

    # subtract from 180 to get the external angle
    angle = 180 - angle

    # insert np at the end and beginning to keep length of array the same as the dataframe
    angle = np.insert(angle, 0, np.nan)
    angle = np.append(angle, np.nan)

    return angle


def aggregate_dataframe(watlas_df, interval="15s"):
    """
    Aggregate a polars dataframe containing WATLAS data to the time specified interval.
    This thins the data to only have rows with given intervals.

    Args:
        watlas_df (pl.DataFrame): a polars dataframe containing WATLAS data
        interval (str): the time interval to aggregate (default 15 seconds)

    Returns:
        dist_series: a polars series with euclidian distances.
    """
    # TODO check if behavior is desired. atl_thin_data does always not give the same result but this more of how I
    #  would expect it to work. atl_thin_data uses a rounded TIME to bucket its data and this uses left leaning
    #  timestamp to group dataframe by time into intervals. This gives more even timestamps.
    watlas_df = watlas_df.group_by_dynamic("timestamp", every=interval, group_by="TAG").agg(
        [
            # aggregate columns X, Y and NBS by getting the mean of those values per interval
            # drop COVXY, covariance loses meaning if an average is taken.
            # TIME is now the average unix time
            pl.col("*").exclude("VARX", "VARY", "COVXY").mean(),
            # the variance of an average is the sum of variances / sample size square
            (pl.col("VARX").sum() / (pl.col("VARX").count() ** 2)).alias("VARX"),
            (pl.col("VARY").sum() / (pl.col("VARY").count() ** 2)).alias("VARY"),
        ]
    )

    # set float to full avoid scientific notation of numbers
    pl.Config(set_fmt_float="full")
    return watlas_df


def smooth_data(watlas_df, moving_window=5):
    """
    Applies a median smooth defined by a rolling window to the X and Y

    Args:
        watlas_df (pl.DataFrame): a polars dataframe containing WATLAS data:
        moving_window (int): the window size:

    Returns:
        smooth_df: a polars dataframe with median smoothed data.

    """
    # TODO there is a shift in the smoothing compard to the r function runmed form the stats librabry. needs discussion.
    smooth_df = watlas_df.with_columns(
        pl.col("X").alias("X_raw"),  # Keep original values
        pl.col("Y").alias("Y_raw"),  # Keep original values
        # Apply the forward and reverse rolling median on X
        pl.col("X")
        .reverse()
        .rolling_median(window_size=moving_window, min_periods=1)
        .reverse()
        .rolling_median(window_size=moving_window, min_periods=1)
        .alias("X"),
        # Apply the forward and reverse rolling median on Y
        pl.col("Y")
        .reverse()
        .rolling_median(window_size=moving_window, min_periods=1)
        .reverse()
        .rolling_median(window_size=moving_window, min_periods=1)
    )

    return smooth_df


if __name__ == '__main__':
    # TODO time process
    start = timeit.default_timer()
    # data = get_watlas_data([3001],
    #                        tracking_time_start="2023-08-21 09:20:00",
    #                        tracking_time_end="2023-08-21 10:20:00")

    data = get_watlas_data([3001],
                           tracking_time_start="2023-08-01 00:00:00",
                           tracking_time_end="2023-08-21 00:00:00")

    # print("watlas data found: ")
    # print(data)
    #
    # print()
    # print("speed:")
    # print(get_speed(data))
    # print()
    # print("turn angle:")
    # print(get_turn_angle(data))
    print("aggrage data")
    argg_data = aggregate_dataframe(data)
    smooth_df = (smooth_data(data))
    print(smooth_df)
    print(smooth_df.select("X").head())
    stop = timeit.default_timer()



    print('Time: ', stop - start)

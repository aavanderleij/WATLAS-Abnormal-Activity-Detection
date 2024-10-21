"""
This class create an object with WATLAS tracking data and calculates varius other values based on this data.

WATLAS is project from NIOZ, the Royal Netherlands Institute for Sea Research.

For technical specifications or other information, please contact Allert Bijleveld (allert.bijleveld@nioz.nl)

author: Antsje van der Leij (https://github.com/aavanderleij)
"""
# imports
import os
import sys
import math
import timeit
import warnings
import numpy as np
from datetime import datetime, timezone
import polars as pl


class WatlasDataframe:
    """
    Class for retrieving and processing WATLAS data.
    """

    def __init__(self, watlas_df=None, tags_df=None):

        # set instance variable
        self.tags_df = tags_df
        self.watlas_df = watlas_df
        self.species_list = ["islandica", "oystercatcher", "spoonbill", "bar-tailed godwit", "redshank", "sanderling",
                             "dunlin", "turnstone", "curlew"]

    # TODO add remote database access
    def get_watlas_data_sqlite(self, tags, tracking_time_start, tracking_time_end, sqlite_path):
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
        # format sql uri
        sqlite_file = "sqlite://" + sqlite_path
        print(sqlite_file)

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
        self.watlas_df = pl.read_database_uri(query, sqlite_file)

        # check if dataframe is empty
        if self.watlas_df.shape[0] == 0:
            # let user know and exit
            print("No data found in SQLite file for given tags and times. No data to process.")
            sys.exit(1)

        # add short tag number column
        self.watlas_df = self.watlas_df.with_columns([
            pl.col("TAG"),
            pl.col("TAG").cast(pl.String).str.slice(-4).cast(pl.Int64).alias("tag")])

        # get readable time
        self.get_datetime()

        return self.watlas_df

    def get_watlas_dataframe(self):
        """
        Return watlas dataframe.

        Returns:
            watlas_df (pd.DataFrame): a polars dataframe with WATLAS data.

        """
        # set float to full avoid scientific notation of numbers
        pl.Config(set_fmt_float="full")
        return self.watlas_df

    def get_datetime(self):
        """
        Adds column to dataframe that contains a human-readable date and time.
        """

        self.watlas_df = self.watlas_df.with_columns(
            # convert unix time from TIME column to human-readable time
            pl.from_epoch(pl.col("TIME"), time_unit="ms").alias("time")
        )

    def aggregate_dataframe(self, interval="15s"):
        """
        Aggregate a polars dataframe containing WATLAS data to the time specified interval.
        This thins the data to only have rows with given intervals.

        Args:
            interval (str): the time interval to aggregate (default 15 seconds)

        """
        self.watlas_df = self.watlas_df.group_by_dynamic("time", every=interval, group_by="TAG").agg(
            [
                # aggregate columns X, Y and NBS by getting the mean of those values per interval
                # drop COVXY, covariance loses meaning if an average is taken.
                pl.col("*").exclude("VARX", "VARY", "COVXY", "TIME", "tag").exclude(pl.Utf8).mean(),

                # keep first value of string columns
                pl.col(pl.Utf8).first(),
                # keep first value of TIME
                pl.col("TIME").first(),
                # keep "tag" as int
                pl.col("tag").first(),

                # the variance of an average is the sum of variances / sample size square
                (pl.col("VARX").sum() / (pl.col("VARX").count() ** 2)).alias("VARX"),
                (pl.col("VARY").sum() / (pl.col("VARY").count() ** 2)).alias("VARY"),
            ]
        )

    def filter_num_localisations(self, min_num_localisations=4):
        """
        If a tag appears less then this number of localisations it's removed from the dataframe.

        Args:
            min_num_localisations (int): minimum number of localisations

        """
        # count times tag is in df
        tag_count = self.watlas_df.group_by("tag").len()
        # get tags that are less than min_num_localisations
        tags_to_remove = tag_count.filter(pl.col("len") < min_num_localisations).select("tag")
        # filter tags if tag is not in tags_to_remove
        self.watlas_df = self.watlas_df.filter(~pl.col("tag").is_in(tags_to_remove))

    def get_tag_data(self, tag_csv_path):
        """
        read exel file (.xmlx) containing tag data

        Args:
            tag_csv_path (str): the tag csv file path:

        Returns:
            tag_df: a polars dataframe with tag data from tag file.

        """
        # TODO check if file exist and contains right columns
        self.tags_df = pl.read_excel(tag_csv_path)

    def get_all_tags(self):
        """
        Get all tag numbers from tag_df
        Returns:
            all_tags (pl.Series): a list of all tag numbers.

        """

        all_tags = self.tags_df.get_column("tag")

        return all_tags

    def get_species(self):
        """
        match tag ids to id's in tag_df to get the species of every instace in the dataframe

        """

        # check if tags_df is not none
        if self.tags_df is None:
            # warn user species is not added because tag_df is none
            warnings.warn("No tag data found, species not added! Load tag data with get_tag_data before executing "
                          "this function!")
        else:
            # match tag id's to get species
            self.watlas_df = self.watlas_df.join(self.tags_df.select(["tag", "species"]), on="tag", how="left")

    def add_tide_data(self, path_tide_data):
        """
        Add water level to watlas dataframe
        Args:
            path_tide_data (str): the path to the tide data file:

        Returns:

        """
        # load in tidal data csv
        tide_df = pl.read_csv(path_tide_data, dtypes={"waterlevel": pl.Float64,
                                                      "dateTime": pl.Datetime(time_unit="ms")})
        # drop unnecessary columns
        tide_df = tide_df.drop(["time", "date"])
        # rename column to be the same as watlas_df
        tide_df = tide_df.rename({"dateTime": "time"})

        # check if sorted
        if not self.watlas_df["time"].is_sorted():
            # if not sorted, sort
            self.watlas_df = self.watlas_df.sort(by="time")
        # check if sorted
        if not tide_df["time"].is_sorted():
            # if not sorted, sort
            tide_df = tide_df.sort(by="time")

        # join tidal dataframe with watlas_df on time, this adds the closest measured water level to watlas_df
        self.watlas_df = self.watlas_df.join_asof(tide_df, on="time", strategy="nearest")

    def process_for_prediction(self, start_time, end_time, sqlite_path, tag_file):
        """
        get WATLAS data and process it for prediction.

        Args:
            start_time (str): start time for data fetching
            end_time (str): end time for data fetching
            tag_file (str): tag csv file path:
            sqlite_path (str): sqlite file path:
        """
        print("in process_for_prediction")

        column_names = ["speed", "distance", "turn_angle"]
        self.get_tag_data(tag_file)

        # get data from sqlite file
        all_tags = self.get_all_tags()
        self.get_watlas_data_sqlite(all_tags,
                                    tracking_time_start=start_time,
                                    tracking_time_end=end_time, sqlite_path=sqlite_path)

        # filter minimum localisations
        self.filter_num_localisations()

        # aggregate data
        self.aggregate_dataframe()

        # add species column
        self.get_species()

        # remove species not in species list
        self.watlas_df = self.watlas_df.filter(pl.col("species").is_in(self.species_list))

        # empty dataframe list
        df_list = []

        # group data frame by time stamp
        for time, wat_df in self.watlas_df.group_by("time"):
            # get distance
            wat_df = get_group_metrics(wat_df, 500, self.species_list)

            # cast columns to floats
            wat_df = wat_df.with_columns([
                pl.col("mean_dist").cast(pl.Float64),
                pl.col("median_dist").cast(pl.Float64),
                pl.col("std_dist").cast(pl.Float64)
            ])

            df_list.append(wat_df)

        # concat dataframes into single dataframe
        self.watlas_df = pl.concat(df_list)

        df_list = []
        # smooth and calculate per tag (these calculations have to be done per individual bird)
        for tag, wat_df in self.watlas_df.group_by("tag"):
            # order by time
            wat_df = wat_df.sort(by="TIME")
            # apply median smooth
            wat_df = smooth_data(wat_df)
            # calculate distance and speed per tag
            wat_df = get_speed(wat_df)
            # calculate turn angle per tag
            wat_df = get_turn_angle(wat_df)
            # get speed, distance and turn angle form the previous rows
            wat_df = wat_df.with_columns(
                [pl.col(col_name).shift(1).alias(f"{col_name}_1") for col_name in column_names] +
                [pl.col(col_name).shift(2).alias(f"{col_name}_2") for col_name in column_names] +
                [pl.col(col_name).shift(3).alias(f"{col_name}_3") for col_name in column_names] +
                [pl.col(col_name).shift(4).alias(f"{col_name}_4") for col_name in column_names]
            )

            # add tag to list
            df_list.append(wat_df)

        # concat dataframes to single dataframe
        self.watlas_df = pl.concat(df_list)

        # # sort by time
        self.watlas_df = self.watlas_df.sort(by="time")
        self.add_tide_data("data/watlas_data/allYears-gemeten_waterhoogte-west_terschelling-clean-UTC.csv")
        # set NaN and NULLs to 0
        self.watlas_df = self.watlas_df.fill_nan(0).fill_null(0)

        # write to csv
        self.watlas_df.write_csv("watlas_all.csv")
        print("processed data for prediction!")
        print("results saved in:")
        print(os.path.abspath("watlas_all.csv"))


def smooth_data(watlas_df, moving_window=5):
    """
    Applies a median smooth defined by a rolling window to the X and Y

    Args:
        watlas_df (pl.Datafame):
        moving_window (int): the window size:
    """
    # TODO there is a shift in the smoothing compared to the r function runmed from the stats library.
    watlas_df = watlas_df.with_columns(
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
    return watlas_df


def get_simple_travel_distance(watlas_df):
    """
    Gets the Euclidean distance in meters between consecutive localization in a coordinate.
    Add claculated distance to each as column "distance"
    """

    dist_series = (
            (
                # get the difference between the current coordinate and the next coordinate in the X and Y columns
                # multiply by the power of 2
                    (watlas_df["X"] - watlas_df["X"].shift(1)) ** 2 +
                    (watlas_df["Y"] - watlas_df["Y"].shift(1)) ** 2
            ) ** 0.5  # multiply by power of 0.5 to get square root, to get euclidian distance
    )

    # add dist to dataframe
    watlas_df = watlas_df.with_columns(dist_series.alias("distance"))

    return watlas_df


def get_speed(watlas_df):
    """
    Calculate speed in meters per second for a watlas dataframe.
    Add claculated speed as column "speed"
    """
    # check if distance is already calculated
    if "distance" not in watlas_df.columns:
        watlas_df = get_simple_travel_distance(watlas_df)

    # get distance
    distance = watlas_df["distance"]
    # get the time interval between rows in the "TIME" column
    time = (watlas_df["TIME"] - watlas_df["TIME"].shift(1)) / 1000
    # calculate speed
    speed = distance / time

    watlas_df = watlas_df.with_columns(speed.alias("speed"))

    return watlas_df


def get_turn_angle(watlas_df):
    """
    Calculate turn angle in degrees for a watlas dataframe.
    Using the law of cosines this function returns the turning angle in degrees based on the x an y coordinates.
    Negative  degrees indicate left turns (counter-clockwise)

    Add claculated turn angle as column "turn_angle"
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

    watlas_df = watlas_df.with_columns(pl.Series(angle).alias("turn_angle"))

    return watlas_df


def count_species(watlas_df, species_list):
    """
    Counts the number of species in a dataframe.
    Args:
        watlas_df:

    Returns:

    """
    species_count_dict = {}
    # Loop through each unique species and calculate the count
    for species in species_list:
        species_count = watlas_df.filter(pl.col("species") == species).height
        species_count_dict[f"{species}_in_group"] = [species_count]

    # Create a new DataFrame with the species counts
    return species_count_dict


def get_group_metrics(watlas_df, group_area, species_list):
    """
    Calculate distance between individuals. Return the mean, median, std deviation of all individuals within
    group_area

    Args:
        watlas_df (pl.dataframe): training dataframe of individuals that are in the same timeframe
        group_area (int): Maximum distance between individuals to be considered part of same group in meters

    Returns:
        watlas_df (pl.dataframe): training dataframe with new columns "mean_dist", "median_dist", "std_dist"

    """
    # set empty list
    mean_distances = []
    median_distances = []
    std_distances = []
    group_size = []
    n_species = []

    # set empty dict
    species_dict = {}

    # get x and y columns from dataframe
    x_coords = watlas_df["X"]
    y_coords = watlas_df["Y"]

    for animal in species_list:
        species_dict[f"{animal}_in_group"] = []

    # per row in watlas_df
    for i in range(watlas_df.shape[0]):
        # TODO this is not done! Its a mess.... very slow and species is idk....
        # get current row coords
        x1, y1 = watlas_df[i, "X"], watlas_df[i, "Y"]

        # get distances of current row
        distances = ((x_coords - x1) ** 2 + (y_coords - y1) ** 2).sqrt()

        # get boolian mask for when distance is not 0 (self) or distance is smaller than group_area
        mask = (distances != 0) & (distances <= group_area)

        # filter distances
        distances = distances.filter(mask)

        # filter self and member out of group
        group = watlas_df.filter(mask)
        # get group size and species
        group_size.append(group.height)
        group_species = group["species"].to_list()
        n_species.append(len(set(group_species)))

        for animal in species_list:
            if animal in group_species:
                species_dict[f"{animal}_in_group"].append(1)
            else:
                species_dict[f"{animal}_in_group"].append(0)

        # if there are more one member of group
        if distances.shape[0] != 0:

            # get mean, median and standard deviation
            mean = distances.mean()
            median_dist = distances.median()
            std_dist = distances.std()

        else:
            # if there are no members of group, set all to 0
            mean = 0.0
            median_dist = 0.0
            std_dist = 0.0

        mean_distances.append(mean)
        median_distances.append(median_dist)
        std_distances.append(std_dist)

        # print(f"Distances for row {i}: {distances}")
        # print(f"mean distance: {mean}")
        # print(f"Mask for row {i}: {mask}")
        # print(f"Group for row {i}: {group}")
        # print(f"Number of species for row {i}: {n_species[i]}")

    watlas_df = watlas_df.with_columns(
        pl.Series("mean_dist", mean_distances),
        pl.Series("median_dist", median_distances),
        pl.Series("std_dist", std_distances),
        pl.Series("group_size", group_size),
        pl.Series("n_species", n_species))

    # Convert each list in species_dict into a column
    for species_name, in_group_list in species_dict.items():
        watlas_df = watlas_df.with_columns(
            pl.Series(species_name, in_group_list)
        )

    return watlas_df


def main():
    # time process
    start = timeit.default_timer()

    sqlite_path = "data/SQLite/watlas-2023.sqlite"
    watlas_df = WatlasDataframe()

    watlas_df.process_for_prediction(start_time="2023-08-01 00:00:00", end_time="2023-08-21 00:00:00",
                                     sqlite_path="data/SQLite/watlas-2023.sqlite",
                                     tag_file="data/watlas_data/tags_watlas_all.xlsx")
    # watlas_df.get_watlas_data_sqlite([3001, 3002, 3016],
    #                                  tracking_time_start="2023-08-01 00:00:00",
    #                                  tracking_time_end="2023-08-21 00:00:00", sqlite_path=sqlite_path)
    #
    # print("watlas data found ")
    # print(watlas_df.get_watlas_dataframe())
    #
    # watlas_df.filter_num_localisations()
    # print("filtered data ")
    # # print(watlas_df.get_watlas_dataframe())
    #
    # watlas_df.aggregate_dataframe()
    # print("aggregated data ")
    # # print(watlas_df.get_watlas_dataframe())
    #
    # watlas_df.get_tag_data("data/watlas_data/tags_watlas_all.xlsx")
    #
    # print("species")
    # watlas_df.get_species()
    # # print(watlas_df.get_watlas_dataframe())
    #
    # watlas_df = watlas_df.get_watlas_dataframe()
    #
    # df_list = []
    # for tag, wat_df in watlas_df.group_by("tag"):
    #     wat_df = smooth_data(wat_df)
    #     wat_df = get_speed(wat_df)
    #     wat_df = get_turn_angle(wat_df)
    #
    #     df_list.append(wat_df)
    #
    # watlas_df = pl.concat(df_list)
    # watlas_df.write_csv("watlas_all.csv")
    # print(watlas_df)
    # print(watlas_df.columns)

    stop = timeit.default_timer()
    print('Time: ', stop - start)


if __name__ == '__main__':
    sys.exit(main())

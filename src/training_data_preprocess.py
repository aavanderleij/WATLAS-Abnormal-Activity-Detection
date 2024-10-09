"""
preprocess the training data
"""
import sys

import polars as pl
from pytools4watlas import WatlasDataframe, get_speed, get_turn_angle, get_simple_travel_distance


class WatlasTrainingDataframe(WatlasDataframe):

    def __init__(self, path_train_csv, tag_file_path):

        super().__init__(watlas_df=self.get_training_data(path_train_csv=path_train_csv))
        # change for readability
        self.training_df = self.watlas_df
        # get tag data
        self.get_tag_data(tag_csv_path=tag_file_path)

    def get_training_data(self, path_train_csv):
        """
        Read file with training data.
        Training data should contain tag numbers, localization times and labels

        Args:
            path_train_csv (str): path to training csv:

        Returns:
            pl.dataframe: training dataframe

        """
        # TODO add file checks
        training_df = pl.read_csv(path_train_csv)

        return training_df

    def get_group_distance(self, dataframe, group_area):
        """
        Calculate distance between individuals. Return the mean, median, std deviation of all individuals within
        group_area

        Args:
            dataframe (pl.dataframe): training dataframe
            group_area (int): Maximum distance between individuals to be considered part of same group in meters

        Returns:
            dataframe (pl.dataframe): training dataframe with new columns "mean_dist", "median_dist", "std_dist"

        """

        mean_distances = []
        median_distances = []
        std_distances = []

        # get x and y columns from dataframe
        x_coords = dataframe["X"]
        y_coords = dataframe["Y"]

        for i in range(dataframe.shape[0]):
            # get current row coords
            x1, y1 = dataframe[i, "X"], dataframe[i, "Y"]

            # get distances of current row
            distances = ((x_coords - x1) ** 2 + (y_coords - y1) ** 2).sqrt()

            # if there are more one member of group
            if distances.shape[0] != 1:
                # remove dist to self
                distances = distances.filter(distances != 0)
                # remove dist smaller than 500
                distances = distances.filter(distances <= group_area)

                # get mean, median and standard deviation
                mean = distances.mean()
                median_dist = distances.median()
                std_dist = distances.std()
            else:
                # if there are no members of group, set all to 0
                mean = 0
                median_dist = 0
                std_dist = 0

            mean_distances.append(mean)
            median_distances.append(median_dist)
            std_distances.append(std_dist)

        # save values as columns
        dataframe = dataframe.with_columns(
            pl.Series("mean_dist", mean_distances),
            pl.Series("median_dist", mean_distances),
            pl.Series("std_dist", std_distances))

        return dataframe

    def prepare_train_df(self):
        # TODO get group data from previus colums too
        # TODO drop test tags
        # TODO check where self should be used
        df_list = []
        column_names = ["speed", "distance", "turn_angle", "alert", "large_disturb", "small_disturb"]
        species_names = ["sanderling", "curlew", "spoonbill", "dunlin", "grey plover", "turnstone", "redshank", "avocet"
                         "oystercatcher", "common tern", "bar-tailed godwit"]

        # get species
        # TODO traning df does not cotain species. watlas_df does
        self.get_species()

        # group data frame by time stamp
        for time, data in self.watlas_df.group_by("time"):
            # add group size
            data = data.with_columns(pl.len().alias("group_size_per_timestamp"))
            # add number of unique species
            n_unique_species = data["species"].n_unique()
            data = data.with_columns(pl.lit(n_unique_species).alias("n_unique_species"))

            # get distance
            data = self.get_group_distance(data, 500)

            #
            data = data.with_columns([
                pl.col("mean_dist").cast(pl.Float64),
                pl.col("median_dist").cast(pl.Float64),
                pl.col("std_dist").cast(pl.Float64)
            ])

            df_list.append(data)

        # concat dataframes into single dataframe
        self.training_df = pl.concat(df_list)

        # TODO should be own function
        # clean list
        df_list = []
        # group by tag so every individual is treated as its own entity
        for tag, data in self.training_df.group_by("tag"):
            # get dist
            # get speed
            data = get_speed(data)
            # get angle
            data = get_turn_angle(data)
            # add last 4 time
            data = data.with_columns(
                [pl.col(col_name).shift(1).alias(f"{col_name}_1") for col_name in column_names] +
                [pl.col(col_name).shift(2).alias(f"{col_name}_2") for col_name in column_names] +
                [pl.col(col_name).shift(3).alias(f"{col_name}_3") for col_name in column_names] +
                [pl.col(col_name).shift(4).alias(f"{col_name}_4") for col_name in column_names]
            )
            # fil
            df_list.append(data)

        training_df = pl.concat(df_list)

        # print(training_df.columns)
        # drop exact coords
        training_df = training_df.drop("X", "Y")
        # # sort by time
        training_df = training_df.sort(by="time_agg")
        # set NaN and NULLs to 0
        training_df = training_df.fill_nan(0).fill_null(0)

        print(training_df.columns)

        return training_df




def main():
    #
    train_wat_df = WatlasTrainingDataframe("data/training_data/train_data_frame.csv",
                                           "data/watlas_data/tags_watlas_all.xlsx")
    print(train_wat_df.get_watlas_dataframe())

    # training_df = get_train_df("data/training_data/train_data_frame.csv")
    preped_data = (train_wat_df.prepare_train_df())
    # preped_data.write_csv("data/training_data/processed_train_data.csv", separator=",")
    # print(preped_data.head())
    # print(preped_data.columns)


if __name__ == '__main__':
    sys.exit(main())

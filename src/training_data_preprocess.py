"""
preprocess the training data
"""
import sys

import polars as pl
import pytools4watlas


def get_train_df(path_train_csv):
    training_df = pl.read_csv(path_train_csv)

    return training_df


def calculate_simple_dist(x1, y1, x2, y2):
    distance = ((x1 - x2) ** 2 +
                (y1 - y2) ** 2) ** 0.5  # multiply by power of 0.5 to get square root, to get euclidian distance


def get_group_distance(df):
    avg_distances = []
    std_distances = []

    # get x and y columns from dataframe
    x_coords = df["X"]
    y_coords = df["Y"]

    for i in range(df.shape[0]):
        # get current row coords
        x1, y1 = df[i, "X"], df[i, "Y"]

        distances = ((x_coords - x1) ** 2 + (y_coords - y1) ** 2).sqrt()

        # if there are more one member of group
        if distances.shape[0] != 1:
            # remove dist to self
            distances = distances.filter(distances != 0)

            avg_dist = distances.mean()
            std_dist = distances.std()
        else:
            avg_dist = 0
            std_dist = 0

        avg_distances.append(avg_dist)
        std_distances.append(std_dist)

    df = df.with_columns(
        pl.Series("avg_dist", avg_distances),
        pl.Series("std_dist", std_distances))

    return df


def prepare_train_df(training_df):
    df_list = []
    column_names = ["speed", "distance", "turn_angle", "alert", "large_disturb", "small_disturb"]

    # get species
    tag_df = pytools4watlas.get_tag_data("data/watlas_data/tags_watlas_all.xlsx")
    training_df = pytools4watlas.get_species(tag_df, training_df)

    # get info of surrounding birds
    for time, data in training_df.group_by("time"):
        data = data.with_columns(pl.len().alias("group_size_per_timestamp"))
        n_unique_species = data["species"].n_unique()
        data = data.with_columns(pl.lit(n_unique_species).alias("n_unique_species"))
        data = get_group_distance(data)

        data = data.with_columns([
            pl.col("avg_dist").cast(pl.Float64),
            pl.col("std_dist").cast(pl.Float64)
        ])

        df_list.append(data)

    training_df = pl.concat(df_list)

    # clean list
    df_list = []
    # group by tag so every individual is treated as its own entity
    for tag, data in training_df.group_by("tag"):
        # get dist
        data = pytools4watlas.get_simple_distance(data)
        # get speed
        data = pytools4watlas.get_speed(data)
        # get angle
        data = pytools4watlas.get_turn_angle(data)
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

    return training_df


def main():
    training_df = get_train_df("data/training_data/train_data_frame.csv")
    preped_data = (prepare_train_df(training_df))
    preped_data.write_csv("data/training_data/processed_train_data.csv", separator=",")
    print(preped_data.head())
    print(preped_data.columns)


if __name__ == '__main__':
    sys.exit(main())

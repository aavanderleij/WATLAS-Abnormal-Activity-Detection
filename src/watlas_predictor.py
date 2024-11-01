"""
load data and model and write results to csv file
"""
import sys

import tensorflow as tf
import pandas as pd

class WatlasPredictor:

    def __init__(self):
        self.columns_to_drop = ['TAG', 'time', 'X', 'Y', 'TIME', 'tag', 'X_raw', 'Y_raw']

        loaded_model = tf.keras.models.load_model("models/test_model")

        train_data_df = pd.read_csv("watlas_all.csv")
        # drop non parameters
        train_data_df = train_data_df.drop(columns=self.columns_to_drop)
        # make dataframe ready for being loaded into model
        df = {key: value.to_numpy()[:, tf.newaxis] for key, value in train_data_df.items()}
        # predict
        results = loaded_model.predict(df)

        print(results)

        #save results in csv

        # clean up output


def main():
    WatlasPredictor()

if __name__ == "__main__":
    sys.exit(main())




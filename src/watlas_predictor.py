"""
load data and model and write results to csv file
"""
import sys
from pathlib import Path
import tensorflow as tf
import pandas as pd
import configparser


class WatlasPredictor:

    def __init__(self, config_file):

        try:
            self.config = configparser.ConfigParser()
            self.config.read(config_file)
        except configparser.Error as e:
            print(sys.exit(f"could not open config file: {config_file}"))

        start_time = self.config['timeframe']['start_time']

        result_dir = Path(self.config['output']['output directory'])
        # add start time folder to prevent accidental overwrite
        result_dir = result_dir / str(start_time).replace(":", "-").replace(" ", "_")

        self.columns_to_drop = ['TAG', 'time', 'X', 'Y', 'TIME', 'tag', 'X_raw', 'Y_raw']

        loaded_model = tf.keras.models.load_model("models/test_model")

        train_data_df = pd.read_csv(Path(result_dir) / "watlas_preprediction.csv")
        # drop non parameters
        train_data_df = train_data_df.drop(columns=self.columns_to_drop)
        # make dataframe ready for being loaded into model
        df = {key: value.to_numpy()[:, tf.newaxis] for key, value in train_data_df.items()}
        # predict
        results = loaded_model.predict(df)

        results_dataframe = train_data_df.assign(predicted=results)

        results_dataframe.to_csv(Path(result_dir) / "predictions.csv")




def main():
    config_file = sys.argv[1]
    WatlasPredictor(config_file)


if __name__ == "__main__":
    sys.exit(main())

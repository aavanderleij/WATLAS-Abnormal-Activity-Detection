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

        # get start time from config dir
        self.start_time = self.config['timeframe']['start_time']
        # get output dir from config file
        self.result_dir = Path(self.config['output']['output directory'])
        
        loaded_model = tf.keras.models.load_model("models/disturbance_model")

        input_df = pd.read_csv(Path(self.result_dir) / "watlas_preprediction.csv")

        dataframe_with_predictions = self.run_model(input_df, loaded_model)

        self.write_results(dataframe_with_predictions)



    
    def run_model(self, input_df, loaded_model):
        """
        runs loaded model in input dataframe
        Returns dataframe with predictions

        Returns (pd.DataFrame):
            dataframe with predictions

        """


        # drop non parameters
        predict_data_df = input_df.drop(columns=self.columns_to_drop)
        # make dataframe ready for being loaded into model, with column name being parameter name and value as tensor
        input_dict = {key: value.to_numpy()[:, tf.newaxis] for key, value in predict_data_df.items()}
        # predict
        results = loaded_model.predict(input_dict)



        results_dataframe = input_df.assign(predicted=results)
        
        return results_dataframe

    def write_results(self, results_dataframe):
        """
        Write WATLAS data and prediction results to a csv file
        
        Args:
            results_dataframe: 

        Returns:

        """

        

        # add start time folder to prevent accidental overwrite
        result_dir = self.result_dir / str(self.start_time).replace(":", "-").replace(" ", "_")

        results_dataframe.to_csv(Path(result_dir) / "predictions.csv")



def main():
    config_file = sys.argv[1]
    WatlasPredictor(config_file)


if __name__ == "__main__":
    sys.exit(main())

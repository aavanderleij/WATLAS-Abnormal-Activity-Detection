"""
Wrapper for preprosessing and model
detects disturbance in watlas data given a time
"""
import sys
import configparser
import pytools4watlas
import subprocess




class DisturbanceDetector:

    def __init__(self, config_file="config/config.ini"):


        try:
            self.config = configparser.ConfigParser()
            self.config.read(config_file)
        except configparser.Error as e:
            print(sys.exit(f"could not open config file: {config_file}"))

        self.preprocess_watlas_data(config_file)


    def preprocess_watlas_data(self, config_file):

        watlas_tool = pytools4watlas.WatlasDataframe(config_file=config_file)
        watlas_tool.process_for_prediction()


    # load model
    def load_model(self):
        ...

        # something subprocces

        # run model

        # process results

        # profit

def main():

    DisturbanceDetector(config_file="config/config.ini")



if __name__ == "__main__":
    sys.exit(main())
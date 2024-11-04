"""
Wrapper for detecting distrubances based on settings in the configuration file.
Detects disturbance in watlas data given a time
"""

import sys
import argparse
import configparser
import subprocess
from pathlib import Path

from src import pytools4watlas
import polars as pl


class DisturbanceDetector:

    def __init__(self, config_file):

        try:
            self.config = configparser.ConfigParser()
            self.config.read(config_file)
        except configparser.Error as e:
            print(sys.exit(f"could not open config file: {config_file}"))

        self.preprocess_watlas_data(config_file)

        self.run_model(config_file)

        self.process_output()

    def preprocess_watlas_data(self, config_file):

        watlas_tool = pytools4watlas.WatlasDataframe(config_file=config_file)
        watlas_tool.process_for_prediction()

    # load model
    def run_model(self, config_file):
        subprocess.run(["conda", "--version"], shell=True)

        subprocess.run(['conda', 'run', '-n', 'tensorflow', 'python', 'src/watlas_predictor.py', config_file],
                       shell=True)

        # process results

    def process_output(self):
        """

        Returns:

        """
        start_time = self.config['timeframe']['start_time']

        result_dir = Path(self.config['output']['output directory'])
        # add start time folder to prevent accidental overwrite
        result_dir = result_dir / str(start_time).replace(":", "-").replace(" ", "_")

        output_df = pl.read_csv(result_dir / "predictions.csv")

        output_df = output_df.select(
            ['TAG', 'time', 'X', 'Y', 'NBS', 'TIME', 'tag', 'VARX', 'VARY', 'species', 'distance', 'speed_in',
             'speed_out', 'turn_angle', 'mean_dist_group', 'median_dist_group', 'std_dist_group', 'group_size',
             'n_species_group', 'mean_turn_angle_group', 'median_turn_angle_group', 'std_turn_angle_group',
             'mean_speed_group', 'speed_median_group', 'speed_std_group', 'islandica_in_group',
             'oystercatcher_in_group', 'spoonbill_in_group', 'bar-tailed_godwit_in_group', 'redshank_in_group',
             'sanderling_in_group', 'dunlin_in_group', 'turnstone_in_group', 'curlew_in_group', 'gray_plover_in_group',
             'waterlevel', 'predicted'])

        output_df = output_df.with_columns(
            (pl.when(pl.col('predicted') >= 0.5).then(1).otherwise(0).alias('Alert')
             ))

        output_df.write_csv(result_dir / "predictions.csv")


def prarse_args():
    parser = argparse.ArgumentParser(
        prog='watlas_disturbance_detect',
        description='Detects disturbances in birds using WATLAS tracking data',
        epilog='For an example of the config file: config/config.ini')

    parser.add_argument("config_file", help="path to the config file")

    args = parser.parse_args()

    config_file = args.config_file

    return config_file


def main():
    DisturbanceDetector(config_file=prarse_args())


if __name__ == "__main__":
    sys.exit(main())

"""
Functions for creating and switching conda environments.

"""
import subprocess
import sys


def create_conda_env(env_name, requirements_file):
    """
    activates a conda env and if it doesn't exist, creates it
    :param env_name (str): name of the conda environment:
    :param requirements_file:
    :return:
    """
    # TODO check if still needed

    try:
        subprocess.run(f"conda create --name {env_name} --file {requirements_file}", shell=True, check=True)
        print(f"Conda environment {env_name} created.")
    except subprocess.CalledProcessError as e:
        print(f"could not create conda environment {env_name}. Error: {e.output}")
        sys.exit(1)


def start_conda_env(env_name):
    try:
        subprocess.run(f"conda activate {env_name} ", shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"could not activate conda environment {env_name}. Error: {e.output}")
        sys.exit(1)


def stop_conda_env():
    try:
        subprocess.run(f"conda deactivate ", shell=True, check=True)
    except subprocess.CalledProcessError as e:
        # TODO create if not exist
        pass

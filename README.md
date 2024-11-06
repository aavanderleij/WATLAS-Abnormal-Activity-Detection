# model-watlas-disturbance
Bachelor thesis of Antsje van der Leij

This is a program for detecting disturbances in birds using WATLAS.

## Setup

This library uses conda. 
If you don't have conda, please follow this link to install conda
https://docs.anaconda.com/miniconda/

to install the nessesery enviroments run the following code from the location of this readme:

```
conda env create -f env/watlas_disturbance.yaml
conda env create -f env/tensorflow.yaml
```

## usage

This script uses a config file to retrieve settings.
An example file can be found at config/config.ini
Please edit all necessary values before running.

Afterward activate the conda enviroment

```
conda activate watlas_disturbance
```

Then run the following to start the program
``
python -m scripts.watlas_disturbance_detect config.ini
``
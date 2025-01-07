# WATLAS Abnormal Activity Detection
Bachelor project of Antsje van der Leij

This is a program for detecting abnormal behavior patterns in shore birds using WATLAS.

The WATLAS system tracks shorebird movements in the Dutch Wadden Sea. This project uses WATLAS data and machine learning
to detect abnormal bird activity patterns indicative of disturbances. 

A supervised deep-learning model was developed using TensorFlow and Keras, to identify these behaviors.
Data preprocessing was handled with the Python Polars library. The model, designed to classify behavior as normal or
abnormal, achieved an overall accuracy of 99.7%, with strong recall (82%), moderate precision (46%), and a F1 score of
0.59. This pipeline presents proof-of-concept for analyzing disturbances and supporting conservation efforts in the
Wadden Sea.


## Setup

This library uses conda. 
If you don't have conda, please follow this link to install conda
https://docs.anaconda.com/miniconda/

To install the nessesery enviroments run the following code from the location of this readme:

```
conda env create -f env/watlas_disturbance.yaml
conda env create -f env/tensorflow.yaml
```

## Usage

This script uses a config file to retrieve settings.

An example file with instructions can be found at config/config.ini. 
Please edit all necessary values before running.

To run the pipeline first activate the conda environment

```
conda activate watlas_disturbance
```

Then run the following to start the program
```
python -m scripts.watlas_disturbance_detect config/config.ini
```

## Refrences

More information on the WATLAS tracking system please reference the following paper:


Bijleveld, A.I., van Maarseveen, F., Denissen, B. et al. WATLAS: high-throughput and real-time tracking of many small birds in the Dutch Wadden Sea. Anim Biotelemetry 10, 36 (2022). https://doi.org/10.1186/s40317-022-00307-w

## Contact

For access and questions about the WATLAS data please contact Allert Bijleveld at Allert.Bijleveld@nioz.nl

For questions regarding the code please contact me at antsjevdleij@gmail.com
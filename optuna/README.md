# Gretel Synthetics Tuning With Optuna

* This notebook will let you tune the Gretel synthetic model hyperparameters of several datasets at once.
* It is also setup to run multiple Optuna trials at once using an SQLite database (prepackaged with most operating systems).
* This notebook makes use of our python module Optuna_Trials.py. In most cases, you won't need to modify this module. It is configured with all the relelvant synyhetic model hyperparameters and their relevant ranges. If you'd like to change which parameters are tuned or the range of values to tune over, then you will need to modify that module.
* This notebook works seemlessly on Linux and Ubuntu, but not on a Mac.


# Requirements

In order to run this code, make sure to install the requirements in `requirements.txt`:

`pip install -r requirements.txt`

It is advised to first make a virtual environment before installing these requirements.


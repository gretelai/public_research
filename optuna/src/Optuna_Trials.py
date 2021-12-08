import sys
import time

from smart_open import open
import yaml
import pandas as pd
import optuna
from optuna.samplers import TPESampler

from gretel_client import configure_session, ClientConfig
from gretel_client import projects
from gretel_client import create_project
from gretel_client.config import RunnerMode
from gretel_client.helpers import poll

# Read in the study name, trial cnt, dataset, api_key and storage

study_name = sys.argv[1]
trial_cnt = int(sys.argv[2])
dataset = sys.argv[3]
api_key = sys.argv[4]
storage = sys.argv[5]

# Connect to Gretel with your API key

configure_session(ClientConfig(api_key=api_key,
                               endpoint="https://api.gretel.cloud"))

# Grab the default Synthetic Config file:

with open("https://raw.githubusercontent.com/gretelai/gretel-blueprints/main/config_templates/gretel/synthetics/default.yml", 'r') as stream:
    config = yaml.safe_load(stream)

# Turning off the privacy filtering will speed up the tuning, but could impact the true
# accuracy of the results if you plan to eventually run with them on.

config['models'][0]['synthetics']['privacy_filters']['outliers'] = None
config['models'][0]['synthetics']['privacy_filters']['similarity'] = None  

# This objective function is what Optuna calls each time/trial.  It picks the next set of params
# to use, creates the model and then returns the SQS value.  When I create the study, I'll tell it I want
# to maximize the result of my objective function

def objective(trial: optuna.Trial):
 
    # Set which params you want to tune

    config['models'][0]['synthetics']['params']['rnn_units'] = trial.suggest_int(name="rnn_units", low=64, high=1024, step=64)
    config['models'][0]['synthetics']['params']['dropout_rate'] = trial.suggest_float("dropout_rate", .1, .75)
    config['models'][0]['synthetics']['params']['gen_temp'] = trial.suggest_float("gen_temp", .8, 1.2)
    config['models'][0]['synthetics']['params']['learning_rate'] = trial.suggest_float("learning_rate",  .0005, 0.01, step=.0005)
    config['models'][0]['synthetics']['params']['reset_states'] = trial.suggest_categorical(
        "reset_states", choices=[True, False])
    config['models'][0]['synthetics']['params']['vocab_size'] = trial.suggest_categorical(
        "vocab_size", choices=[0, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000])

    # Create a new project
    seconds = int(time.time())
    project_name = "Tuning Experiment" + str(seconds)
    project = create_project(display_name=project_name)
    
    # Create a model 
    
    model = project.create_model_obj(model_config=config)
    model.data_source = dataset
    model.submit(upload_data_source=True)
        
    # Watch for completion
                
    status = "active"
    sqs = 0

    while ((status == "active") or (status == "pending")):
        #Sleep a bit here
        time.sleep(60)
        model._poll_job_endpoint()
        status = model.__dict__['_data']['model']['status']
        if status == "completed":
            report = model.peek_report()
            if report:
                sqs = report['synthetic_data_quality_score']['score']
            else:
                sqs = 0
        elif status == "error":
            sqs = 0
           
    project.delete()
    return sqs

# Load the Optuna study
study=optuna.load_study(study_name=study_name,storage=storage)

# Run trial_cnt number of trials
study.optimize(objective,n_trials=trial_cnt)
    

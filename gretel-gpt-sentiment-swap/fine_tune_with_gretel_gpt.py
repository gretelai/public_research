import getpass
import os
from argparse import ArgumentParser

import gretel_client as gretel
import pandas as pd

PROJECT_NAME = "gretel-gpt-sentiment-swap"

parser = ArgumentParser()
parser.add_argument("--data-subset", type=str, default="Video_Games_v1_00")
args = parser.parse_args()

api_key = os.getenv("GRETEL_API_DEV_KEY") or getpass.getpass("Enter your Gretel API key: ")

gretel.configure_session(
    api_key=api_key,
    endpoint="https://api-dev.gretel.cloud",
    validate=True,
    clear=True,
)

project = gretel.projects.create_or_get_unique_project(name=PROJECT_NAME)

config = {
    "schema_version": 1,
    "models": [
        {
            "gpt_x": {
                "data_source": "__",
                "pretrained_model": "gretelai/mpt-7b",
                "batch_size": 4,
                "epochs": 4,
                "weight_decay": 0.01,
                "warmup_steps": 100,
                "lr_scheduler": "linear",
                "learning_rate": 0.0002,
                "validation": None,
                "generate": {"num_records": 100, "maximum_text_length": 500},
            }
        }
    ],
}
model = project.create_model_obj(model_config=config)
model.data_source = pd.read_csv(f"data/training_product_review_pairs_{args.data_subset}.csv.gz")
model.name = f"{PROJECT_NAME}_{args.data_subset}"
model.submit_cloud()

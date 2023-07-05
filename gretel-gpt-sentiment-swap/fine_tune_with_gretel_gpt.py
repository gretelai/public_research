import getpass
import os
from argparse import ArgumentParser

import gretel_client as gretel
import pandas as pd


parser = ArgumentParser()
parser.add_argument("--project-name", type=str, default="gretel-gpt-sentiment-swap", help="Name of Gretel project.")
parser.add_argument("--data-subset", type=str, default="Video_Games_v1_00", help="Subset of Amazon dataset.")
args = parser.parse_args()

api_key = os.getenv("GRETEL_API_KEY") or getpass.getpass("Enter your Gretel API key: ")

gretel.configure_session(
    api_key=api_key,
    endpoint="https://api.gretel.cloud",
    validate=True,
    clear=True,
)

print(f"Creating or fetching Gretel project with name {args.project_name}")
project = gretel.projects.get_project(name=args.project_name, display_name=args.project_name, create=True)

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
                "learning_rate": 0.0005,
                "validation": None,
                "generate": {"num_records": 100, "maximum_text_length": 500},
            }
        }
    ],
}

print("Creating model object")
model = project.create_model_obj(model_config=config)
model.data_source = pd.read_csv(f"data/training_review_pairs_{args.data_subset}.csv.gz")
model.name = f"{args.project_name}_{args.data_subset}"

print(f"Submitting fine-tuning job to Gretel Cloud with data subset {args.data_subset}")
model.submit_cloud()

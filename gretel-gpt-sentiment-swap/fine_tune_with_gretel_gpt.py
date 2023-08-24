import getpass
import os
from argparse import ArgumentParser

import gretel_client as gretel
import pandas as pd

# Parse command-line arguments
parser = ArgumentParser()
parser.add_argument("--project-name", type=str, default="gretel-gpt-sentiment-swap", help="Name of Gretel project.")
parser.add_argument("--data-subset", type=str, default="Video_Games_v1_00", help="Subset of Amazon dataset.")
parser.add_argument(
    "--pair-metric",
    default="helpful_votes",
    choices=["helpful_votes", "cos_sim"],
    help="Metric used for selecting pairs.",
)
args = parser.parse_args()

# Configure Gretel session
gretel.configure_session(
    api_key=os.getenv("GRETEL_API_KEY") or getpass.getpass("Enter your Gretel API key: "),
    endpoint="https://api.gretel.cloud",
    validate=True,
    clear=True,
)

# Create or fetch a Gretel project
print(f"Creating or fetching Gretel project with name {args.project_name}")
project = gretel.projects.get_project(name=args.project_name, display_name=args.project_name, create=True)

# Configure Gretel model and submit fine-tuning job to Gretel Cloud
config = {
    "schema_version": 1,
    "models": [
        {
            "gpt_x": {
                "data_source": "__",
                "pretrained_model": "gretelai/mpt-7b",
                "batch_size": 16,
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
file_label = f"{args.data_subset}_{args.pair_metric}"
model = project.create_model_obj(model_config=config)
model.data_source = pd.read_csv(f"data/training_review_pairs-{file_label}.csv.gz")
model.name = f"{args.project_name}-{file_label}"

print(f"Submitting fine-tuning job to Gretel Cloud with data subset {file_label}")
model.submit_cloud()

print("🚀 Job submitted! You can monitor its progress in the Gretel Console: https://console.gretel.ai/projects")

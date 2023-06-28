import os
from argparse import ArgumentParser

import gretel_client as gretel
import pandas as pd

PROJECT_NAME = "gretel-gpt-sentiment-swap"

parser = ArgumentParser()
parser.add_argument("--data-subset", type=str, default="Apparel_v1_00")
args = parser.parse_args()


gretel.configure_session(
    api_key=os.getenv("GRETEL_API_DEV_KEY"),
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
                "epochs": 3,
                "weight_decay": 0.02,
                "warmup_steps": 100,
                "lr_scheduler": "linear",
                "learning_rate": 0.001,
                "validation": None,
            }
        }
    ],
}
model = project.create_model_obj(model_config=config)
model.data_source = pd.read_csv(f"data/training_product_review_pairs_{args.data_subset}.csv")
model.name = f"{PROJECT_NAME}_{args.data_subset}"
model.submit_cloud()

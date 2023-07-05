import getpass
import os
from argparse import ArgumentParser

import gretel_client as gretel
import pandas as pd


parser = ArgumentParser()
parser.add_argument("--project-name", type=str, default="gretel-gpt-sentiment-swap", help="Name of Gretel project.")
parser.add_argument("--data-subset", type=str, default="Video_Games_v1_00", help="Subset of Amazon dataset.")
parser.add_argument(
    "--temperature",
    type=float,
    default=1.2,
    help="The value used to module the next token probabilities. "
    "Higher temperatures lead to more randomness in the output.",
)
args = parser.parse_args()

api_key = os.getenv("GRETEL_API_KEY") or getpass.getpass("Enter your Gretel API key: ")

gretel.configure_session(
    api_key=api_key,
    endpoint="https://api.gretel.cloud",
    validate=True,
    clear=True,
)

print(f"Fetching Gretel project with name {args.project_name}")
project = gretel.projects.get_project(name=args.project_name, display_name=args.project_name)

print(f"Finding the most recent model for the {args.data_subset} data subset")
model_list = [m for m in project.search_models(model_name=args.data_subset) if m.status == "completed"]
assert len(model_list) > 0, f"No models found for {args.data_subset}"
model = model_list[-1]

print("Submitting generation job to Gretel Cloud")
record_handler = model.create_record_handler_obj(
    params={"maximum_text_length": 200, "temperature": args.temperature},
    data_source=pd.read_csv(f"data/conditional_prompts_{args.data_subset}.csv.gz"),
)
record_handler.submit_cloud()
gretel.helpers.poll(record_handler)

df_generations = pd.read_csv(record_handler.get_artifact_link("data"), compression="gzip")
df_generations.to_csv(f"model-generations/generations-{args.data_subset}.csv.gz", index=False)
df_prompts = pd.read_csv(f"data/conditional_prompts_{args.data_subset}.csv.gz")

print(f"Saving review pairs to model-generations/review-pairs-{args.data_subset}.txt")
with open(f"model-generations/review-pairs-{args.data_subset}.txt", "w") as out_file:
    for idx, prompt in df_prompts.itertuples():
        generation = df_generations.loc[idx, "text"]
        out_file.write(f"{prompt} {generation}\n\n-----\n")

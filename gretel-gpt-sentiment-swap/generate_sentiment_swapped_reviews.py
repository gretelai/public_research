import getpass
import os
from argparse import ArgumentParser

import gretel_client as gretel
import pandas as pd

# Parse command-line arguments
parser = ArgumentParser()
parser.add_argument("--project-name", type=str, default="gretel-gpt-sentiment-swap", help="Name of Gretel project.")
parser.add_argument("--data-subset", type=str, default="Video_Games_v1_00", help="Subset of Amazon dataset.")
parser.add_argument("--model-id", type=str, default=None, help="Gretel model UID. If None, use the most recent model.")
parser.add_argument(
    "--pair-metric",
    default="helpful_votes",
    choices=["helpful_votes", "cos_sim"],
    help="Metric used for selecting pairs.",
)
parser.add_argument(
    "--temperature",
    type=float,
    default=1.2,
    help="Hyperparameter that regulates the randomness of the LLM outputs. Higher values lead to more randomness.",
)
args = parser.parse_args()

# Configure Gretel session
gretel.configure_session(
    api_key=os.getenv("GRETEL_API_KEY") or getpass.getpass("Enter your Gretel API key: "),
    endpoint="https://api.gretel.cloud",
    validate=True,
    clear=True,
)

# Fetch Gretel project and model
print(f"Fetching Gretel project with name {args.project_name}")
project = gretel.projects.get_project(name=args.project_name, display_name=args.project_name)

if args.model_id is not None:
    print(f"Fetching model with UID {args.model_id}")
    model = gretel.projects.models.Model(project=project, model_id=args.model_id)
else:
    print(f"Finding the most recent model for the {args.data_subset}-{args.pair_metric} data subset")
    model_list = [m for m in project.search_models(model_name=args.data_subset) if m.status == "completed"]
    assert len(model_list) > 0, f"No models found for {args.data_subset}"
    model = model_list[-1]

# Submit job to Gretel Cloud to generate sentiment-swapped reviews
print("Submitting generation job to Gretel Cloud")
record_handler = model.create_record_handler_obj(
    params={"maximum_text_length": 200, "temperature": args.temperature},
    data_source=pd.read_csv(f"data/conditional_prompts-{args.data_subset}.csv.gz"),
)
record_handler.submit_cloud()
gretel.helpers.poll(record_handler)

# Save generated reviews and write complete review pairs to a text file
file_label = f"{args.data_subset}_{args.pair_metric}"
print(f"Saving review pairs to model-generations/review_pairs-{file_label}.txt")
df_generations = pd.read_csv(record_handler.get_artifact_link("data"), compression="gzip")
df_generations.to_csv(f"model-generations/generations-{file_label}.csv.gz", index=False)
with open(f"model-generations/review_pairs_{file_label}.txt", "w") as out_file:
    df_prompts = pd.read_csv(f"data/conditional_prompts-{args.data_subset}.csv.gz")
    for idx, prompt in df_prompts.itertuples():
        generation = df_generations.loc[idx, "text"]
        out_file.write(f"{prompt} {generation}\n\n-----\n")

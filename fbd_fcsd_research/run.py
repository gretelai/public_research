import numpy as np
import pandas as pd
import json
from collections import defaultdict
from scipy.stats import spearmanr, pearsonr
from text_metrics.metrics import *
from typing import List, Tuple, Dict


def grab_data(path: str) -> List:
    """
    Purpose: Loads in the data from the desired path.

    Args:
        path: path to file.

    Outputs:
        data: list of dictionaries containing the data.
    """
    with open(path, "r") as f:
        data = []
        for line in f:
            data.append(json.loads(line))
    return data


def average_human_scores(data: List) -> List:
    """
    Purpose: Based on the experiments done in (https://arxiv.org/pdf/2105.02573.pdf), we need to average each human rating of synthetic text examples, so we can compare
    them to the FBD score accurately. This function takes the human scores (if multiple scores are provided, we take the last score as that should relate to overall score
    instead of subscores, such as grammar or semantics) and averages them into a single score.

    Args:
        data: The data loaded by the grab_data function.

    Outputs:
        human_scores: list of averaged human scores.
    """
    num_scores = len(data[0]["human_scores"])
    human_scores = []
    raw_scores = defaultdict(list)
    for line in data:
        for i in range(num_scores):
            raw_scores[i].append(line["human_scores"][i][-1])

    for i in raw_scores.keys():
        human_scores.append(sum(raw_scores[i]) / len(raw_scores[i]))

    return human_scores


def collate_data(data: list) -> Tuple[List, List]:
    """
    Purpose: In the experimentations done in (https://arxiv.org/pdf/2105.02573.pdf), we need to concatenate the queries and responses in order to make the vectors for FBD.
    This function serves that exact purpose of concatenating the real query with the synthetic responses and real responses respectively.

    Args:
        data: The data loaded by grab_data function.

    Outputs:
        real_inputs: A list of the real query + real response combos.
        synth_inputs: A list of the real query + synthetic response combos.
    """
    num_synth = len(data[0]["hyps"])
    real_queries = []
    synth_answers = defaultdict(list)
    real_answers = []
    for line in data:
        real_queries.append(line["src"])
        real_answers.append(line["refs"])
        for i in range(num_synth):
            synth_answers[i].append(line["hyps"][i])

    real_inputs = []
    synth_inputs = defaultdict(list)
    for real_query, real_answer in zip(real_queries, real_answers):
        real_inputs.append(real_query + " " + real_answer[0])

    for i in range(num_synth):
        for real_query, synth_answer in zip(real_queries, synth_answers[i]):
            synth_inputs[i].append(real_query + " " + synth_answer)

    return real_inputs, synth_inputs


def metric_collater(
    real_data: list, synth_data: list, model_name: str
) -> Tuple[List, List]:
    """
    Purpose: This function runs the fbd and fcsd scores between the real and synthetic text inputs.

    Args:
        real_data: A list of the real inputs.
        synth_data: A list of the synthetic inputs.
        model_name: The name of the model for the embeddings. This will be utilized by SentenceTransformer

    Outputs:
        fbd_scores: list of fbd_scores.
        fcsd_scores: list of fcsd_scores.
    """
    fcsd_scores = []
    fbd_scores = []
    for key in synth_data.keys():
        real_series = pd.DataFrame(real_data, columns=["Text"])
        synth_series = pd.DataFrame(synth_data[key], columns=["Text"])
        fbd_score, fcsd_score = metrics_run(
            real_series.Text, synth_series.Text, model_name
        )
        fbd_scores.append(fbd_score)
        fcsd_scores.append(fcsd_score)

    return fbd_scores, fcsd_scores


def experiments(
    model_names: List, data_names: List, index_names: List, all_data: List
) -> Dict:
    """
    Purpose: Runs the full experiment. Takes all the model names you want to try from Sentence Transformer, all the data that has been loaded in, the metric names, and a
    list of all the data to create the scores. This function is quite rigid, but you can use this code as a template to use for other metrics and data.

    Args:
        model_names: These are names that come from the Sentence Transformer library (https://huggingface.co/sentence-transformers). Any and all names should be able to
        be used.
        data_names: These are names attributed to each data source, so we can keep track of the scores they generate in the output dictionary.
        index_names: These are the names of the specific scores we calculate with the associated text metric. This is part of the reason the code is so brittle. It requires
        that these only have four names associated with four specific metrics. Refactoring would be required to make this more robust.
        all_data: A list of each data source grabbed from the grab_data function.

    Outputs:
        corrs: A dictionary of the pearson and spearman correlations generated for each metric, dataset pair.
    """
    corrs = {}
    for data_name, data in zip(data_names, all_data):
        human_scores = average_human_scores(data)
        real_data, synth_data = collate_data(data)
        corrs[data_name] = {}
        for model_name in model_names:
            corrs[data_name][model_name] = {}
            fbd, fcsd = metric_collater(real_data, synth_data, model_name)
            spearman_fbd = abs(spearmanr(fbd, human_scores)[0])
            spearman_fcsd = abs(spearmanr(fcsd, human_scores)[0])
            pearson_fbd = abs(pearsonr(fbd, human_scores)[0])
            pearson_fcsd = abs(pearsonr(fcsd, human_scores)[0])
            metrics = [spearman_fbd, spearman_fcsd, pearson_fbd, pearson_fcsd]
            for name, metric in zip(index_names, metrics):
                corrs[data_name][model_name][name] = metric

    return corrs


convai_data = grab_data("datasets/convai2_annotation.json")

model_names = [
    "paraphrase-mpnet-base-v2",
    "paraphrase-TinyBERT-L6-v2",
    "paraphrase-distilroberta-base-v2",
    "paraphrase-MiniLM-L12-v2",
    "paraphrase-MiniLM-L6-v2",
    "paraphrase-albert-small-v2",
    "paraphrase-MiniLM-L3-v2",
    "nli-mpnet-base-v2",
    "stsb-mpnet-base-v2",
    "stsb-distilroberta-base-v2",
    "nli-roberta-base-v2",
    "stsb-roberta-base-v2",
    "nli-distilroberta-base-v2",
]

data_names = [
    "convai",
]

index_names = ["spearman_fbd", "spearman_fcsd", "pearson_fbd", "pearson_fcsd"]

all_data = [
    convai_data,
]

corrs = experiments(model_names, data_names, index_names, all_data)

print(corrs)

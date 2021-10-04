from typing import List, Dict, Any, Tuple
import math

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sentence_transformers import util
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import linalg

def get_embeddings(text_column: pd.Series, model) -> List[torch.Tensor]:
    '''
    Will generate sentence transformer embeddings that will be used for text metrics. The
    embeddings are generated using the sentence transformer library that is built on top of
    huggingface. You can use any of the huggingface models to instantiate the sentence
    transformer model, but expect varying levels of accuracy. These embeddings are intended
    to be from the highest caliber models tested by SBert.
    
    Args:
        text_column: This is the column containing text that should be embedded. This usually
        will be identified by the manifest and then passed through. Need to update to either
        be a list of columns or a Pandas DataFrame containing that subsection of the data.
        model: This is the instantiated model. Make sure to instantiate the model before
        passing it through. In future iterations, you will pass a string containing the model
        name and then the model will instantiate in this function.
        
    Return:
        A list of embeddings that are torch tensors. May update to be a list of numpy arrays
        in the future.
    '''
    '''embeddings = []
    for text in text_column:
        if type(text) == str:
            embeddings.append(model.encode(text, convert_to_tensor = True))'''
    
    text = list(text_column)
    embeddings = model.encode(text, convert_to_tensor = True)
    
    return embeddings

def calc_mean_var(embeddings: List[np.array]) -> Tuple[Any, Any]:
    '''
    Calculate the mean and covariance of the data. Will also work on single dimensional data.
    
    Args:
        embeddings: A list of embeddings generated from the sentence transformer model. Before
        passing through this function, make sure you convert from torch.Tensor type to
        np.array type.
        
    Return:
        Will return either a mean vector or a mean float. The variance will always be an array,
        but will be a 1-dim array if the mean is a float.
    '''
    mean = np.mean(embeddings, axis = 0)
    var = np.cov(embeddings, rowvar = False)
    
    return mean, var

def calc_FBD(real_mean: Any, 
             real_var: Any, 
             synth_mean: Any, 
             synth_var: Any, 
             esp:float = 1e-6) -> float:
    '''
    This calculate the Frechet BERT Distance as proposed in the following paper:
    
    https://arxiv.org/pdf/2105.02573.pdf
    
    It is inspired by the Frechet Distance that has become popular (and standard) in GANs
    concerned with image generation. It assumes a Gaussian distribution under the embeddings
    and calculates the distance between the real distribution and the synthetic distribution.
    The closer the distributions, the better the synthetic data represents the real data.
    The positives of this metric are:
    
    1) It does not require fine tuning the underlying embedding model on the data it is scoring
    2) It does not require a reference point for each example, thus minimizing computational
    overhead and not leaking anything about the data (since it is an aggregate metric).
    3) Based on this paper, it seems to correlate better with human judgement (the gold standard)
    than other popular, traditional metrics (BLEU, ROGUE, BERTScore, etc.). However, it should be
    noted that other papers have found better metrics, but usually they require some reference,
    they require fine-tuning on the specific data needed to be scored, or the interpretation of
    the score is more convoluted.
    
    The lowest this score can be is zero, but it is unbounded from above.
    
    Args:
        real_mean: the mean of the real vectors
        synth_mean: the mean of the synthetic vectors
        real_var: the covariance matrix of the real vectors
        synth_var: the covariance matrix of the synthetic vectors
        esp: Because we are taking the product and squareroot of matrices, we may result in
        some strange behavior. To take care of this, we add on a small epsilon to the diagonal.
        Invertible (well-behaved) matrices are a dense space in the space of all matrices, thus
        adding a small pertebation (like epsilon) to the diagonal will more likely than not
        produce an invertible matrix.
        
    Return:
        The Frechet Distance as a float value. This can then be used to generate a score for
        the SQS report.
    '''
    diff = real_mean-synth_mean
    covmean, _ = linalg.sqrtm(real_var.dot(synth_var), disp = False)
    
    if not np.isfinite(covmean).all():
        offset = np.eye(real_var.shape[0]) * esp
        covmean = linalg.sqrtm((real_var + offset).dot(synth_var + offset))
        
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    return (diff.dot(diff) + np.trace(real_var) + np.trace(synth_var) - 2*np.trace(covmean))

def create_scores(real_embeddings: List[torch.Tensor], 
                  synth_embeddings: List[torch.Tensor]) -> Tuple[List[float], List[float]]:
    '''
    Creates the cosine similarity scores between the different embeddings and the average
    embedding of the real vectors. It takes each real embedding and the average real embedding
    and generates a cosine similarity score. It does the same thing for each synthetic embedding
    against the average real embedding. Thus, we get two distributions of similarity scores, all of
    which tell us how much each example relates to the real average embedding. We can then find
    the Frechet distance between these two distributions.
    
    Args:
        real_embeddings: The embeddings generated on the real data
        synth_embeddings: The embeddings generated on the synthetic data
    
    Return:
        A list of real cosine similarity scores and a list of synthetic cosine similarity scores
    '''
    
    real_avg_embedding = sum(real_embeddings)/len(real_embeddings)
    
    real_scores = []
    for real_embedding in real_embeddings:
        real_scores.append(util.pytorch_cos_sim(real_avg_embedding, real_embedding))
    
    synth_scores = []
    for synth_embedding in synth_embeddings:
        synth_scores.append(util.pytorch_cos_sim(real_avg_embedding, synth_embedding))
        
    return real_scores, synth_scores

def calc_FCSD(real_score_mean: float, 
              real_score_var: float,
              synth_score_mean: float, 
              synth_score_var: float) -> float:
    '''
    A Frechet Distance function specifically designed for float data. This is intended to be used
    only for the cosine similarity scores.
    
    Args:
        real_score_mean: the mean of the real cosine similarity scores
        synth_score_mean: the mean of the synthetic cosine similarity scores
        real_score_var: the variance of the real cosine similarity scores
        synth_score_var: the variance of the synthetic cosine similarity scores
        
    Return:
        The Frechet Distance between the two distributions
    '''
    return ((real_score_mean - synth_score_mean)**2 + 
            real_score_var + 
            synth_score_var - 
            2 * math.sqrt(real_score_var * synth_score_var))

def metrics_run(real_text: pd.Series, 
                synth_text: pd.Series, 
                model_name: str) -> Tuple[float, float]:
    '''
    Purpose is to grab the real and synthetic text to create text metrics. The real text should be
    detected in the manifest or through the detection function we have built. The synthetic
    text should be generated by our synthetic generation models, and the columns should be named
    similarly. In this first iteration of the run function, we assume that we are taking in a 
    Pandas Series instead of a data frame. In the future, we will take a list of column names
    and a Pandas DataFrame to generate these metrics. Since we are using the sbert library, we
    need to declare what underlying transformer model will be used for the embeddings.
    
    The outputs should graphs of the real and synthetic cosine similarity distributions, along with
    the FBD and FCSD scores that can then be created into scores for the SQS report.
    
    Args:
        real_text: A series representation of a column of text
        synth_text: Similar to above, but for the synthetic data
        model_name: The name of the underlying model for the SentenceTransformer. This can be
        changed as better models are devleoped, but note that the standard model (which will
        be defined soon) is the one that has been tested to see if it corresponds with human
        judgement.
        
    Outputs:
        fbd_score: This score represents the distance of the embedding distributions.
        fcsd_score: This score represents the distance of the cosine similarity distributions.
    '''
    model = SentenceTransformer(model_name)
    
    real_embeddings = get_embeddings(real_text, model)
    synth_embeddings = get_embeddings(synth_text, model)
    
    real_scores, synth_scores = create_scores(real_embeddings, synth_embeddings)
    
    stats_dict = {}
    values_list = [real_embeddings, synth_embeddings, real_scores, synth_scores]
    dict_names = ['real', 'synth', 'real_scores', 'synth_scores']
    for vals, name in zip(values_list, dict_names):
        if 'scores' not in name:
            vals = [el.numpy() for el in vals]
        mean, var = calc_mean_var(vals)
        stats_dict[name] = {}
        stats_dict[name]['mean'] = mean
        stats_dict[name]['var'] = var
    
    fbd_score = calc_FBD(stats_dict['real']['mean'], 
                         stats_dict['real']['var'],
                         stats_dict['synth']['mean'],
                         stats_dict['synth']['var'])
    
    fcsd_score = calc_FCSD(stats_dict['real_scores']['mean'], 
                           stats_dict['real_scores']['var'],
                           stats_dict['synth_scores']['mean'],
                           stats_dict['synth_scores']['var'])
    
    sns.displot(pd.DataFrame(real_scores, columns = ['Real Scores']), kind = 'kde')
    plt.show()
    
    sns.displot(pd.DataFrame(synth_scores, columns = ['Synth Scores']), kind = 'kde')
    plt.show()
    
    print('FBD Score: {}'.format(fbd_score))
    print('FCSD Score: {}'.format(fcsd_score))
    
    return fbd_score, fcsd_score  
import torch
import pandas as pd
import sys, math
import numpy as np
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer

def template2vec(log_templates_df:pd.DataFrame, st_model: SentenceTransformer):
    templates = log_templates_df['template'].to_list()
    embeddings = st_model.encode(templates)
    embeddings = np.vstack((np.array([0.0 for _ in range(len(embeddings[1]))]), embeddings))
    return torch.from_numpy(embeddings)

def log2vec(log_seq, embeddings):
    return embeddings[log_seq]

def SVD_dimension_reduction(embeddings: torch.Tensor, dimension=64):
    embeddings = embeddings.numpy()
    if embeddings.shape[0] < dimension:
        dimension = min(64, 2** int(math.log2(embeddings.shape[0])))
    pca = PCA(n_components = dimension)  
    reduced_embeddings = pca.fit_transform(embeddings)
    return torch.from_numpy(reduced_embeddings), dimension
    
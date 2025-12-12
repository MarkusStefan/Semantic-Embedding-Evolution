import pandas as pd
import scipy.sparse as ss
import scipy.io as sio
import numpy as np
import os
import copy

def get_pmi_matrix(file_path, vocab_size, row_flag=True):
    """
    Reads the PMI matrix from a CSV file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} not found")
    
    # Try reading without header first, assuming raw data
    data = pd.read_csv(file_path)
    vals = data.values
    
    # If the dataframe has less than 3 columns, it might be that the first row was interpreted as header
    if vals.shape[1] < 3:
        data = pd.read_csv(file_path, header=None)
        vals = data.values
        
    row_inds = vals[:, 0].astype(int)
    col_inds = vals[:, 1].astype(int)
    data_vals = vals[:, 2].astype(float)
    
    X = ss.coo_matrix((data_vals, (row_inds, col_inds)), shape=(vocab_size, vocab_size))
    
    if row_flag:
        X = ss.csr_matrix(X)
    else:
        X = ss.csc_matrix(X)
        
    return X

def get_batches(vocab_size, batch_size):
    batch_inds = []
    current = 0
    while current < vocab_size:
        inds = list(range(current, min(current + batch_size, vocab_size)))
        current = min(current + batch_size, vocab_size)
        batch_inds.append(inds)
    return batch_inds

def load_static_embeddings(time_points, static_file='data/emb_static.mat'):
    if not os.path.exists(static_file):
        return None, None
    try:
        emb = sio.loadmat(static_file)['emb']
        U = [copy.deepcopy(emb) for _ in time_points]
        V = [copy.deepcopy(emb) for _ in time_points]
        return U, V
    except Exception as e:
        print(f"Error loading static embeddings: {e}")
        return None, None

def init_random_embeddings(vocab_size, rank, time_points):
    U, V = [], []
    u0 = np.random.randn(vocab_size, rank) / np.sqrt(rank)
    v0 = np.random.randn(vocab_size, rank) / np.sqrt(rank)
    
    for _ in time_points:
        U.append(u0.copy())
        V.append(v0.copy())
    return U, V

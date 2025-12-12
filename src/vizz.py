import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os

def load_wordlist(path):
    if not os.path.exists(path):
        return None, None
    
    wordlist = []
    word2id = {}
    
    try:
        # Try reading as the original wordlist.txt (one word per line)
        with open(path, 'r') as f:
            first_line = f.readline().strip()
            if ',' in first_line:
                # It's likely the CSV format: ID,Word,Count
                # Reset file pointer
                f.seek(0)
                import pandas as pd
                df = pd.read_csv(path, header=None)
                # Assuming format: ID, Word, ...
                # Sort by ID just in case
                df = df.sort_values(0)
                wordlist = df[1].astype(str).tolist()
                word2id = {w: i for i, w in enumerate(wordlist)}
            else:
                # Standard text file
                f.seek(0)
                wordlist = [line.strip() for line in f]
                word2id = {w: i for i, w in enumerate(wordlist)}
                
    except Exception as e:
        print(f"Error loading wordlist: {e}")
        return None, None
        
    return wordlist, word2id

def to_numpy(tensor):
    if hasattr(tensor, 'cpu'):
        try:
            return tensor.cpu().numpy()
        except RuntimeError:
            # Fallback for numpy version mismatch
            return np.array(tensor.cpu().tolist())
    return tensor

def plot_norm_trajectory(U_list, words, word2id, time_points, save_path=None):
    """
    Plots the norm of the word embeddings over time.
    U_list: list of numpy arrays (T x V x r) or list of tensors
    """
    plt.figure(figsize=(10, 6))
    markers = ['+', 'o', 'x', '*', 's', 'd', '^', 'v']
    
    for i, w in enumerate(words):
        if w not in word2id:
            print(f"Word {w} not in vocabulary")
            continue
            
        wid = word2id[w]
        norms = []
        for t in range(len(time_points)):
            emb = U_list[t]
            emb = to_numpy(emb)
            
            vec = emb[wid, :]
            norms.append(np.linalg.norm(vec))
            
        norms = np.array(norms)
        # Normalize by sum
        if np.sum(norms) > 0:
            norms = norms / np.sum(norms)
            
        plt.plot(time_points, norms, marker=markers[i % len(markers)], markersize=7, label=w)
        
    plt.legend()
    plt.xlabel('Year')
    plt.ylabel('Normalized Norm')
    plt.title('Word Norm Trajectory')
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def plot_tsne_trajectory(U_list, word, word2id, wordlist, time_points, top_k=50, save_path=None):
    """
    Plots the t-SNE trajectory of a word and its nearest neighbors.
    """
    if word not in word2id:
        print(f"Word {word} not in vocabulary")
        return
        
    wid = word2id[word]
    
    X = []
    labels = []
    is_target = []
    
    for t_idx, year in enumerate(time_points):
        emb = U_list[t_idx]
        emb = to_numpy(emb)
        
        # Normalize embedding for cosine similarity
        norm = np.linalg.norm(emb, axis=1, keepdims=True)
        norm[norm == 0] = 1e-10
        emb_norm = emb / norm
        
        v = emb_norm[wid, :]
        d = np.dot(emb_norm, v)
        
        idx = np.argsort(d)[::-1][:top_k]
        
        for k in range(top_k):
            neighbor_idx = idx[k]
            neighbor_word = wordlist[neighbor_idx]
            labels.append(f"{neighbor_word} ({year})")
            is_target.append(neighbor_idx == wid)
            X.append(emb[neighbor_idx, :])
            
    X = np.vstack(X)
    
    tsne = TSNE(n_components=2, metric='euclidean', random_state=42)
    Z = tsne.fit_transform(X)
    
    plt.figure(figsize=(12, 10))
    
    # Plot trajectory
    target_indices = [i for i, x in enumerate(is_target) if x]
    target_Z = Z[target_indices, :]
    plt.plot(target_Z[:, 0], target_Z[:, 1], 'r-', alpha=0.5)
    
    for i in range(len(labels)):
        if is_target[i]:
            plt.plot(Z[i, 0], Z[i, 1], 'ro', markersize=8)
            plt.text(Z[i, 0], Z[i, 1], labels[i], fontsize=10, fontweight='bold')
        else:
            plt.plot(Z[i, 0], Z[i, 1], 'b.', alpha=0.3)
            
    plt.title(f"Trajectory of '{word}'")
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

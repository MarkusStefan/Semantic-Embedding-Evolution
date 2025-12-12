import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def update_embeddings(U, Y, Vm1, Vp1, lam, tau, gam, ind, iflag, device='cpu'):
    """
    Update V (or U) given the other matrices.
    Solves min 1/2 |Y - U*V'|^2 + ...
    
    U: n x r (Tensor)
    Y: n x b (Tensor)
    Vm1: b x r (Tensor) - V at t-1
    Vp1: b x r (Tensor) - V at t+1
    ind: list or Tensor - indices of the batch
    iflag: bool - True if t is not 0 or T-1 (has both neighbors), False otherwise
    """
    
    # Ensure inputs are tensors
    if not isinstance(U, torch.Tensor): U = torch.tensor(U, dtype=torch.float32, device=device)
    if not isinstance(Y, torch.Tensor): Y = torch.tensor(Y, dtype=torch.float32, device=device)
    if not isinstance(Vm1, torch.Tensor): Vm1 = torch.tensor(Vm1, dtype=torch.float32, device=device)
    if not isinstance(Vp1, torch.Tensor): Vp1 = torch.tensor(Vp1, dtype=torch.float32, device=device)
    
    r = U.shape[1]
    
    UtU = torch.matmul(U.T, U) # r x r
    eye = torch.eye(r, device=device)
    
    if iflag:
        M = UtU + (lam + 2*tau + gam) * eye
    else:
        M = UtU + (lam + tau + gam) * eye
    
    Uty = torch.matmul(U.T, Y) # r x b
    Ub = U[ind, :].T # r x b
    
    A = Uty + gam * Ub + tau * (Vm1.T + Vp1.T) # r x b
    
    # Solve M * Vhat = A
    # torch.linalg.solve(M, A) solves M X = A.
    
    Vhat = torch.linalg.solve(M, A)
    return Vhat.T # b x r

def get_closest_words(wid, U, top_k=10):
    C = []
    for t in range(len(U)):
        temp = U[t]
        # Use torch operations if tensor
        if isinstance(temp, torch.Tensor):
            # Normalize
            temp_norm = torch.nn.functional.normalize(temp, p=2, dim=1)
            target = temp_norm[wid, :].unsqueeze(0)
            
            # Cosine similarity
            K = torch.mm(target, temp_norm.T).squeeze()
            
            # Top k
            values, indices = torch.topk(K, top_k)
            C.append(indices.cpu().tolist())
        else:
            # Fallback to sklearn for numpy arrays
            K = cosine_similarity(temp[wid, :].reshape(1, -1), temp)
            mxinds = np.argsort(-K)
            mxinds = mxinds[0, 0:top_k] # Top k
            C.append(mxinds)
    return C

def compute_symscore(U, V):
    if isinstance(U, torch.Tensor) and isinstance(V, torch.Tensor):
        return torch.norm(U - V).item()**2
    
    if isinstance(U, torch.Tensor): U = U.cpu().numpy()
    if isinstance(V, torch.Tensor): V = V.cpu().numpy()
    return np.linalg.norm(U - V)**2

def compute_smoothscore(U, Um1, Up1):
    if isinstance(U, torch.Tensor) and isinstance(Um1, torch.Tensor) and isinstance(Up1, torch.Tensor):
        return (torch.norm(U - Up1)**2 + torch.norm(U - Um1)**2).item()

    if isinstance(U, torch.Tensor): U = U.cpu().numpy()
    if isinstance(Um1, torch.Tensor): Um1 = Um1.cpu().numpy()
    if isinstance(Up1, torch.Tensor): Up1 = Up1.cpu().numpy()
    X = np.linalg.norm(U - Up1)**2 + np.linalg.norm(U - Um1)**2
    return X

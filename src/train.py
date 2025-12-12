import argparse
import numpy as np
import torch
import time
import os
import sys

# Add the parent directory to sys.path to allow imports from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataloader import get_pmi_matrix, get_batches, load_static_embeddings, init_random_embeddings
from src.utils import update_embeddings
from src.models import DynamicWord2Vec

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Using device: {device}")
    
    # Parameters
    nw = args.nw
    T = range(args.start_year, args.end_year + 1)
    
    model = DynamicWord2Vec(nw, T, args.rank, device=device)
    
    # Initialization
    print("Initializing...")
    U_init, V_init = load_static_embeddings(T, args.static_file)
    if U_init is None:
        print("Static embeddings not found, initializing randomly.")
        U_init, V_init = init_random_embeddings(nw, args.rank, T)
    
    model.initialize(U_init, V_init)
    
    # Batches
    if args.batch_size < nw:
        b_ind = get_batches(nw, args.batch_size)
    else:
        b_ind = [list(range(nw))]
        
    save_file_prefix = f"L{args.lam}T{args.tau}G{args.gam}A{args.emph}"
    save_path = os.path.join(args.save_dir, save_file_prefix)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    start_time = time.time()
    
    for iteration in range(args.iters):
        print(f"Iteration {iteration}")
        
        # Check if already trained
        # Note: The original code saved with a prefix. My model.save uses the directory.
        # I should adjust model.save or the path passed to it.
        # Let's stick to the original naming convention if possible, or just use a clean folder structure.
        # I'll use a subdirectory for the parameters.
        
        iter_save_dir = os.path.join(args.save_dir, save_file_prefix)
        if model.load(iter_save_dir, iteration):
            print(f"Iteration {iteration} loaded successfully")
            continue
            
        # Shuffle times
        if iteration == 0:
            times = list(range(len(T)))
        else:
            times = np.random.permutation(range(len(T)))
            
        for t_idx in times:
            t = T[t_idx]
            print(f"Iteration {iteration}, Time {t} (Index {t_idx})")
            
            # Assuming data files are named 0.csv, 1.csv... corresponding to index
            f = f"{args.data_dir}{t_idx}.csv" 
            
            try:
                pmi = get_pmi_matrix(f, nw, row_flag=False) # CSC for column slicing
                pmi_row = get_pmi_matrix(f, nw, row_flag=True) # CSR for row slicing
            except FileNotFoundError:
                print(f"Data file {f} not found. Skipping.")
                continue

            # Determine neighbors for U
            if t_idx == 0:
                up = model.U[t_idx+1]
                um = torch.zeros_like(up)
                iflag = False
            elif t_idx == len(T) - 1:
                up = torch.zeros_like(model.U[t_idx-1])
                um = model.U[t_idx-1]
                iflag = False
            else:
                up = model.U[t_idx+1]
                um = model.U[t_idx-1]
                iflag = True
                
            # Update U
            for ind in b_ind:
                # Get batch of PMI
                pmi_seg = pmi[:, ind].todense() # numpy matrix
                pmi_seg_tensor = torch.tensor(pmi_seg, dtype=torch.float32, device=device)
                
                um_batch = um[ind, :]
                up_batch = up[ind, :]
                
                # Call update
                u_new_batch = update_embeddings(model.V[t_idx], pmi_seg_tensor, um_batch, up_batch, 
                                                args.lam, args.tau, args.gam, ind, iflag, device=device)
                
                model.U[t_idx][ind, :] = u_new_batch
                
            # Determine neighbors for V
            if t_idx == 0:
                vp = model.V[t_idx+1]
                vm = torch.zeros_like(vp)
                iflag = False
            elif t_idx == len(T) - 1:
                vp = torch.zeros_like(model.V[t_idx-1])
                vm = model.V[t_idx-1]
                iflag = False
            else:
                vp = model.V[t_idx+1]
                vm = model.V[t_idx-1]
                iflag = True
                
            # Update V
            for ind in b_ind:
                pmi_seg = pmi_row[ind, :].todense() # b x n
                pmi_seg_tensor = torch.tensor(pmi_seg, dtype=torch.float32, device=device).T # n x b
                
                vm_batch = vm[ind, :]
                vp_batch = vp[ind, :]
                
                v_new_batch = update_embeddings(model.U[t_idx], pmi_seg_tensor, vm_batch, vp_batch,
                                                args.lam, args.tau, args.gam, ind, iflag, device=device)
                
                model.V[t_idx][ind, :] = v_new_batch
                
        # Save after iteration
        model.save(iter_save_dir, iteration)
        print(f"Saved iteration {iteration}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--nw', type=int, default=20936)
    parser.add_argument('--start_year', type=int, default=1990)
    parser.add_argument('--end_year', type=int, default=2015)
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--data_dir', type=str, default='data/wordPairPMI_')
    parser.add_argument('--save_dir', type=str, default='results/')
    parser.add_argument('--static_file', type=str, default='data/emb_static.mat')
    parser.add_argument('--iters', type=int, default=5)
    parser.add_argument('--lam', type=float, default=10)
    parser.add_argument('--gam', type=float, default=100)
    parser.add_argument('--tau', type=float, default=50)
    parser.add_argument('--rank', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=20936) # Default to full batch
    parser.add_argument('--emph', type=float, default=1)
    
    args = parser.parse_args()
    train(args)

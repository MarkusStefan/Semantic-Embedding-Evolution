import torch
import pickle
import os
import numpy as np

class DynamicWord2Vec:
    def __init__(self, vocab_size, time_points, rank, device='cpu'):
        self.vocab_size = vocab_size
        self.time_points = time_points
        self.rank = rank
        self.device = device
        
        self.U = [] # List of tensors
        self.V = [] # List of tensors
        
    def initialize(self, U_init, V_init):
        # U_init, V_init are lists of numpy arrays
        self.U = [torch.tensor(u, dtype=torch.float32, device=self.device) for u in U_init]
        self.V = [torch.tensor(v, dtype=torch.float32, device=self.device) for v in V_init]
        
    def save(self, path, iteration):
        if not os.path.exists(path):
            os.makedirs(path)
        
        # Save using torch.save to avoid numpy dependency issues
        torch.save(self.U, os.path.join(path, f"ngU_iter{iteration}.pt"))
        torch.save(self.V, os.path.join(path, f"ngV_iter{iteration}.pt"))
            
    def load(self, path, iteration):
        try:
            # Try loading torch format first
            u_path = os.path.join(path, f"ngU_iter{iteration}.pt")
            v_path = os.path.join(path, f"ngV_iter{iteration}.pt")
            
            if os.path.exists(u_path) and os.path.exists(v_path):
                self.U = torch.load(u_path, map_location=self.device)
                self.V = torch.load(v_path, map_location=self.device)
                return True
            
            # Fallback to pickle (legacy)
            with open(os.path.join(path, f"ngU_iter{iteration}.p"), "rb") as f:
                U_np = pickle.load(f)
            with open(os.path.join(path, f"ngV_iter{iteration}.p"), "rb") as f:
                V_np = pickle.load(f)
            
            self.initialize(U_np, V_np)
            return True
        except IOError:
            return False

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import MultiModalEncoder
from autoAug import autoAUG
from augmentations import jitter, scaling

import copy

def l2_normalize(x, dim=1):
    return x / torch.sqrt(torch.sum(x**2, dim=dim).unsqueeze(dim))

class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, temperature=0.5, verbose=False):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature))
        self.verbose = verbose
            
    def forward(self, emb_i, emb_j):
        """
        emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
        z_i, z_j as per SimCLR paper
        """
        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)

        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        if self.verbose: print("Similarity matrix\n", similarity_matrix, "\n")
            
        def l_ij(i, j):
            z_i_, z_j_ = representations[i], representations[j]
            sim_i_j = similarity_matrix[i, j]
            if self.verbose: print(f"sim({i}, {j})={sim_i_j}")
                
            numerator = torch.exp(sim_i_j / self.temperature)
            one_for_not_i = torch.ones((2 * self.batch_size, )).to(emb_i.device).scatter_(0, torch.tensor([i]).to(emb_i.device), 0.0)
            if self.verbose: print(f"1{{k!={i}}}",one_for_not_i)
            
            denominator = torch.sum(
                one_for_not_i * torch.exp(similarity_matrix[i, :] / self.temperature)
            )    
            if self.verbose: print("Denominator", denominator)
                
            loss_ij = -torch.log(numerator / denominator)
            if self.verbose: print(f"loss({i},{j})={loss_ij}\n")
                
            return loss_ij.squeeze(0)

        N = self.batch_size
        loss = 0.0
        for k in range(0, N):
            loss += l_ij(k, k + N) + l_ij(k + N, k)
        return 1.0 / (2*N) * loss

class SimCLR(nn.Module):
    def __init__(self, in_channel, seq_encoder, use_leaves=False):
        super().__init__()
        self.use_leaves = use_leaves
        if use_leaves:
            self.view = autoAUG(num_channel = in_channel)
            # self.view2 = autoAUG(num_channel = in_channel)
        self.encoder = seq_encoder
        # self.fc = nn.Linear(512, seq_emb_dimension)
    
    def forward(self, desc, common, tabular):
        if self.use_leaves:
            x1 = self.view(common)
            x2 = self.view(common)
        else:
            x1 = jitter(scaling(common, 0.03), 0.03)
            x2 = jitter(scaling(common, 0.03), 0.03)
        
        view1_emb = self.encoder(x1)
        view2_emb = self.encoder(x2)
        
        contrastive_loss = contrastiveLoss(x1_emb, x2_emb)
        
        return contrastive_loss

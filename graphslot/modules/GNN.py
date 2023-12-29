from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
from torch import nn
from torch.nn import functional as F

from graphslot.lib import utils
import graphslot.modules as modules

Array = torch.Tensor

class ConstructGraph:
    def __init__(self):
        pass
    
    def construct(self, slots, attn):
        """Constructs a graph from slot representations.
        
        Args:
            slots: Tensor of shape (batch_size, num_slots, slot_dim)
        
        Returns:
            adj: Adjacency matrix of shape (batch_size, num_slots, num_slots) 
            feats: Feature matrix of shape (batch_size, num_slots, slot_dim)
        """

        feats = slots
        device = feats.device
        
        num_heads, _, num_slots, attn_size = attn.shape 

        img_size = int(attn_size ** 0.5)
        attn = attn.view(num_heads, num_slots, -1)

        adj = torch.zeros((num_heads, num_slots, num_slots), dtype=torch.float).to(device)
        for i in range(num_heads):
            max_index = torch.tensor(torch.argmax(attn[i], dim=-1), dtype=torch.float)
            positions = torch.stack([max_index // img_size, max_index % img_size], dim=1)
            distances = torch.cdist(positions, positions, p=1)

            # 1-nearest neighbor
            for j in range(len(distances)):
                min_index = torch.argmin(distance[j])
                adj[i][j][min_index] = 1

            # # threshold distance
            # threshold = 2.0
            # adj[i] = (distances <= threshold).float()
        
        return adj, feats
    
class GraphEmb(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.gcn = nn.ModuleList()
        self.gcn.append(nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU()))
        self.gcn.append(nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim), 
                        nn.ReLU()))
        
    def forward(self, adj, feats):
        for gcn in self.gcn:
            feats = gcn(torch.bmm(adj, feats))
            
        return feats


class GraphCorrector(nn.Module):
    """Encoder for single video frame."""

    def __init__(self, 
                 slot_attention: nn.Module, 
                 construct_graph: nn.Module, 
                 graph_emb: nn.Module,
                 alpha: float = 1.0,
                 beta: float = 1.0
                 ):
        super().__init__()

        self.slot_attention = slot_attention
        self.construct_graph = construct_graph
        self.graph_emb = graph_emb
        self.alpha = alpha
        self.beta = beta
    
    def forward(self,
                slots: Array, 
                inputs: Array,
                padding_mask: Optional[Array] = None
                ):

        inputs = inputs.flatten(1, 2)

        slots, attn = self.slot_attention(slots, inputs, padding_mask)
        adj, feats = self.construct_graph.construct(slots, attn)
        refined_feats = self.graph_emb(adj, feats)

        slots = self.alpha * feats + self.beta * refined_feats

        return slots, attn


# nohup python -m savi.main --seed 42 --gpu 0,1,2,3 --mode=movic_ > movic_.txt &
# python -m savi.main --seed 42 --gpu 0,1,2,3 --mode=dev
# nohup gsutil -m cp -r "gs://kubric-public/tfds/movi_d/128x128" ./movi_d > movi_d.txt &
# python -m savi.main --seed 42 --gpu 0,1,2,3 --mode=dev --eval --resume_from /home/shared/data/hanwen-data/SAVi-pytorch/experiments/test_test/snapshots/movia_res_gcn_100000.pt

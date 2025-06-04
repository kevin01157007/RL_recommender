# LightGCN.py
import torch
import torch.nn as nn
from .LightGCNConv import LightGCNConv

class LightGCN(nn.Module):
    def __init__(self,num_users,num_items,emb_size=64,n_layers=2):
        super().__init__()
        self.num_users, self.num_items = num_users, num_items
        self.embedding=nn.Embedding(num_users+num_items, emb_size)
        nn.init.xavier_uniform_(self.embedding.weight)
        self.convs=nn.ModuleList([LightGCNConv() for _ in range(n_layers)])
    def forward(self, edge_index):
        x=self.embedding.weight; embs=[x]
        for conv in self.convs:
            x=conv(x,edge_index); embs.append(x)
        return torch.stack(embs).mean(0)
    def get_user_item(self, edge_index):
        all_emb=self(edge_index)
        return all_emb[:self.num_users], all_emb[self.num_users:]

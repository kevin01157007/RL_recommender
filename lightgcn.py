# lightgcn.py
import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.typing import Adj
from LightGCNConv import LightGCNConv

class LightGCN(nn.Module):
    def __init__(self, config: dict, device=None):
        super().__init__()
        self.num_users = config["n_users"]
        self.num_items = config["m_items"]
        self.embedding_size = config["embedding_size"]
        self.num_layers = config["num_layers"]

        self.embedding = nn.Embedding(self.num_users + self.num_items, self.embedding_size)
        nn.init.normal_(self.embedding.weight, std=0.1)
        print('Use normal distribution initializer')

        self.convs = nn.ModuleList([LightGCNConv() for _ in range(self.num_layers)])
        self.device = device
        if device:
            self.to(device)

    def forward(self, edge_index: Adj) -> Tensor:
        x = self.embedding.weight
        all_embeddings = [x]

        for conv in self.convs:
            x = conv(x, edge_index)
            all_embeddings.append(x)

        all_embeddings = torch.stack(all_embeddings, dim=0)  # [num_layers+1, num_nodes, dim]
        out = torch.mean(all_embeddings, dim=0)  # average over all layers
        return out

    def get_user_item_embeddings(self):
        out = self.forwarded_embedding
        user_emb = out[:self.num_users]
        item_emb = out[self.num_users:]
        return user_emb, item_emb

    def __repr__(self):
        return f'LightGCN(num_users={self.num_users}, num_items={self.num_items}, embedding_size={self.embedding_size})'

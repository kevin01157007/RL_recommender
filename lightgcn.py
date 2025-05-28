# lightgcn.py
import torch
import torch.nn as nn
from LightGCNConv import LightGCNConv

class LightGCN(nn.Module):
    def __init__(self,num_users,num_items,emb_size=64,n_layers=2, initial_user_emb=None, initial_item_emb=None):
        super().__init__()
        self.num_users, self.num_items = num_users, num_items
        self.embedding=nn.Embedding(num_users+num_items, emb_size)
        
        if initial_user_emb is not None and initial_item_emb is not None:
            print("Initializing LightGCN embeddings with pre-trained embeddings.")
            if initial_user_emb.shape[0] == num_users and initial_item_emb.shape[0] == num_items and \
               initial_user_emb.shape[1] == emb_size and initial_item_emb.shape[1] == emb_size:
                self.embedding.weight.data[:num_users] = initial_user_emb
                self.embedding.weight.data[num_users:] = initial_item_emb
            else:
                print("Warning: Shape mismatch for pre-trained embeddings. Using Xavier initialization instead.")
                nn.init.xavier_uniform_(self.embedding.weight)
        else:
            print("No pre-trained embeddings provided for LightGCN. Using Xavier initialization.")
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

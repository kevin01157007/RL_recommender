# q_network.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class QNetwork(nn.Module):
    def __init__(self, num_nodes, emb_dim, gcn_layers=2):
        super().__init__()
        # --- learnable initial embeddings (LightGCN 預訓練權重稍後載入) -------------
        self.node_emb = nn.Embedding(num_nodes, emb_dim)

        # --- GCN stack ----------------------------------------------------------------
        self.convs = nn.ModuleList(
            [GCNConv(emb_dim, emb_dim) for _ in range(gcn_layers)]
        )

        # --- Estimator (MLP) ----------------------------------------------------------
        self.fc1 = nn.Linear(emb_dim * 3, emb_dim)
        self.fc2 = nn.Linear(emb_dim, 1)

        # virtual node index  (= num_nodes-1)
        self.v_id = num_nodes - 1

    # --------------------------------------------------------------------------
    # 建立「全連結」edge_index_ext：把 virtual node 與所有真節點雙向相連
    # --------------------------------------------------------------------------
    def _augment_edge_index(self, edge_index):
        N = self.v_id                      # 真節點數
        device = edge_index.device
        full = torch.arange(N, device=device) #0~N-1

        # s ↔ i  兩向邊
        row = torch.cat([full, torch.full((N,), self.v_id, device=device)])
        col = torch.cat([torch.full((N,), self.v_id, device=device), full])
        v_edges = torch.stack([row, col], dim=0)           

        return torch.cat([edge_index, v_edges], dim=1)      

    def encode(self, edge_index):
        x = self.node_emb.weight
        ei = self._augment_edge_index(edge_index)

        # 前 gcn_layers 層
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        y = F.relu(conv(x, self._augment_edge_index(edge_index)))
        return x, y                             

    def forward(self, edge_index, u_idx, v_idx):
        z_all, y = self.encode(edge_index)
        z_u   = z_all[u_idx]                   
        z_v   = z_all[v_idx]                   
        z_s   = y[self.v_id].expand_as(z_u)

        h = torch.cat([z_u, z_v, z_s], dim=-1)
        h = F.relu(self.fc1(h))
        q = self.fc2(h).squeeze(-1) #把最後一個維度移除
        return q

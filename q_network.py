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
        # 最後再加一層，把 virtual node 與 full graph 聚合
        self.final_conv = GCNConv(emb_dim, emb_dim)

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
        full = torch.arange(N, device=device)

        # s ↔ i  兩向邊
        row = torch.cat([full, torch.full((N,), self.v_id, device=device)])
        col = torch.cat([torch.full((N,), self.v_id, device=device), full])
        v_edges = torch.stack([row, col], dim=0)           # (2, 2N)

        return torch.cat([edge_index, v_edges], dim=1)      # (2, |E|+2N)

    # --------------------------------------------------------------------------
    # encoding：所有節點 → z_all (含 z_s = z_all[self.v_id])
    # --------------------------------------------------------------------------
    def encode(self, edge_index):
        x = self.node_emb.weight
        ei = self._augment_edge_index(edge_index)

        # 前 gcn_layers 層
        for conv in self.convs:
            x = F.relu(conv(x, ei))

        # 再跑一層，讓 virtual node 聚合全域訊息
        x = F.relu(self.final_conv(x, ei))
        return x                                # (N+1, d)  最後一列就是 z_s

    # --------------------------------------------------------------------------
    # forward：不再手動傳 z_s，網路自己取最後那一行
    #   edge_index : (2, |E|)
    #   u_idx, v_idx : (k,)
    # --------------------------------------------------------------------------
    def forward(self, edge_index, u_idx, v_idx):
        z_all = self.encode(edge_index)
        z_u   = z_all[u_idx]                   # (k, d)
        z_v   = z_all[v_idx]                   # (k, d)
        z_s   = z_all[self.v_id].expand_as(z_u)

        h = torch.cat([z_u, z_v, z_s], dim=-1)
        h = F.relu(self.fc1(h))
        q = self.fc2(h).squeeze(-1)            # (k,)
        return q

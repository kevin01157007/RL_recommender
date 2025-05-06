import torch
import pandas as pd
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops
from torch_geometric.loader import NeighborLoader
from LightGCNRS import LightGCNRS
from LightGCN import LightGCN

class RecSimEnv:
    def __init__(self,
                 init_edge_index,  # tensor([2,E])
                 n_user,
                 n_item,
                 agent,
                 rec_model,
                 sim,
                 device="cuda"):
        self.device   = torch.device(device)
        self.n_user   = n_user
        self.n_item   = n_item
        self.agent    = agent

        # 1) 建立動態圖
        self.edge_index = init_edge_index.clone().to(self.device)

        # 2) 建立推薦與 Simulator（可先保留）
        self.rec_model = rec_model
        self.sim       = sim

        # 3) **載入 pre-saver 的 embeddings**
        self.user_emb = torch.load("user_emb.pt", map_location=self.device)  # [n_user, d]
        self.item_emb = torch.load("item_emb.pt", map_location=self.device)  # [n_item, d]

    def run(self, n_round=5, k_rec=20):
        rec_items_list  = []
        new_interactions = []
        seen = {u: set() for u in range(self.n_user)}

        for t in range(n_round):
            print(f"===== Time step {t} =====")
            for u in range(self.n_user):
                # 1) 取得 Top-k 推薦
                rec_items = self.rec_model.recommend(u, k=k_rec, exclude=seen[u])
                print(f"[User {u}] Recommended {len(rec_items)} items (k_rec={k_rec})")

                # 2) **用 raw dot-product 打分，不用 sim.score**
                rec_with_scores = [
                    (item_id, (self.user_emb[u] @ self.item_emb[item_id]).item())
                    for item_id in rec_items
                ]

                for item_id, score_val in rec_with_scores:
                    print(f"[User {u}] Item {item_id} RawScore = {score_val:.4f}")

                new_interactions.extend([
                    (u, i_id)
                    for i_id, _ in rec_with_scores
                    if torch.rand(1).item() < self.sim.score(u, i_id).item()
                ])

                # 標記為已推薦
                seen[u].update(rec_items)

                # 收集推薦記錄
                rec_items_list.extend([(u, t, item_id, item_raw_score) for item_id, item_raw_score in rec_with_scores])

            # (Step2-c) 更新 edge_index
            if new_interactions:
                ei_extra = torch.tensor(new_interactions, dtype=torch.long).T.to(self.device)
                
                if ei_extra.numel() > 0:
                    self.edge_index = torch.cat([self.edge_index, ei_extra], dim=1)

        # 最後存檔
        print(f"Attempting to save rec_items.csv with {len(rec_items_list)} entries.")
        if rec_items_list:
            print(f"First entry in rec_items_list: {rec_items_list[0]} (should have 4 elements)")
        pd.DataFrame(rec_items_list,
                 columns=['user_id', 'time_step', 'item', 'score']
                ).to_csv('rec_items.csv', index=False)
        
        print(f"Attempting to save new_interactions.csv with {len(new_interactions)} entries.")
        pd.DataFrame(new_interactions, columns=['u','i']).to_csv('new_interactions.csv', index=False)
        print("CSV files saving process finished.")


    # ---------------- 內部工具 ----------------
    def finetune_lightgcn(self, epochs=3, batch_size=2048):
        loader = NeighborLoader(Data(edge_index=add_self_loops(self.edge_index)[0]),
                                num_neighbors=[10],
                                batch_size=batch_size,
                                input_nodes=None,
                                shuffle=True)
        opt = torch.optim.Adam(self.rec_model.parameters(), lr=1e-3)
        bpr = torch.nn.BCEWithLogitsLoss()

        for _ in range(epochs):
            for batch in loader:
                users = batch.n_id[batch.n_id < self.n_user]
                pos_i = torch.randint(0, self.n_item, (len(users),), device=self.device)
                neg_i = torch.randint(0, self.n_item, (len(users),), device=self.device)

                uemb  = self.sim.get_user_emb(users)
                p_emb = self.sim.get_item_emb(pos_i)
                n_emb = self.sim.get_item_emb(neg_i)
                pos_logits = (uemb * p_emb).sum(-1)
                neg_logits = (uemb * n_emb).sum(-1)
                loss = bpr(pos_logits, torch.ones_like(pos_logits)) + \
                       bpr(neg_logits, torch.zeros_like(neg_logits))

                opt.zero_grad()
                loss.backward()
                opt.step()


    def evaluate(self):
        import networkx as nx
        import community as community_louvain
        import torch
        import numpy as np
        from collections import defaultdict
        from torch.nn.functional import softplus

        # 1. 圖結構指標：社群數 & 平均度數
        g = nx.Graph()
        n_nodes = self.n_user + self.n_item
        g.add_nodes_from(range(n_nodes))
        # edge_index shape = [2, E]
        edges = self.edge_index.t().tolist()
        g.add_edges_from(edges)
        part = community_louvain.best_partition(g, resolution=1.0)
        n_comm = len(set(part.values()))
        degs = dict(g.degree())
        avg_deg = sum(degs.values()) / n_nodes

        # 2. 推薦模型嵌入 & BPR loss
        #    2.1 取出所有 user/item 嵌入
        user_emb, item_emb = self.rec_model.model.get_user_item(self.edge_index)
        #    2.2 隨機負採樣測試 BPR
        samples = []
        for u, i in self.test_data:
            # 以每個測試正例配一個隨機負例
            neg = np.random.choice(self.n_item)
            samples.append((u, i, neg))
        users, pos, neg = zip(*samples)
        users = torch.tensor(users, dtype=torch.long, device=self.rec_model.device)
        pos   = torch.tensor(pos,   dtype=torch.long, device=self.rec_model.device)
        neg   = torch.tensor(neg,   dtype=torch.long, device=self.rec_model.device)

        #    2.3 計算 BPR 損失
        u_e = user_emb[users]
        p_e = item_emb[pos]
        n_e = item_emb[neg]
        pos_scores = torch.sum(u_e * p_e, dim=1)
        neg_scores = torch.sum(u_e * n_e, dim=1)
        loss_bpr = torch.mean(softplus(neg_scores - pos_scores))
        #    正則項：對第0層嵌入做 L2
        e0 = self.rec_model.model.embedding.weight
        reg = (e0[users].norm(2).pow(2)
            + e0[pos + self.n_user].norm(2).pow(2)
            + e0[neg + self.n_user].norm(2).pow(2)) / users.size(0)
        loss = loss_bpr + self.lambda_reg * reg

        # 3. Top-K 指標：Precision, Recall, NDCG
        K = self.k_eval
        #    3.1 計算所有 user-item 原始分數矩陣
        all_u, all_i = user_emb, item_emb
        scores = torch.matmul(all_u, all_i.t())  # shape = (U, I)
        #    3.2 對每個 user 排序、計算 metrics
        precisions, recalls, ndcgs = [], [], []
        test_pos = defaultdict(set)
        for u, i in self.test_data:
            test_pos[u].add(i)

        for u in range(self.n_user):

            topk = torch.topk(scores[u], K).indices.tolist()
            hits = [1 if i in test_pos[u] else 0 for i in topk]
            tp = sum(hits)
            precisions.append(tp / K)
            recalls.append(tp / max(1, len(test_pos[u])))
            # NDCG
            dcg = sum(h / np.log2(idx+2) for idx, h in enumerate(hits))
            ideal = min(len(test_pos[u]), K)
            idcg = sum(1 / np.log2(i+2) for i in range(ideal)) or 1.0
            ndcgs.append(dcg / idcg)

        prec = float(np.mean(precisions))
        rec  = float(np.mean(recalls))
        ndcg = float(np.mean(ndcgs))

        # 4. 返回所有統計指標
        return {
            "n_comm":   n_comm,
            "avg_deg":  avg_deg,
            "bpr_loss": float(loss.item()),
            "precision": prec,
            "recall":    rec,
            "ndcg":      ndcg
        }

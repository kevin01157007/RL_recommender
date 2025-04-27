import torch, copy, collections
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops
from torch_geometric.loader import NeighborLoader
from LightGCNSimulator import LightGCNSimulator
from LightGCNRS import LightGCNRS
from lightgcn import LightGCN
from LightGCNConv import LightGCNConv
import pandas as pd
class RecSimEnv:
    def __init__(self,
                 init_edge_index,      # tensor([2,E])
                 n_user,
                 n_item,
                 agent,
                 rec_model,
                 sim,
                 device="cuda"):
        self.device   = device
        self.n_user   = n_user
        self.n_item   = n_item
        self.agent    = agent

        # 1) 建立動態圖
        self.edge_index = init_edge_index.clone().to(device)

        # 2) 建立 LightGCN + Simulator
        self.rec_model = rec_model
        self.sim = sim

    # ---------------- 主流程 ----------------
    def run(self, n_round=5, k_rec=20):
        hist_metrics = []
        rec_items_list=[]
        new_interactions = []
        seen = {u: set() for u in range(5)}
        for t in range(n_round):
            print(f"===== Time step {t} =====")

            # (Step2) 產生推薦 + 使用 simulator 回饋
            for u in range(5):
                rec_items = self.rec_model.recommend(u, k=k_rec, exclude=seen[u])
                for item in rec_items:
                    rec_items_list.append((u, t, item))  # (user_id, time_step, recommended_item)
                    seen[u].add(item)
                for i in rec_items:
                    p = self.sim.score(u, i).item()
                    if torch.rand(1).item() < p:
                        new_interactions.append((u, i))
            print(seen)

            # # (Step2‑b) Agent 決策額外曝光
            # extra_edges = self.agent.select_edges(self)
            # new_interactions.extend(extra_edges)

            # (Step2‑c) 更新 edge_index
            ei_extra = torch.tensor(new_interactions, dtype=torch.long).T.to(self.device)
            if ei_extra.numel() > 0:
                for u, i in new_interactions:
                    self.edge_index[u][i] = 1
        rec_items_df = pd.DataFrame(rec_items_list, columns=['user_id', 'time_step', 'recommended_item'])
        rec_items_df.to_csv('rec_items11.csv', index=False)
        in_df = pd.DataFrame(new_interactions, columns=['u', 'i'])
        in_df.to_csv('new_interactions.csv', index=False)
            # # (Step3) 評估
            # metrics = self.evaluate()
            # hist_metrics.append(metrics)
            # print(metrics)

            # # (Step4) 重新訓練 LightGCN（可微調幾 epoch 就行）
            # self.finetune_lightgcn(epochs=3, batch_size=2048)

        # return hist_metrics

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
        # 簡易統計：社群數量 ＆ 平均度數
        import community as community_louvain, networkx as nx
        g = nx.Graph()
        g.add_nodes_from(range(self.n_user + self.n_item))
        g.add_edges_from(self.edge_index.t().tolist())
        part = community_louvain.best_partition(g, resolution=1.0)
        n_comm = len(set(part.values()))
        degs = dict(g.degree())
        avg_deg = sum(degs.values()) / len(degs)
        return {"n_comm": n_comm, "avg_deg": avg_deg}
from env import RecSimEnv      # 你自己的實作
import torch
import LightGCNSimulator
import LightGCNRS
from LightGCN import LightGCN
from LightGCNConv import LightGCNConv

if __name__ == "__main__":
    # --------------- 載入初始離線資料 ----------------
    # 這裡以隨機圖示範，實驗時請用 Gowalla / Yelp split 的 train‑set
    n_user, n_item = 5, 3883
    init_edges = torch.zeros(n_user, n_item)

    # --------------- 建立環境 ----------------
    agent = 0
    env   = RecSimEnv(init_edge_index=init_edges,
                      n_user=n_user,
                      n_item=n_item,
                      agent=agent,
                      device="cuda:0")

    env.run(n_round=5)
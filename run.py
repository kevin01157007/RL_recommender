from env import RecSimEnv      # 你自己的實作
import torch
from LightGCNSimulator import LightGCNSimulator
from LightGCNRS import LightGCNRS
from lightgcn import LightGCN
from LightGCNConv import LightGCNConv
from MovieLens import MovieLens
import os
def trans_ml(dat, thres):
    """
    Transform function that assign non-negative entries >= thres 1, and non-
    negative entries <= thres 0. Keep other entries the same.
    """
    thres = thres[0]
    matrix = dat['edge_index']
    matrix[(matrix < thres) & (matrix > -1)] = 0
    matrix[(matrix >= thres)] = 1
    dat['edge_index'] = matrix
    return dat

rec_model_config = {
"n_users": 200,
"m_items": 3883,
"embedding_size": 64,
"num_layers": 3,
}

if __name__ == "__main__":
    # --------------- 載入初始離線資料 ----------------
    # 這裡以隨機圖示範，實驗時請用 Gowalla / Yelp split 的 train‑set
    model = torch.load('model.pth',weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = os.getcwd()
    movielens = MovieLens(root=root, transform=trans_ml)
    data = movielens.get()
    n_user, n_item = rec_model_config["n_users"], rec_model_config["m_items"]
    init_edges = torch.zeros(n_user, n_item)

    rs = LightGCN(rec_model_config, device=device)
    rec_model = LightGCNRS(rs, {
            "edge_index": init_edges,
            "users":      list(range(n_user)),
            "items":      list(range(n_item))})
    
    sim = LightGCNSimulator(model, data)
    # --------------- 建立環境 ----------------
    agent = 0
    env   = RecSimEnv(init_edge_index=init_edges,
                      n_user=n_user,
                      n_item=n_item,
                      agent=agent,
                      rec_model = rec_model,
                      sim = sim,  
                      device="cuda:0")

    env.run(n_round=5)
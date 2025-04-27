import torch
from torch import Tensor
from typing import List, Dict

class LightGCNRS:
    def __init__(self,
                 model,                  # 已初始化好的 LightGCN
                 data,             # {"edge_index": ..., "users": [...], "items":[...]}
                 device="cuda"):
        self.device = device
        self.model  = model.to(device).eval()
        self.data   = data
        self.update_all_emb()            # 先把全圖 embedding 算好並快取

    # ------------------------------------------------------------------
    def update_all_emb(self) -> None:
        self.model.eval()
        with torch.no_grad():
            # 直接调用 forward(edge_index)：
            all_emb: Tensor = self.model(
                self.data["edge_index"].to(self.device)
            )
        n_user = len(self.data["users"])
        self.users_emb: Tensor = all_emb[:n_user]
        self.items_emb: Tensor = all_emb[n_user:]
       # (I, d)

    # ------------------------------------------------------------------
    #  Embedding 讀取
    def get_user_emb(self, uid: int) -> Tensor:
        return self.users_emb[uid]

    def get_item_emb(self, iid: int) -> Tensor:
        return self.items_emb[iid]
    def recommend(self, uid, k=20, exclude=None):
        """
        回傳 uid 的 top‑k item id list
        exclude: set() -> 不推薦已互動商品
        """
        u = self.get_user_emb(uid)
        items_emb = self.items_emb
        scores = torch.matmul(items_emb, u)   # |I|
        if exclude is not None:
            scores[list(exclude)] = -1e9
        topk = torch.topk(scores, k).indices.cpu().tolist()
        return topk
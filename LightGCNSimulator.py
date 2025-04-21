import torch
from torch import Tensor
from typing import List, Dict

class LightGCNSimulator:
    def __init__(self,
                 model,                  # 已初始化好的 LightGCN
                 data: Dict,             # {"edge_index": ..., "users": [...], "items":[...]}
                 device="cuda"):
        self.device = device
        self.model  = model.to(device).eval()
        self.data   = data
        self.update_all_emb()            # 先把全圖 embedding 算好並快取

    # ------------------------------------------------------------------
    def update_all_emb(self) -> None:
        self.model.eval()
        with torch.no_grad():
            all_emb: Tensor = self.model(
                self.model.embedding_user_item.weight,
                self.data["edge_index"].to(self.device)
            )
        n_user = len(self.data["users"])
        self.users_emb: Tensor = all_emb[:n_user]        # (U, d)
        self.items_emb: Tensor = all_emb[n_user:]        # (I, d)

    # ------------------------------------------------------------------
    #  Embedding 讀取
    def get_user_emb(self, uid: int) -> Tensor:
        return self.users_emb[uid]

    def get_item_emb(self, iid: int) -> Tensor:
        return self.items_emb[iid]

    def score(self, uid: int, iid: int, sigmoid: bool = True) -> Tensor:
        s: Tensor = torch.matmul(self.get_user_emb(uid),self.get_item_emb(iid).t())
        return torch.sigmoid(s) if sigmoid else s         # shape = ()

    def recommend(self, uid: int, k: int = 20,
                  exclude: set = None) -> List[int]:
        """
        回傳 uid 的 top‑k item id list
        exclude: 不放入已互動/曝光商品
        """
        u: Tensor = self.get_user_emb(uid)                # (d,)
        scores: Tensor = torch.matmul(self.items_emb, u)  # (I,)
        if exclude:
            scores[list(exclude)] = -1e9
        return torch.topk(scores, k).indices.cpu().tolist()
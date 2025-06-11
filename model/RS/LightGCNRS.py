import torch
from ..LightGCN.LightGCN import LightGCN

class LightGCNRS:
    def __init__(self, n_users, model, device="cuda"):
        self.device  = device
        self.n_users = n_users
        self.model   = model.to(device)

    def recommend(self, uid, k=20, exclude=None):
        model = self.model.eval()
        user_emb = model.embedding.weight[:self.n_users]
        item_emb = model.embedding.weight[self.n_users:]
        u = user_emb[uid]
        scores = torch.matmul(item_emb, u)   # |I|
        if exclude is not None:
            scores[list(exclude)] = -1e9
        topk = torch.topk(scores, k).indices.cpu().tolist()
        return topk

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# rs = LightGCN(5000, 3883).to(device)
# rec_model = LightGCNRS(n_users=5000, model=rs, device=device)




import torch
import random
from utility.bulid_dtrain_graph import build_dtrain_graph
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ❶ Load fixed LightGCN embeddings
USER_EMB = torch.load("user_emb.pt", map_location=DEVICE)
ITEM_EMB = torch.load("item_emb.pt", map_location=DEVICE)

# ❷ Build initial graph and Louvain community count

EMB_DIM = USER_EMB.size(1)
NUM_USERS = USER_EMB.shape[0]
NUM_ITEMS = ITEM_EMB.shape[0]
print(f"Embeddings  |  users={NUM_USERS}  items={NUM_ITEMS}  dim={EMB_DIM}")


# ❸ Candidate pool (Top-100 per user)
with torch.no_grad():
    score = USER_EMB[:10] @ ITEM_EMB.t()
TOP_K = 10
_, topk_idx = torch.topk(score, TOP_K, dim=1)
CANDIDATES = topk_idx.cpu().tolist()
print(CANDIDATES)

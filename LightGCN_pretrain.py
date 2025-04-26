#!/usr/bin/env python3
# coding: utf-8

import os
import random
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn, optim, Tensor
from torch_geometric.nn import MessagePassing
import matplotlib.pyplot as plt
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# Step 1. 讀取 ratings (只為了互動資料)
ratings = pd.read_csv('raw/ml-1m/ratings.dat', sep='::', names=['user_id', 'movie_id', 'rating', 'timestamp'], engine='python',encoding='latin-1' )

# --------------------
# Step 2. 讀取所有 movie id / user id 來建完整 mapping
# - user_id 直接從 ratings.user_id.unique() 即可 (因為user是乾淨的，不會有被過濾掉的user)
# - movie_id 要從完整的 movies.dat 讀，不要從 ratings來
movies = pd.read_csv('raw/ml-1m/movies.dat', sep='::', names=['movie_id', 'title', 'genres'], engine='python',encoding='latin-1' )

users = pd.read_csv('raw/ml-1m/users.dat', sep='::', names=['user_id', 'gender', 'age', 'occupation', 'zip'], engine='python', encoding='latin-1')

selected_user_ids = users.user_id.tolist()[:1000]
all_movie_ids = movies.movie_id.unique()
uid_map = {old: new for new, old in enumerate(selected_user_ids)}

mid_map = {old: new for new, old in enumerate(sorted(all_movie_ids))}

num_users = len(uid_map)
num_items = len(mid_map)

print(f"Users: {num_users}, Items: {num_items}")

# --------------------
# Step 3. 過濾高分資料
ratings = ratings[ratings.user_id.isin(selected_user_ids)]  # 只保留這500個user的互動
ratings_high = ratings[ratings.rating >= 3].copy()

# 加入新的 0-based id
ratings_high['u'] = ratings_high.user_id.map(uid_map)
ratings_high['i'] = ratings_high.movie_id.map(mid_map)

# --------------------
# Step 4. 產生 (user,item) 互動資料
interactions = list(zip(ratings_high.u.tolist(), ratings_high.i.tolist()))

# -----------------------------
# 2. per-user split: train/val/test
# -----------------------------
def split_interactions(interactions, num_users, seed=42):
    random.seed(seed)
    user2items = defaultdict(list)
    for u, i in interactions:
        user2items[u].append(i)

    train, val, test = [], [], []
    for u, items in user2items.items():
        if len(items) < 3:
            # 少於 3 筆全部留在 train
            train.extend([(u, i) for i in items])
            continue
        random.shuffle(items)
        test.append((u, items.pop()))
        val.append((u, items.pop()))
        train.extend([(u, i) for i in items])
    return train, val, test

train_inter, val_inter, test_inter = split_interactions(interactions, num_users)

# -----------------------------
# 3. 建立雙向 graph adjacency (for LightGCN)
# -----------------------------
def build_edge_index(pairs, num_users, num_items):
    edge_list = []
    for u, i in pairs:
        edge_list.append((u, i + num_users))
        edge_list.append((i + num_users, u))
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    return edge_index  # shape [2, 2*|pairs|]

train_edge_index = build_edge_index(train_inter, num_users, num_items)
val_edge_index   = build_edge_index(val_inter,   num_users, num_items)
test_edge_index  = build_edge_index(test_inter,  num_users, num_items)
full_interactions = train_inter + val_inter + test_inter
full_edge_index = build_edge_index(full_interactions, num_users, num_items)
# -----------------------------
# 4. Sample positive／negative
# -----------------------------
def sample_pos_neg(train_pairs, num_users, num_items, num_negatives=1, seed=None):
    """
    回傳 tensor(shape=[N,3]) of (user, pos_item, neg_item)
    只從 train_pairs 採樣正例，負例從 user 未互動過中隨機抽。
    """
    if seed is not None:
        random.seed(seed)
    user2pos = defaultdict(set)
    for u, i in train_pairs:
        user2pos[u].add(i)
    all_items = set(range(num_items))

    samples = []
    for u in range(num_users):
        pos_items = list(user2pos[u])
        if not pos_items:
            continue
        for _ in range(num_negatives):
            pos = random.choice(pos_items)
            neg = random.choice(list(all_items - user2pos[u]))
            samples.append((u, pos, neg))
    return torch.tensor(samples, dtype=torch.long)

# -----------------------------
# 5. LightGCN 定義
# -----------------------------
class LightGCNConv(MessagePassing):
    def __init__(self):
        super().__init__(aggr='add')

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        row, col = edge_index
        deg = torch.bincount(row, minlength=x.size(0)).float()
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j: Tensor, norm: Tensor) -> Tensor:
        return norm.view(-1, 1) * x_j

class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, emb_size=64, n_layers=2):
        super().__init__()
        self.num_users, self.num_items = num_users, num_items
        self.embedding = nn.Embedding(num_users + num_items, emb_size)
        nn.init.xavier_uniform_(self.embedding.weight)
        self.convs = nn.ModuleList([LightGCNConv() for _ in range(n_layers)])

    def forward(self, edge_index: Tensor) -> Tensor:
        x = self.embedding.weight
        embs = [x]
        for conv in self.convs:
            x = conv(x, edge_index)
            embs.append(x)
        embs = torch.stack(embs, dim=0).mean(dim=0)  # 均值聚合
        return embs

    def get_user_item(self, edge_index: Tensor):
        all_emb = self(edge_index)
        return all_emb[:self.num_users], all_emb[self.num_users:]

# -----------------------------
# 6. BPR Loss
# -----------------------------
def bpr_loss(model, users, pos, neg, edge_index, lambda_reg=1e-4):
    user_emb, item_emb = model.get_user_item(edge_index)
    u_emb = user_emb[users]
    p_emb = item_emb[pos]
    n_emb = item_emb[neg]

    pos_score = (u_emb * p_emb).sum(dim=1)
    neg_score = (u_emb * n_emb).sum(dim=1)
    loss_bpr = F.softplus(neg_score - pos_score).mean()

    # L2 正則化 on 原始 embedding
    e0 = model.embedding.weight
    reg = (e0[users].norm(2).pow(2) +
           e0[pos + model.num_users].norm(2).pow(2) +
           e0[neg + model.num_users].norm(2).pow(2)) / users.size(0)
    return loss_bpr + lambda_reg * reg, loss_bpr.detach(), reg.detach()

# -----------------------------
# 7. Precision & Recall@K
# -----------------------------
def precision_recall_at_k(model, edge_index_train, test_pairs, K=10):
    """
    edge_index_train: 用於計算 embedding
    test_pairs: list of (u,i) ground truth
    """
    user_emb, item_emb = model.get_user_item(edge_index_train)
    user_pos_test = defaultdict(set)
    for u,i in test_pairs:
        user_pos_test[u].add(i)

    precisions, recalls = [], []
    for u, pos_set in user_pos_test.items():
        scores = (user_emb[u] @ item_emb.t())
        topk = torch.topk(scores, K).indices.tolist()
        hit = len([i for i in topk if i in pos_set])
        precisions.append(hit / K)
        recalls.append(hit / len(pos_set))
    return np.mean(precisions), np.mean(recalls)

# -----------------------------
# 8. Training Loop
# -----------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = LightGCN(
    num_users=num_users,
    num_items=num_items,
    emb_size=64,
    n_layers=2
).to(device)

opt = optim.Adam(model.parameters(), lr=1e-3)

# 將 edge_index 移到 GPU/CPU
train_edge_index = train_edge_index.to(device)

num_epochs    = 50
batch_size    = 1024
num_neg_per_u = 200
K             = 10

loss_history = []
val_prec_history = []
val_rec_history = []

for epoch in range(1, num_epochs+1):
    model.train()
    t0 = time.time()

    samples = sample_pos_neg(train_inter, num_users, num_items,
                             num_negatives=num_neg_per_u, seed=epoch)
    samples = samples[torch.randperm(len(samples))].to(device)

    total_loss = 0
    for st in range(0, len(samples), batch_size):
        batch = samples[st: st+batch_size]
        u, p, n = batch[:,0], batch[:,1], batch[:,2]
        opt.zero_grad()
        loss, loss_bpr, loss_reg = bpr_loss(model, u, p, n, train_edge_index)
        loss.backward()
        opt.step()
        total_loss += loss.item() * u.size(0)

    loss_history.append(total_loss / len(samples))

    # 評估 validation
    model.eval()
    with torch.no_grad():
        prec_val, rec_val = precision_recall_at_k(
            model, train_edge_index, val_inter, K=K
        )
    val_prec_history.append(prec_val)
    val_rec_history.append(rec_val)
    print(f"Epoch {epoch:02d} | "
          f"Time {time.time()-t0:.1f}s | "
          f"AvgLoss {total_loss/len(samples):.4f} | "
          f"Val P@{K} {prec_val:.4f}, R@{K} {rec_val:.4f}")

# 訓練結束，單獨測試
model.eval()
with torch.no_grad():
    prec_test, rec_test = precision_recall_at_k(
        model, train_edge_index, test_inter, K=K
    )
print(f"Final Test P@{K}: {prec_test:.4f}, R@{K}: {rec_test:.4f}")


# 最後可儲存模型
# torch.save(model.state_dict(), "lightgcn_ml1m.pth")
full_edge_index = full_edge_index.to(device)
user_emb, item_emb = model.get_user_item(full_edge_index)
torch.save(user_emb, "user_emb.pt")
torch.save(item_emb, "item_emb.pt")


# 假設已經有以下三個 list：
# loss_history: 每個 epoch 的平均 BPR Loss
# val_prec_history: 每個 epoch 的 Validation Precision@K
# val_rec_history: 每個 epoch 的 Validation Recall@K

epochs = list(range(1, len(loss_history) + 1))

# 繪製 Loss
plt.figure()
plt.plot(epochs, loss_history, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Average BPR Loss')
plt.title('Loss over Epochs')
plt.show()

# 繪製 Precision@K
plt.figure()
plt.plot(epochs, val_prec_history, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Precision@K')
plt.title('Validation Precision over Epochs')
plt.show()

# 繪製 Recall@K
plt.figure()
plt.plot(epochs, val_rec_history, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Recall@K')
plt.title('Validation Recall over Epochs')
plt.show()

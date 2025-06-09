import os, random, time
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch_geometric.nn import MessagePassing
import matplotlib.pyplot as plt
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ────────────────────────────────────────────────────────────────────────────────
# 1. Load ratings / movies / users (MovieLens‑1M)
#   (unchanged – assumes the same three *.dat files already pre‑splitted)
# ────────────────────────────────────────────────────────────────────────────────
ratings_train = pd.read_csv('../../data/train.dat', sep=',', names=['user_id', 'movie_id', 'rating', 'timestamp'], engine='python', encoding='latin-1')
ratings_val   = pd.read_csv('../../data/val.dat',   sep=',', names=['user_id', 'movie_id', 'rating', 'timestamp'],  engine='python', encoding='latin-1')
ratings_test  = pd.read_csv('../../data/test.dat',  sep=',', names=['user_id', 'movie_id', 'rating', 'timestamp'],  engine='python', encoding='latin-1')

num_users, num_items = 5950, 3191  # make sure these match the real max‑id + 1
print(f"Users: {num_users}, Items: {num_items}")

# ────────────────────────────────────────────────────────────────────────────────
# 2. Convert <DataFrame> → interaction‑pair lists
# ────────────────────────────────────────────────────────────────────────────────

def pairs_from(df):
    return list(zip(df.user_id, df.movie_id))

train_inter, val_inter, test_inter = map(pairs_from, (ratings_train, ratings_val, ratings_test))
print(f"train {len(train_inter):,} | val {len(val_inter):,} | test {len(test_inter):,}")

# ────────────────────────────────────────────────────────────────────────────────
# 3. Graph helpers (bidirectional (u,i) → two edges (u‑>i, i‑>u))
# ────────────────────────────────────────────────────────────────────────────────

def build_edge_index(pairs, num_users, num_items):
    edges = []
    for u, i in pairs:
        edges.append((u, i + num_users))
        edges.append((i + num_users, u))
    return torch.tensor(edges, dtype=torch.long).t().contiguous()

train_edge_index = build_edge_index(train_inter, num_users, num_items)           # train‑only
val_edge_index   = build_edge_index(train_inter + val_inter, num_users, num_items)  # train ∪ val (for propagation)
full_edge_index  = build_edge_index(train_inter + val_inter + test_inter, num_users, num_items)

# ────────────────────────────────────────────────────────────────────────────────
# 4. Negative‑sampling utilities
# ────────────────────────────────────────────────────────────────────────────────

# Pre‑compute user → positive‑item list for O(1) sampling later
train_user_pos_dict = defaultdict(list)
for u, i in train_inter:
    train_user_pos_dict[u].append(i)

val_user_pos_dict = defaultdict(list)
for u, i in val_inter:
    val_user_pos_dict[u].append(i)

test_user_pos_dict = defaultdict(list)
for u, i in test_inter:
    test_user_pos_dict[u].append(i)

all_items_set = set(range(num_items))

def sample_mini_batch(batch_size, user_pos_dict):
    """Uniformly sample *users*, then one positive & one negative item each."""
    users, pos_items, neg_items = [], [], []
    for _ in range(batch_size):
        u = random.randrange(num_users)
        # guarantee the user has at least 1 interaction (ML‑1M true for all)
        p = random.choice(user_pos_dict[u])
        # sample negative not in pos set
        while True:
            n = random.randrange(num_items)
            if n not in user_pos_dict[u]:
                break
        users.append(u)
        pos_items.append(p)
        neg_items.append(n)
    return (torch.tensor(users,     dtype=torch.long),
            torch.tensor(pos_items, dtype=torch.long),
            torch.tensor(neg_items, dtype=torch.long))

# ────────────────────────────────────────────────────────────────────────────────
# 5. LightGCN layers / model
# ────────────────────────────────────────────────────────────────────────────────

class LightGCNConv(MessagePassing):
    def __init__(self):
        super().__init__(aggr='add')

    def forward(self, x, edge_index):
        row, col = edge_index
        deg = torch.bincount(row, minlength=x.size(0)).float()
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, emb_size=64, n_layers=3):
        super().__init__()
        self.num_users, self.num_items = num_users, num_items
        self.embedding = nn.Embedding(num_users + num_items, emb_size)
        nn.init.xavier_uniform_(self.embedding.weight)
        self.convs = nn.ModuleList([LightGCNConv() for _ in range(n_layers)])

    def forward(self, edge_index):
        x = self.embedding.weight
        embs = [x]
        for conv in self.convs:
            x = conv(x, edge_index)
            embs.append(x)
        return torch.stack(embs).mean(0)

    def get_user_item(self, edge_index):
        all_emb = self(edge_index)
        return all_emb[: self.num_users], all_emb[self.num_users :]


# ────────────────────────────────────────────────────────────────────────────────
# 6. BPR loss
# ────────────────────────────────────────────────────────────────────────────────

def bpr_loss(model, users, pos, neg, edge_index, lambda_reg=1e-4):
    user_emb, item_emb = model.get_user_item(edge_index)
    u_emb = user_emb[users]
    p_emb = item_emb[pos]
    n_emb = item_emb[neg]
    pos_score = (u_emb * p_emb).sum(1)
    neg_score = (u_emb * n_emb).sum(1)
    loss_bpr = F.softplus(neg_score - pos_score).mean()

    # L2 regularisation on the *raw* embeddings (not the aggregated ones)
    e0 = model.embedding.weight
    reg = (
        e0[users].norm(2).pow(2)
        + e0[pos + model.num_users].norm(2).pow(2)
        + e0[neg + model.num_users].norm(2).pow(2)
    ) / users.size(0)

    return loss_bpr + lambda_reg * reg, loss_bpr.detach(), reg.detach()


# ────────────────────────────────────────────────────────────────────────────────
# 7. Dataset‑level BPR loss (for val / test monitoring)
# ────────────────────────────────────────────────────────────────────────────────

def compute_bpr_loss_dataset(model, pairs, edge_index_prop, exclude_pairs=None):
    model.eval()
    # negative sampling using *only the evaluated pairs* to avoid label leakage
    samples = []
    user2pos_tmp = defaultdict(list)
    for u, i in pairs:
        user2pos_tmp[u].append(i)
    for u, pos_list in user2pos_tmp.items():
        for p in pos_list:
            while True:
                n = random.randrange(num_items)
                if n not in user2pos_tmp[u] and (exclude_pairs is None or (u, n) not in exclude_pairs):
                    break
            samples.append((u, p, n))
    samples = torch.tensor(samples, dtype=torch.long)
    u, p, n = samples[:, 0], samples[:, 1], samples[:, 2]
    with torch.no_grad():
        loss, _, _ = bpr_loss(model, u.to(device), p.to(device), n.to(device), edge_index_prop)
    return loss.item()


# ────────────────────────────────────────────────────────────────────────────────
# 8. Top‑K ranking metrics
# ────────────────────────────────────────────────────────────────────────────────

def precision_recall_ndcg_at_k(model, edge_index_prop, test_pairs, train_pairs=None, K=20):
    user_emb, item_emb = model.get_user_item(edge_index_prop)

    seen = defaultdict(set)
    if train_pairs is not None:
        for u, i in train_pairs:
            seen[u].add(i)

    user_pos = defaultdict(set)
    for u, i in test_pairs:
        user_pos[u].add(i)

    pre, rec, ndcg = [], [], []
    for u, pos_set in user_pos.items():
        scores = (user_emb[u] @ item_emb.t()).clone()
        scores[list(seen[u])] = -1e9  # exclude seen items
        topk = torch.topk(scores, K).indices.tolist()
        hits = [1 if i in pos_set else 0 for i in topk]
        hit_cnt = sum(hits)
        pre.append(hit_cnt / K)
        rec.append(hit_cnt / len(pos_set))
        # DCG / IDCG
        dcg = sum(h / np.log2(idx + 2) for idx, h in enumerate(hits))
        ideal_hits = [1] * min(len(pos_set), K)
        idcg = sum(1 / np.log2(idx + 2) for idx in range(len(ideal_hits))) or 1
        ndcg.append(dcg / idcg)
    return np.mean(pre), np.mean(rec), np.mean(ndcg)


# ────────────────────────────────────────────────────────────────────────────────
# 9. Training loop (mini‑batch, lr = 1e‑3)
# ────────────────────────────────────────────────────────────────────────────────
K = 10
num_epochs = 1000
batch_size = 1024

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LightGCN(num_users, num_items, emb_size=64, n_layers=3).to(device)
opt = optim.Adam(model.parameters(), lr=1e-3)

train_edge_index = train_edge_index.to(device)
val_edge_index   = val_edge_index.to(device)
full_edge_index  = full_edge_index.to(device)

loss_hist, val_loss_hist = [], []
val_prec_hist, val_rec_hist, val_ndcg_hist = [], [], []

patience = 20
best_val_loss = float('inf')
patience_counter = 0

steps_per_epoch = int(np.ceil(len(train_inter) / batch_size))  # heuristic
val_steps_per_epoch = int(np.ceil(len(val_inter) / batch_size))  # heuristic
test_steps_per_epoch = int(np.ceil(len(test_inter) / batch_size))  # heuristic

for epoch in range(1, num_epochs + 1):
    model.train()
    t0 = time.time()
    total_loss = 0.0

    for _ in range(steps_per_epoch):
        u, p, n = sample_mini_batch(batch_size, train_user_pos_dict)
        u, p, n = u.to(device), p.to(device), n.to(device)
        opt.zero_grad()
        loss, _, _ = bpr_loss(model, u, p, n, train_edge_index)
        loss.backward()
        opt.step()
        total_loss += loss.item() * u.size(0)

    avg_train_loss = total_loss / (steps_per_epoch * batch_size)
    loss_hist.append(avg_train_loss)

    # ─── Validation ────────────────────────────────────────────────────────────
    model.eval()
    for _ in range(val_steps_per_epoch):
        u, p, n = sample_mini_batch(batch_size, val_user_pos_dict)
        u, p, n = u.to(device), p.to(device), n.to(device)
        with torch.no_grad():
            loss, _, _ = bpr_loss(model, u, p, n, train_edge_index)
        total_loss += loss.item() * u.size(0)

    avg_val_loss = total_loss / (val_steps_per_epoch * batch_size)
    val_loss_hist.append(avg_val_loss)

    
    with torch.no_grad():
        prec, rec, ndcg = precision_recall_ndcg_at_k(
            model, val_edge_index, val_inter, train_inter, K
        )
    val_prec_hist.append(prec)
    val_rec_hist.append(rec)
    val_ndcg_hist.append(ndcg)

    print(
        f"Epoch {epoch:02d} | {time.time() - t0:.1f}s | TrainLoss {avg_train_loss:.4f} | "
        f"ValLoss {avg_val_loss:.4f} | P@{K} {prec:.4f} R@{K} {rec:.4f} NDCG@{K} {ndcg:.4f}"
    )

    # Early stopping -----------------------------------------------------------
    # Early stopping logic
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        best_model_state = model.state_dict()  # Save best model
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

# ────────────────────────────────────────────────────────────────────────────────
# 10. Final Test evaluation (load best model)
# ────────────────────────────────────────────────────────────────────────────────
model.load_state_dict(best_model_state)
model.eval()
for _ in range(test_steps_per_epoch):
    u, p, n = sample_mini_batch(batch_size, test_user_pos_dict)
    u, p, n = u.to(device), p.to(device), n.to(device)
    with torch.no_grad():
        loss, _, _ = bpr_loss(model, u, p, n, train_edge_index)
    total_loss += loss.item() * u.size(0)

avg_test_loss = total_loss / (test_steps_per_epoch * batch_size)
with torch.no_grad():
    prec_test, rec_test, ndcg_test = precision_recall_ndcg_at_k(
        model, full_edge_index, test_inter, train_inter+val_inter, K
    )
print(
    f"\nFinal Test | P@{K} {prec_test:.4f} | R@{K} {rec_test:.4f} | "
    f"NDCG@{K} {ndcg_test:.4f} | BPR Loss {avg_test_loss:.4f}"
)

# ────────────────────────────────────────────────────────────────────────────────
# 11. Save (optional) & plots
# ────────────────────────────────────────────────────────────────────────────────
user_emb, item_emb = model.get_user_item(full_edge_index)
# torch.save(model.state_dict(), "lightgcn_ml1m_fixed.pth")
# torch.save(user_emb.cpu(), "user_emb.pt")
# torch.save(item_emb.cpu(), "item_emb.pt")

# Curves
epochs_r = list(range(1, len(loss_hist) + 1))
plt.figure()
plt.plot(epochs_r, loss_hist, label='Train')
plt.plot(epochs_r, val_loss_hist, label='Val')
plt.xlabel('Epoch'); plt.ylabel('BPR Loss'); plt.legend(); plt.title('Loss')

plt.figure(); plt.plot(epochs_r, val_prec_hist); plt.xlabel('Epoch'); plt.ylabel(f'Precision@{K}'); plt.title('Validation Precision')
plt.figure(); plt.plot(epochs_r, val_rec_hist);  plt.xlabel('Epoch'); plt.ylabel(f'Recall@{K}');    plt.title('Validation Recall')
plt.figure(); plt.plot(epochs_r, val_ndcg_hist); plt.xlabel('Epoch'); plt.ylabel(f'NDCG@{K}');      plt.title('Validation NDCG')
plt.show()

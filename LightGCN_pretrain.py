import os, random, time
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn, optim, Tensor
from torch_geometric.nn import MessagePassing
import matplotlib.pyplot as plt
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ────────────────────────────────────────────────────────────────────────────────
# 1. Load ratings / movies / users (MovieLens‑1M)
# ────────────────────────────────────────────────────────────────────────────────
ratings_train = pd.read_csv('data_split/train.dat', sep=',', names=['user_id', 'movie_id', 'rating', 'timestamp'], engine='python', encoding='latin-1')
ratings_val = pd.read_csv('data_split/val.dat', sep=',', names=['user_id', 'movie_id', 'rating', 'timestamp'], engine='python', encoding='latin-1')
ratings_test = pd.read_csv('data_split/test.dat', sep=',', names=['user_id', 'movie_id', 'rating', 'timestamp'], engine='python', encoding='latin-1')

movies  = pd.read_csv('raw/ml-1m/movies.dat', sep='::', names=['movie_id', 'title', 'genres'],   engine='python', encoding='latin-1')
users   = pd.read_csv('raw/ml-1m/users.dat',  sep='::', names=['user_id', 'gender', 'age', 'occupation', 'zip'], engine='python', encoding='latin-1')

num_users, num_items = 5950, 3191
print(f"Users: {num_users}, Items: {num_items}")

# high‑rating implicit positives
# ratings = ratings[ratings.user_id.isin(selected_user_ids)]
# ratings_high = ratings[ratings.rating >= 4].copy()
# ratings_high['u'] = ratings_high.user_id.map(uid_map)
# ratings_high['i'] = ratings_high.movie_id.map(mid_map)

# ────────────────────────────────────────────────────────────────────────────────
# 2. Split last‑k per user → val / test (k=5 here) ; rest = train
# ────────────────────────────────────────────────────────────────────────────────
test_df        = ratings_test
val_df         = ratings_val
train_df       = ratings_train
print(train_df.head())
print(f"train {len(train_df):,} | val {len(val_df):,} | test {len(test_df):,}")

def pairs_from(df):
    return list(zip(df.user_id, df.movie_id))
train_inter, val_inter, test_inter = map(pairs_from, (train_df, val_df, test_df))
# ────────────────────────────────────────────────────────────────────────────────
# 3. Graph helpers
# ────────────────────────────────────────────────────────────────────────────────

def build_edge_index(pairs, num_users, num_items):
    edges = []
    for u,i in pairs:
        edges.append((u, i+num_users))
        edges.append((i+num_users, u))
    return torch.tensor(edges, dtype=torch.long).t().contiguous()

train_edge_index = build_edge_index(train_inter, num_users, num_items)
val_edge_index = build_edge_index(val_inter+train_inter, num_users, num_items)
full_edge_index  = build_edge_index(train_inter+val_inter+test_inter, num_users, num_items)

# ────────────────────────────────────────────────────────────────────────────────
# 4. Negative sampling for BPR
# ────────────────────────────────────────────────────────────────────────────────

def sample_pos_neg(train_pairs, num_users, num_items, num_negatives=1, seed=None, exclude_pairs=None):
    if seed is not None:
        random.seed(seed)
    user2pos = defaultdict(set)
    for u,i in train_pairs:
        user2pos[u].add(i)
    user2exclude = defaultdict(set)
    for u in range(num_users):
        user2exclude[u] = set(user2pos[u])

    if exclude_pairs is not None:
        for u, i in exclude_pairs:
            user2exclude[u].add(i)
    all_items = set(range(num_items))
    samples=[]
    for u in range(num_users):
        pos_items=list(user2pos[u])
        if not pos_items: continue
        for t in pos_items:
            p=t
            n=random.choice(list(all_items-user2pos[u]-user2exclude[u]))
            samples.append((u,p,n))
    return torch.tensor(samples, dtype=torch.long)

# ────────────────────────────────────────────────────────────────────────────────
# 5. LightGCN
# ────────────────────────────────────────────────────────────────────────────────
class LightGCNConv(MessagePassing):
    def __init__(self): super().__init__(aggr='add') #指定使用加和方式聚合鄰居節點的消息
    def forward(self, x, edge_index):
        row,col=edge_index
        deg=torch.bincount(row, minlength=x.size(0)).float() #統計每個節點的度（連接數）
        deg_inv_sqrt=deg.pow(-0.5); deg_inv_sqrt[deg_inv_sqrt==float('inf')]=0
        norm=deg_inv_sqrt[row]*deg_inv_sqrt[col]
        return self.propagate(edge_index,x=x,norm=norm)
    def message(self,x_j,norm): return norm.view(-1,1)*x_j

class LightGCN(nn.Module):
    def __init__(self,num_users,num_items,emb_size=64,n_layers=2):
        super().__init__()
        self.num_users, self.num_items = num_users, num_items
        self.embedding=nn.Embedding(num_users+num_items, emb_size)
        nn.init.xavier_uniform_(self.embedding.weight)
        self.convs=nn.ModuleList([LightGCNConv() for _ in range(n_layers)])
    def forward(self, edge_index):
        x=self.embedding.weight; embs=[x]
        for conv in self.convs:
            x=conv(x,edge_index); embs.append(x)
        return torch.stack(embs).mean(0)
    def get_user_item(self, edge_index):
        all_emb=self(edge_index)
        return all_emb[:self.num_users], all_emb[self.num_users:]

# ────────────────────────────────────────────────────────────────────────────────
# 6. BPR loss helpers
# ────────────────────────────────────────────────────────────────────────────────

def bpr_loss(model, users, pos, neg, edge_index, lambda_reg=1e-4):
    user_emb,item_emb=model.get_user_item(edge_index)
    u_emb=user_emb[users]; p_emb=item_emb[pos]; n_emb=item_emb[neg]
    pos_score=(u_emb*p_emb).sum(1); neg_score=(u_emb*n_emb).sum(1)
    loss_bpr=F.softplus(neg_score-pos_score).mean()
    e0=model.embedding.weight
    reg=(e0[users].norm(2).pow(2)+e0[pos+model.num_users].norm(2).pow(2)+e0[neg+model.num_users].norm(2).pow(2))/users.size(0)
    return loss_bpr+lambda_reg*reg, loss_bpr.detach(), reg.detach()

# quick evaluator for BPR loss on a (u,i) positive set

def compute_bpr_loss_dataset(model, pairs, edge_index_train, num_negatives=1, exclude_pairs=None):
    model.eval()
    samples=sample_pos_neg(pairs, num_users, num_items, num_negatives, exclude_pairs=exclude_pairs)
    u,p,n=samples[:,0],samples[:,1],samples[:,2]
    with torch.no_grad():
        loss,_ ,_=bpr_loss(model,u.to(device),p.to(device),n.to(device),edge_index_train)
    return loss.item()

# ────────────────────────────────────────────────────────────────────────────────
# 7. Top‑K metrics (Precision / Recall / NDCG)
# ────────────────────────────────────────────────────────────────────────────────

def precision_recall_ndcg_at_k(model, edge_index_train, test_pairs, train_pairs=None, K=10):
    user_emb,item_emb=model.get_user_item(edge_index_train)
    seen=defaultdict(set)
    if train_pairs is not None:
        for u,i in train_pairs: seen[u].add(i)
    user_pos=defaultdict(set)
    for u,i in test_pairs: user_pos[u].add(i)

    pre, rec, ndcg = [], [], []
    for u,pos_set in user_pos.items():
        scores=(user_emb[u]@item_emb.t()).clone()
        scores[list(seen[u])]=-1e9
        topk=torch.topk(scores,K).indices.tolist()
        hits=[1 if i in pos_set else 0 for i in topk]
        hit_cnt=sum(hits)
        pre.append(hit_cnt/K)
        rec.append(hit_cnt/len(pos_set))
        # DCG / IDCG
        dcg=sum(h/np.log2(idx+2) for idx,h in enumerate(hits))
        ideal_hits=[1]*min(len(pos_set),K)
        idcg=sum(1/np.log2(idx+2) for idx in range(len(ideal_hits))) or 1
        ndcg.append(dcg/idcg)
    return np.mean(pre), np.mean(rec), np.mean(ndcg)

# ────────────────────────────────────────────────────────────────────────────────
# 8. Training loop
# ────────────────────────────────────────────────────────────────────────────────
K = 20
num_epochs=200
batch_size=1024
num_neg_per_u=10

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model=LightGCN(num_users,num_items,emb_size=64,n_layers=4).to(device)
opt=optim.Adam(model.parameters(), lr=1e-3)
train_edge_index=train_edge_index.to(device)
val_edge_index=val_edge_index.to(device)

loss_hist=[]; val_loss_hist=[]; val_prec_hist=[]; val_rec_hist=[]; val_ndcg_hist=[]

for epoch in range(1,num_epochs+1):
    model.train(); t0=time.time()
    samples=sample_pos_neg(train_inter,num_users,num_items,num_neg_per_u,seed=epoch)
    samples=samples[torch.randperm(len(samples))].to(device)
    total_loss=0
    for st in range(0,len(samples),batch_size):
        batch=samples[st:st+batch_size]
        u,p,n=batch[:,0],batch[:,1],batch[:,2]
        opt.zero_grad()
        loss,_,_=bpr_loss(model,u,p,n,train_edge_index)
        loss.backward(); opt.step()
        total_loss += loss.item()*u.size(0)
    avg_train_loss=total_loss/len(samples)
    loss_hist.append(avg_train_loss)

    # validation
    val_loss = compute_bpr_loss_dataset(model,val_inter,train_edge_index,exclude_pairs=train_inter+val_inter)
    val_loss_hist.append(val_loss)
    model.eval()
    with torch.no_grad():
        prec,rec,ndcg = precision_recall_ndcg_at_k(model,train_edge_index,val_inter,train_inter,K)
    val_prec_hist.append(prec); val_rec_hist.append(rec); val_ndcg_hist.append(ndcg)
    print(f"Epoch {epoch:02d} | {time.time()-t0:.1f}s | TrainLoss {avg_train_loss:.4f} | ValLoss {val_loss:.4f} | P@{K} {prec:.4f} R@{K} {rec:.4f} NDCG@{K} {ndcg:.4f}")

# ────────────────────────────────────────────────────────────────────────────────
# 9. Final Test evaluation
# ────────────────────────────────────────────────────────────────────────────────
model.eval()
with torch.no_grad():
    prec_test, rec_test, ndcg_test = precision_recall_ndcg_at_k(model,train_edge_index,test_inter,train_inter+val_inter,K)

test_loss = compute_bpr_loss_dataset(model,test_inter,train_edge_index,exclude_pairs=train_inter+val_inter+test_inter)
print(f"\nFinal Test | P@{K} {prec_test:.4f} | R@{K} {rec_test:.4f} | NDCG@{K} {ndcg_test:.4f} | BPR Loss {test_loss:.4f}")

# save model embeddings if needed
full_edge_index=full_edge_index.to(device)
user_emb,item_emb=model.get_user_item(full_edge_index)
# torch.save(model.state_dict(),"lightgcn_ml1m.pth")
torch.save(user_emb.cpu(),"user_emb.pt"); torch.save(item_emb.cpu(),"item_emb.pt")

# ────────────────────────────────────────────────────────────────────────────────
# 10. Plots
# ────────────────────────────────────────────────────────────────────────────────
epochs=list(range(1,len(loss_hist)+1))
plt.figure(); plt.plot(epochs,loss_hist,label='Train'); plt.plot(epochs,val_loss_hist,label='Val'); plt.xlabel('Epoch'); plt.ylabel('BPR Loss'); plt.legend(); plt.title('Loss')
plt.figure(); plt.plot(epochs,val_prec_hist); plt.xlabel('Epoch'); plt.ylabel(f'Precision@{K}'); plt.title('Validation Precision')
plt.figure(); plt.plot(epochs,val_rec_hist); plt.xlabel('Epoch'); plt.ylabel(f'Recall@{K}'); plt.title('Validation Recall')
plt.figure(); plt.plot(epochs,val_ndcg_hist); plt.xlabel('Epoch'); plt.ylabel(f'NDCG@{K}'); plt.title('Validation NDCG')
plt.show()

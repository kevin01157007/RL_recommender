import os, random, time
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
def pairs_from(df):
    return list(zip(df.user_id, df.movie_id))

def build_edge_index(pairs, num_users):
    edges = []
    for u,i in pairs:
        edges.append((u, i+num_users))
        edges.append((i+num_users, u))
    return torch.tensor(edges, dtype=torch.long).t().contiguous()

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

def bpr_loss(model, users, pos, neg, edge_index, lambda_reg=1e-4):
    user_emb,item_emb=model.get_user_item(edge_index)
    u_emb=user_emb[users]; p_emb=item_emb[pos]; n_emb=item_emb[neg]
    pos_score=(u_emb*p_emb).sum(1); neg_score=(u_emb*n_emb).sum(1)
    loss_bpr=F.softplus(neg_score-pos_score).mean()
    e0=model.embedding.weight
    reg=(e0[users].norm(2).pow(2)+e0[pos+model.num_users].norm(2).pow(2)+e0[neg+model.num_users].norm(2).pow(2))/users.size(0)
    return loss_bpr+lambda_reg*reg, loss_bpr.detach(), reg.detach()

def compute_bpr_loss_dataset(model, pairs, device, edge_index_train, num_users, num_items, num_negatives=1, exclude_pairs=None):
    model.eval()
    samples=sample_pos_neg(pairs, num_users, num_items, num_negatives, exclude_pairs=exclude_pairs)
    u,p,n=samples[:,0],samples[:,1],samples[:,2]
    with torch.no_grad():
        loss,_ ,_=bpr_loss(model,u.to(device),p.to(device),n.to(device),edge_index_train)
    return loss.item()

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
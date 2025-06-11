import os, random, time
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
def pairs_from(df):
    return list(zip(df.user_id, df.movie_id))

def build_edge_index(pairs, num_users):
    edges = []
    for u,i in pairs:
        edges.append((u, i+num_users))
        edges.append((i+num_users, u))
    return torch.tensor(edges, dtype=torch.long).t().contiguous()

def sample_mini_batch(batch_size, user_pos_dict, num_items, exclude_user_pos_dict=None):
    """Uniformly sample *users*, then one positive & one negative item each."""
    users, pos_items, neg_items = [], [], []

    for _ in range(batch_size):
        valid_users = [u for u in user_pos_dict.keys() if len(user_pos_dict[u]) > 0]
        u = random.choice(valid_users)
        
        # guarantee the user has at least 1 interaction (ML‑1M true for all)
        p = random.choice(user_pos_dict[u])
        # sample negative not in pos set
        while True:
            n = random.randrange(num_items)
            if n not in user_pos_dict[u] and (exclude_user_pos_dict is None or n not in exclude_user_pos_dict[u]):
                break
        users.append(u)
        pos_items.append(p)
        neg_items.append(n)
    return (torch.tensor(users,     dtype=torch.long),
            torch.tensor(pos_items, dtype=torch.long),
            torch.tensor(neg_items, dtype=torch.long))

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

def train_val(model, num_items, num_epochs, batch_size, device, opt, train_inter, val_inter, train_edge_index, val_edge_index, patience, K):
    train_user_pos_dict = defaultdict(list)
    for u, i in train_inter:
        train_user_pos_dict[u].append(i)

    val_user_pos_dict = defaultdict(list)
    for u, i in val_inter:
        val_user_pos_dict[u].append(i)


    val_train_user_pos_dict = defaultdict(list)
    for u, i in val_inter + train_inter:
        val_train_user_pos_dict[u].append(i)
    loss_hist, val_loss_hist = [], []
    val_prec_hist, val_rec_hist, val_ndcg_hist = [], [], []

    best_precision = 0
    patience_counter = 0

    steps_per_epoch = int(np.ceil(len(train_inter) / batch_size))  # heuristic
    val_steps_per_epoch = int(np.ceil(len(val_inter) / batch_size))  # heuristic

    for epoch in range(1, num_epochs + 1):
        model.train()
        t0 = time.time()

        total_train_loss = 0.0
        total_val_loss = 0.0

        for _ in range(steps_per_epoch):
            u, p, n = sample_mini_batch(batch_size, train_user_pos_dict, num_items)
            u, p, n = u.to(device), p.to(device), n.to(device)
            opt.zero_grad()
            loss, _, _ = bpr_loss(model, u, p, n, train_edge_index)
            loss.backward()
            opt.step()
            total_train_loss += loss.item() * u.size(0)

        avg_train_loss = total_train_loss / (steps_per_epoch * batch_size)
        loss_hist.append(avg_train_loss)

        # ─── Validation ────────────────────────────────────────────────────────────
        model.eval()
        for _ in range(val_steps_per_epoch):
            u, p, n = sample_mini_batch(batch_size, val_user_pos_dict, num_items, train_user_pos_dict)
            u, p, n = u.to(device), p.to(device), n.to(device)
            with torch.no_grad():
                loss, _, _ = bpr_loss(model, u, p, n, train_edge_index)
            total_val_loss += loss.item() * u.size(0)

        avg_val_loss = total_val_loss / (val_steps_per_epoch * batch_size)
        val_loss_hist.append(avg_val_loss)

        
        with torch.no_grad():
            prec, rec, ndcg = precision_recall_ndcg_at_k(
                model, train_edge_index, val_inter, train_inter, K
            )
        val_prec_hist.append(prec)
        val_rec_hist.append(rec)
        val_ndcg_hist.append(ndcg)

        print(
            f"Epoch {epoch:02d} | {time.time() - t0:.1f}s | TrainLoss {avg_train_loss:.4f} | "
            f"ValLoss {avg_val_loss:.4f} | P@{K} {prec:.4f} R@{K} {rec:.4f} NDCG@{K} {ndcg:.4f}"
        )
        best_model_state = model.state_dict()
        # Early stopping -----------------------------------------------------------
        # Early stopping logic
        if prec > best_precision:
            best_precision = prec
            patience_counter = 0
            best_model_state = model.state_dict()  # Save best model
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                break
    model.load_state_dict(best_model_state)

def test(model, num_items, batch_size, device, train_inter, val_inter, test_inter, train_edge_index, full_edge_index, K):
    test_steps_per_epoch = int(np.ceil(len(test_inter) / batch_size))  # heuristic 
    total_test_loss = 0

    val_user_pos_dict = defaultdict(list)
    for u, i in val_inter:
        val_user_pos_dict[u].append(i)

    test_user_pos_dict = defaultdict(list)
    for u, i in test_inter:
        test_user_pos_dict[u].append(i)

    val_train_user_pos_dict = defaultdict(list)
    for u, i in val_inter + train_inter:
        val_train_user_pos_dict[u].append(i)
    for _ in range(test_steps_per_epoch):
        u, p, n = sample_mini_batch(batch_size, test_user_pos_dict, num_items, val_train_user_pos_dict)
        u, p, n = u.to(device), p.to(device), n.to(device)
        with torch.no_grad():
            loss, _, _ = bpr_loss(model, u, p, n, train_edge_index)
        total_test_loss += loss.item() * u.size(0)

    avg_test_loss = total_test_loss / (test_steps_per_epoch * batch_size)
    with torch.no_grad():
        prec_test, rec_test, ndcg_test = precision_recall_ndcg_at_k(
            model, train_edge_index, test_inter, train_inter+val_inter, K
        )
    print(
        f"\nFinal Test | P@{K} {prec_test:.4f} | R@{K} {rec_test:.4f} | "
        f"NDCG@{K} {ndcg_test:.4f} | BPR Loss {avg_test_loss:.4f}"
    )
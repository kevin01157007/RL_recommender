import random, time
from collections import defaultdict
from tqdm import trange
import torch, networkx as nx
import community.community_louvain as community_louvain
import numpy as np
def build_dtrain_graph(inter):

    B = nx.Graph()
    # 從輸入數據中提取唯一的用戶和項目ID
    users = sorted(set(pair[0] for pair in inter))  # 獲取所有唯一的用戶ID
    items = sorted(set(pair[1] for pair in inter))  # 獲取所有唯一的項目ID

    # 添加用戶節點
    for uid in users:
        B.add_node(f"u{uid}", bipartite=0)

    # 添加項目節點
    for mid in items:
        B.add_node(f"m{mid}", bipartite=1)

    # 添加邊 (使用者和項目之間的關係)
    for user, item in inter:
        B.add_edge(f"u{user}",  f"m{item}")
    
    return B

def community_cnt(g):
    part = community_louvain.best_partition(g, resolution=1.25, random_state=42)
    return len(set(part.values()))

def build_edge_index(G, NUM_USERS, DEVICE):
    edges = [(int(u[1:]), int(m[1:]) + NUM_USERS) for u, m in G.edges()]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous().to(DEVICE)
    return torch.cat([edge_index, edge_index[[1, 0], :]], dim=1)

def epsilon(step, EPS_START, EPS_END, EPS_DECAY_RATE):
    return EPS_END + (EPS_START - EPS_END) * np.exp(-EPS_DECAY_RATE * step)

def rl_exposure(agent, interaction, ITEM_EMB, USER_EMB, DEVICE,
                total_epidsode=16, EPS_START=1, EPS_END=0.01,
                EPS_DECAY_RATE=0.001, TOTAL_STEPS=100):
    random.seed(42)
    NUM_USERS = USER_EMB.shape[0]

    with torch.no_grad():
        score = USER_EMB @ ITEM_EMB.t()
    TOP_K = 100
    _, topk_idx = torch.topk(score, TOP_K, dim=1)
    CANDIDATES = topk_idx.cpu().tolist()
    print("Candidate pool ready (Top‑100)")

    exposure = []
    start = time.time()

    for episode in range(total_epidsode):
        G = build_dtrain_graph(interaction)
        C_now = community_cnt(G)
        if episode == 0:
            print(f"初始社群數: {C_now}")
        agent.step = 0

        cand_pairs = []
        for u in range(NUM_USERS):
            seen = set(G.neighbors(f"u{u}"))
            cands = [v for v in CANDIDATES[u] if f"m{v}" not in seen]
            cand_pairs.extend([(u, v) for v in cands])
        print("候選邊數量:", len(cand_pairs))

        cand_pairs_tensor = torch.tensor(cand_pairs, dtype=torch.long, device=DEVICE)

        for t in trange(TOTAL_STEPS, desc=f"RL‑train-{episode}"):
            if len(cand_pairs) == 0:
                print("候選邊用盡，提前結束")
                break

            edge_index = build_edge_index(G, NUM_USERS, DEVICE)

            u_batch = cand_pairs_tensor[:, 0]
            v_batch = cand_pairs_tensor[:, 1] + NUM_USERS

            idx = agent.select(edge_index, u_batch, v_batch, epsilon(t, EPS_START, EPS_END, EPS_DECAY_RATE))
            u, v = cand_pairs[idx]

            # 建立 next state 前先儲存 edge list
            G_prev_edges = list(G.edges())
            G.add_edge(f"u{u}", f"m{v}")
            G_next_edges = list(G.edges())

            if episode == total_epidsode - 1:
                exposure.append((u, v))

            edge_index_next = build_edge_index(G, NUM_USERS, DEVICE)

            # 計算 reward
            C_prev, C_new = C_now, community_cnt(G)
            reward = 1.0 / C_new - 1.0 / C_prev
            C_now = C_new

            done = (t == TOTAL_STEPS - 1)
            agent.store(
                state=edge_index,
                action=(u, v),
                reward=reward,
                next_state=edge_index_next,
                graph_prev=G_prev_edges,
                graph_next=G_next_edges,
                done=done
            )

            # 移除已選擇的 pair
            cand_pairs.pop(idx)
            cand_pairs_tensor = torch.tensor(cand_pairs, dtype=torch.long, device=DEVICE) if cand_pairs else torch.empty((0, 2), dtype=torch.long, device=DEVICE)

            if len(agent.buffer) >= 100:
                agent.update(u_batch, v_batch)

        print(f"Episode {episode} 結束後社群數: {C_now}")

    print(f"Done | Final community count = {C_now} | Elapsed {time.time() - start:.1f}s")
    return exposure

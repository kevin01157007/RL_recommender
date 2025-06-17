import random, time
from collections import defaultdict
from tqdm import trange
import pickle
import torch, networkx as nx
import community.community_louvain as community_louvain
import numpy as np
from utility.bulid_dtrain_graph import build_dtrain_graph
from model.dqn.rl_agent import RLAgent

import random

INTERACTIONS_PER_USER = 10

random.seed(42)
interactions = []

for user in range(5950):
    items = random.sample(range(3191), INTERACTIONS_PER_USER)
    for item in items:
        interactions.append((user, item))   # (user_id, item_id)


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ❶ Load fixed LightGCN embeddings
USER_EMB = torch.load("../user_emb.pt", map_location=DEVICE, weights_only=True)
ITEM_EMB = torch.load("../item_emb.pt", map_location=DEVICE, weights_only=True)

# ❷ Build initial graph and Louvain community count
G = build_dtrain_graph(interactions)
EMB_DIM = USER_EMB.size(1)
NUM_USERS = USER_EMB.shape[0]
NUM_ITEMS = ITEM_EMB.shape[0]
print(f"Embeddings  |  users={NUM_USERS}  items={NUM_ITEMS}  dim={EMB_DIM}")

def community_cnt(g):
    part = community_louvain.best_partition(g, resolution=2, random_state=42)
    return len(set(part.values()))

C_now = community_cnt(G)
print(f"Initial community count = {C_now}")

# ❸ Candidate pool (Top-100 per user)
with torch.no_grad():
    score = USER_EMB @ ITEM_EMB.t()
TOP_K = 100
_, topk_idx = torch.topk(score, TOP_K, dim=1)
CANDIDATES = topk_idx.cpu().tolist()
print("Candidate pool ready (Top‑100)")

# ❹ Instantiate RL agent
TOTAL_NODES = NUM_USERS + NUM_ITEMS + 1
agent = RLAgent(num_nodes=TOTAL_NODES, emb_dim=EMB_DIM, device=DEVICE)
with torch.no_grad():
    agent.q_net.node_emb.weight.data[:NUM_USERS] = USER_EMB
    agent.q_net.node_emb.weight.data[NUM_USERS:NUM_USERS+NUM_ITEMS] = ITEM_EMB
    agent.target_net.node_emb.weight.data.copy_(agent.q_net.node_emb.weight.data)

# ❺ ε‑greedy schedule
EPS_START, EPS_END, EPS_DECAY_RATE = 1, 0.01, 0.001
def epsilon(step):
    eps = EPS_END + (EPS_START - EPS_END) * np.exp(-EPS_DECAY_RATE*step)
    return eps

# ❻ Main loop
TOTAL_STEPS = 100
start = time.time()
last_edge_index = None
for episode in range(10):
    G = build_dtrain_graph(interactions)
    agent.step = 0
    u_all, v_all, cand_pairs = [], [], []
    for u in range(NUM_USERS):
        seen = set(G.neighbors(u))
        cands = [v for v in CANDIDATES[u] if (v + NUM_USERS) not in seen]
        u_all.extend([u] * len(cands))
        v_all.extend([v + NUM_USERS for v in cands])
        cand_pairs.extend([(u, v + NUM_USERS) for v in cands])

    for t in trange(TOTAL_STEPS, desc=f"RL‑train-{episode}"):

        # Build edge_index (這步做完代表當前狀態 s)
        edge_index = torch.tensor(list(G.edges()), dtype=torch.long).t().contiguous().to(DEVICE)
        edge_index = torch.cat([edge_index, edge_index[[1, 0], :]], dim=1)

        u_batch = torch.tensor(u_all, dtype=torch.long, device=DEVICE)
        v_batch = torch.tensor(v_all, dtype=torch.long, device=DEVICE)

        idx = agent.select(edge_index, u_batch, v_batch, epsilon(t))
        u, v = cand_pairs[idx] #v是原v+5950

        # 存 G (edge list) 做為 G_t，方便 buffer
        G_prev_edges = list(G.edges())

        # Execute action
        G.add_edge(u, v)

        # 下一狀態
        edge_index_next = torch.tensor(list(G.edges()), dtype=torch.long).t().contiguous().to(DEVICE)
        edge_index_next = torch.cat([edge_index_next, edge_index_next[[1, 0], :]], dim=1)
        
        G_next_edges = list(G.edges())

        # Reward
        C_prev, C_new = C_now, community_cnt(G)
        reward = 1.0 / C_new - 1.0 / C_prev
        C_now = C_new

        done = (t == TOTAL_STEPS - 1)
        agent.store(
            state=edge_index,                      # s
            action=(u, v),                         # a
            reward=reward,                         # r
            next_state=edge_index_next,             # s'
            graph_prev=G_prev_edges,               # G
            graph_next=G_next_edges,               # G'
            done=done
        )
        if len(agent.buffer) >= 100:
            agent.update(u_batch, v_batch)

print(f"Done | Final community count = {C_now} | elapsed {time.time()-start:.1f}s")

with open("graph_after_rl.gpickle", "wb") as f:
    pickle.dump(G, f)

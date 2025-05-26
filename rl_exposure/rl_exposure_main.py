import random, time
from collections import defaultdict
from tqdm import trange

import torch, networkx as nx
import community.community_louvain as community_louvain

from bulid_dtrain_graph import build_dtrain_graph
from rl_agent import RLAgent

import random

NUM_USERS = 5950
NUM_ITEMS = 3201
INTERACTIONS_PER_USER = 10

random.seed(42)
interactions = []

for user in range(NUM_USERS):
    items = random.sample(range(NUM_ITEMS), INTERACTIONS_PER_USER)
    for item in items:
        interactions.append((user, item))   # (user_id, item_id)


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ❶ Load fixed LightGCN embeddings
USER_EMB = torch.load("../user_emb.pt", map_location=DEVICE)
ITEM_EMB = torch.load("../item_emb.pt", map_location=DEVICE)

# ❷ Build initial graph and Louvain community count
G= build_dtrain_graph(interactions)
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
EPS_START, EPS_END, EPS_DECAY_STEPS = 0.2, 0.05, 5000
def epsilon(step):
    if step >= EPS_DECAY_STEPS:
        return EPS_END
    return EPS_START - (EPS_START-EPS_END) * step / EPS_DECAY_STEPS

# ❻ Main loop
TOTAL_STEPS = 100
start = time.time()
last_edge_index = None

for t in trange(TOTAL_STEPS, desc="RL‑train"):
    u = random.randint(0, NUM_USERS-1)
    seen = set(G.neighbors(u))
    cand = [i for i in CANDIDATES[u] if (i + NUM_USERS) not in seen]
    if not cand:
        continue

    # Build edge_index (這步做完代表當前狀態 s)
    edge_index = torch.tensor(list(G.edges()), dtype=torch.long).t().contiguous().to(DEVICE)
    edge_index = torch.cat([edge_index, edge_index[[1, 0], :]], dim=1)

    u_batch = torch.full((len(cand),), u, dtype=torch.long, device=DEVICE)
    v_batch = torch.tensor(cand, dtype=torch.long, device=DEVICE) + NUM_USERS

    idx = agent.select(edge_index, u_batch, v_batch, epsilon(t))
    v = cand[idx]

    # 存 G (edge list) 做為 G_t，方便 buffer
    G_prev_edges = list(G.edges())

    # Execute action
    G.add_edge(u, v + NUM_USERS)

    # 下一狀態
    edge_index_next = torch.tensor(list(G.edges()), dtype=torch.long).t().contiguous().to(DEVICE)
    G_next_edges = list(G.edges())

    # Reward
    C_prev, C_new = C_now, community_cnt(G)
    reward = 1.0 / C_new - 1.0 / C_prev
    C_now = C_new

    done = False  # 若有明確 episode 結束再設 True
    agent.store(
        state=edge_index,                      # s
        action=(u, v + NUM_USERS),             # a
        reward=reward,                         # r
        next_state=edge_index_next,             # s'
        graph_prev=G_prev_edges,               # G
        graph_next=G_next_edges,               # G'
        done=done
    )
    if len(agent.buffer) >= 100:
        agent.update(edge_index)

# 訓練結束，把 buffer 裡未寫入部分補齊（flush，確保 n-step buffer 不遺漏）
agent.store(
    state=None, action=None, reward=0, next_state=None, graph_prev=None, graph_next=None, done=True
)

print(f"Done | Final community count = {C_now} | elapsed {time.time()-start:.1f}s")

nx.write_gpickle(G, "graph_after_rl.gpickle")

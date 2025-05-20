# rl_exposure_main.py – Algorithm 1 with a **fully‑connected virtual node**
# ---------------------------------------------------------------------------------
# Compared with the previous version we **remove the mean‑pool z_s** and instead
# create a *virtual node s* that is connected to **all** real nodes.  We then run
# *one extra GCN layer* over this extended graph so the embedding of node s
# automatically aggregates global information (exactly what the paper describes).
#
# ➜ Code changes (synchronised with the new q_network.py / rl_agent.py you will
#   paste in the project folder):
#   • q_network.QNetwork internally builds the extended edge_index and returns
#     both z_all (real nodes) and z_s (virtual node) – therefore rl_agent no
#     longer needs an external z_s argument.
#   • RLAgent.select() / update() signatures become:  (edge_index, u_idx, v_idx, eps)
#
# Put this file next to q_network.py, rl_agent.py, etc.  Make sure you have run
# LightGCN_pretrain.py and have user_emb.pt / item_emb.pt ready.
# ---------------------------------------------------------------------------------
import random, time
from collections import defaultdict
from tqdm import trange

import torch, networkx as nx
import community.community_louvain as community_louvain

from build_nx_graph import build_nx_graph
from rl_agent import RLAgent

# ---------------------------------------------------------------------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ❶ Load fixed LightGCN embeddings ------------------------------------------------
USER_EMB = torch.load("user_emb.pt", map_location=DEVICE)
ITEM_EMB = torch.load("item_emb.pt", map_location=DEVICE)
EMB_DIM   = USER_EMB.size(1)
NUM_USERS = USER_EMB.size(0)
NUM_ITEMS = ITEM_EMB.size(0)
print(f"Embeddings  |  users={NUM_USERS}  items={NUM_ITEMS}  dim={EMB_DIM}")

# ❷ Build initial graph and Louvain community count --------------------------------
G = build_nx_graph()
assert G.number_of_nodes() == NUM_USERS + NUM_ITEMS

def community_cnt(g):
    part = community_louvain.best_partition(g, resolution=2, random_state=42)
    return len(set(part.values()))

C_now = community_cnt(G)
print(f"Initial community count = {C_now}")

# ❸ Candidate pool (Top‑100 per user) ---------------------------------------------
with torch.no_grad():
    score = USER_EMB @ ITEM_EMB.t()
TOP_K = 100
_, topk_idx = torch.topk(score, TOP_K, dim=1)
CANDIDATES = topk_idx.cpu().tolist()
print("Candidate pool ready (Top‑100)")

# ❹ Instantiate RL agent (new signature: node_count = N+1 for virtual node) -------
TOTAL_NODES = NUM_USERS + NUM_ITEMS + 1           # +1 for virtual node s
agent = RLAgent(num_nodes=TOTAL_NODES,
                emb_dim = EMB_DIM,
                device   = DEVICE)
# load LightGCN emb as initial weights
with torch.no_grad():
    agent.q_net.node_emb.weight.data[:NUM_USERS]            = USER_EMB
    agent.q_net.node_emb.weight.data[NUM_USERS:NUM_USERS+NUM_ITEMS] = ITEM_EMB
    agent.target_net.node_emb.weight.data.copy_(agent.q_net.node_emb.weight.data)

# ❺ ε‑greedy schedule --------------------------------------------------------------
EPS_START, EPS_END, EPS_DECAY_STEPS = 0.2, 0.05, 5000

def epsilon(step):
    if step >= EPS_DECAY_STEPS:
        return EPS_END
    return EPS_START - (EPS_START-EPS_END) * step / EPS_DECAY_STEPS

# ❻ Main loop ----------------------------------------------------------------------
TOTAL_STEPS = 20_000
start = time.time()
for t in trange(TOTAL_STEPS, desc="RL‑train"):
    u = random.randint(0, NUM_USERS-1)
    seen = set(G.neighbors(u))
    cand = [i for i in CANDIDATES[u] if (i + NUM_USERS) not in seen]
    if not cand:
        continue

    # Build edge_index *once per step* (cost is OK for medium‑size graphs)
    edge_index = torch.tensor(list(G.edges()), dtype=torch.long).t().contiguous().to(DEVICE)

    # Tensor index for candidate items
    u_batch = torch.full((len(cand),), u, dtype=torch.long, device=DEVICE)
    v_batch = torch.tensor(cand, dtype=torch.long, device=DEVICE) + NUM_USERS  # shift item id

    # ---- ε‑greedy select ---------------------------------------------------------
    idx = agent.select(edge_index, u_batch, v_batch, epsilon(t))
    v  = cand[idx]

    # ---- Execute action ----------------------------------------------------------
    G.add_edge(u, v + NUM_USERS)

    # ---- Reward ------------------------------------------------------------------
    C_prev, C_new = C_now, community_cnt(G)
    reward = 1.0 / C_new - 1.0 / C_prev     # fewer communities ⇒ positive
    C_now = C_new

    # ---- Store & learn ------------------------------------------------------------
    agent.store((u, v + NUM_USERS, reward))   # s 由網路內部自己算，不用額外傳
    agent.update(edge_index)

print(f"Done | Final community count = {C_now} | elapsed {time.time()-start:.1f}s")

# 保存結果 -------------------------------------------------------------------------
nx.write_gpickle(G, "graph_after_rl.gpickle")

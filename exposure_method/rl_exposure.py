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
    part = community_louvain.best_partition(g, resolution=1, random_state=42)
    return len(set(part.values()))

def build_edge_index(G, NUM_USERS, DEVICE):
    edges = [(int(u[1:]), int(m[1:]) + NUM_USERS) for u, m in G.edges()]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous().to(DEVICE)
    return torch.cat([edge_index, edge_index[[1, 0], :]], dim=1)

def epsilon(step, EPS_START, EPS_END, EPS_DECAY_RATE):
    return EPS_END + (EPS_START - EPS_END) * np.exp(-EPS_DECAY_RATE * step)

def rl_exposure(agent, interaction, ITEM_EMB, USER_EMB, DEVICE,
                total_epidsode=5, EPS_START=1, EPS_END=0.01,
                EPS_DECAY_RATE=0.005, TOTAL_STEPS=300):
    random.seed(42)
    NUM_USERS = USER_EMB.shape[0]

    # 創建日誌文件
    with open('training_log.txt', 'w', encoding='utf-8') as f:
        f.write("Training Log\n")
        f.write("=" * 50 + "\n")

    with torch.no_grad():
        score = USER_EMB @ ITEM_EMB.t()
    TOP_K = 100
    _, topk_idx = torch.topk(score, TOP_K, dim=1)
    CANDIDATES = topk_idx.cpu().tolist()
    print("Candidate pool ready (Top‑100)")

    exposure = []
    start = time.time()

    all_episode_avg_losses = []
    all_episode_total_rewards = []

    for episode in range(total_epidsode):
        current_episode_total_reward = 0.0
        current_episode_losses = []
        print(f"--- Starting RL Episode {episode}/{total_epidsode} ---")
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

            # G is the graph for the next state (already updated by adding an edge)
            # CANDIDATES is the list of top-100 items for each user (available in this scope)
            # NUM_USERS is available in this scope
            cand_pairs_for_s_next = []
            for u_idx_loop in range(NUM_USERS):
                # G.neighbors(f"u{u_idx_loop}") will give neighbor node strings like 'm123' or 'u456'
                # We need to handle cases where a user might not be in G yet if G is built progressively,
                # though build_dtrain_graph adds all users/items from initial interactions.
                # For safety, use G.has_node and check if it's an item neighbor.
                seen_items_in_G_next = set()
                if G.has_node(f"u{u_idx_loop}"):
                    for neighbor_node_str in G.neighbors(f"u{u_idx_loop}"):
                        if neighbor_node_str.startswith('m'):
                            seen_items_in_G_next.add(int(neighbor_node_str[1:]))

                user_specific_cands_for_s_next = [item_id for item_id in CANDIDATES[u_idx_loop] if item_id not in seen_items_in_G_next]
                for item_id_loop in user_specific_cands_for_s_next:
                    cand_pairs_for_s_next.append((u_idx_loop, item_id_loop))

            # 計算 reward
            C_prev, C_new = C_now, community_cnt(G)
            reward = float(C_prev - C_new)
            current_episode_total_reward += reward
            log_message = f"Reward: {reward}, C_prev: {C_prev}, C_new: {C_new}"
            # print(log_message)
            
            # 寫入日誌文件
            with open('training_log.txt', 'a', encoding='utf-8') as f:
                f.write(f"Episode {episode}, Step {t}: {log_message}\n")
            
            C_now = C_new

            done = (t == TOTAL_STEPS - 1)
            agent.store(
                state=edge_index,
                action=(u, v),
                reward=reward,
                next_state=edge_index_next,
                graph_prev=G_prev_edges,
                graph_next=G_next_edges,
                done=done,
                next_state_candidate_pairs=cand_pairs_for_s_next
            )

            # 移除已選擇的 pair
            cand_pairs.pop(idx)
            cand_pairs_tensor = torch.tensor(cand_pairs, dtype=torch.long, device=DEVICE) if cand_pairs else torch.empty((0, 2), dtype=torch.long, device=DEVICE)

            if len(agent.buffer) >= agent.batch:  # Check if buffer is ready for update
                loss = agent.update()
                if loss is not None:
                    current_episode_losses.append(loss)

        avg_loss_this_episode = sum(current_episode_losses) / len(current_episode_losses) if current_episode_losses else 0.0
        all_episode_avg_losses.append(avg_loss_this_episode)
        all_episode_total_rewards.append(current_episode_total_reward)
        # The epsilon printed should be the one for the last step of the episode
        final_epsilon_this_episode = epsilon(TOTAL_STEPS - 1, EPS_START, EPS_END, EPS_DECAY_RATE)
        print(f"Episode {episode} finished. Total Reward: {current_episode_total_reward:.4f}, Avg Loss: {avg_loss_this_episode:.4f}, Epsilon_end: {final_epsilon_this_episode:.4f}, Communities: {C_now}")
        # 寫入每個episode的總結
        with open('training_log.txt', 'a', encoding='utf-8') as f:
            f.write(f"\nEpisode {episode} Summary:\n")
            f.write(f"Total Reward: {current_episode_total_reward:.4f}\n")
            f.write(f"Average Loss: {avg_loss_this_episode:.4f}\n")
            f.write(f"Final Epsilon: {final_epsilon_this_episode:.4f}\n")
            f.write(f"Final Communities: {C_now}\n")
            f.write("-" * 50 + "\n")

    print("\n--- RL Training Summary ---")
    for i in range(len(all_episode_total_rewards)):
        print(f"Episode {i}: Total Reward = {all_episode_total_rewards[i]:.4f}, Avg Loss = {all_episode_avg_losses[i]:.4f}")
    avg_reward_overall = sum(all_episode_total_rewards) / len(all_episode_total_rewards) if all_episode_total_rewards else 0.0
    avg_loss_overall = sum(all_episode_avg_losses) / len(all_episode_avg_losses) if all_episode_avg_losses else 0.0

    # 寫入整體訓練總結
    with open('training_log.txt', 'a', encoding='utf-8') as f:
        f.write("\nOverall Training Summary\n")
        f.write("=" * 50 + "\n")
        for i in range(len(all_episode_total_rewards)):
            f.write(f"Episode {i}: Total Reward = {all_episode_total_rewards[i]:.4f}, Avg Loss = {all_episode_avg_losses[i]:.4f}\n")
        f.write(f"\nOverall Avg Reward per Episode: {avg_reward_overall:.4f}\n")
        f.write(f"Overall Avg Loss per Episode: {avg_loss_overall:.4f}\n")
        f.write(f"Final community count: {C_now}\n")
        f.write(f"Total training time: {time.time() - start:.1f}s\n")

    print(f"Done | Final community count = {C_now} | Elapsed {time.time() - start:.1f}s")
    return exposure

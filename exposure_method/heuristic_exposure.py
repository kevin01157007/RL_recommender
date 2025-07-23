import numpy as np
import networkx as nx
import torch
from sklearn.metrics.pairwise import cosine_similarity
import community.community_louvain as community_louvain
from collections import defaultdict

def calculate_ILS(item_emb, rec_item_set):
    """
    計算項目集合的內部列表相似度 (Intra-List Similarity)
    
    參數:
    item_emb: 項目嵌入矩陣，shape=(n_items, embedding_dim)
    rec_item_set: 用戶交互過的項目ID列表
    
    返回:
    ILS值，範圍在0到1之間，值越高表示項目集合越相似
    """
    if len(rec_item_set) <= 1:
        return 0.0
    
    # 獲取項目嵌入
    embeddings = item_emb[rec_item_set].cpu().detach().numpy()
    
    # 計算項目間的相似度
    sim_matrix = cosine_similarity(embeddings)
    
    # 去除對角線上的自我相似度
    np.fill_diagonal(sim_matrix, 0)
    
    # 計算ILS
    k = len(rec_item_set)
    ils = np.sum(sim_matrix) / (k * (k - 1))
    
    return ils

def heuristic_exposure_strategy(dtrain_user_item_graph ,user_item_graph, rec_item_set, item_emb, 
                               n_selected_communities=2,  # 選擇的社群數量
                               n_diverse_users_per_community=20,  # 每個社群中選擇的多樣化用戶數量
                               n_items_per_user=20):  # 每個用戶選擇的項目數量
    """
    
    參數:
    user_item_graph: 用戶-項目二分圖
    rec_item_set: 被推薦項目列表
    item_emb: 項目嵌入矩陣
    n_selected_communities: 每次選擇的社群數量
    n_diverse_users_per_community: 每個社群選擇的最多樣化用戶數量
    n_items_per_user: 每個用戶選擇的項目數量
    
    返回:
    selected_items: 要額外曝光的用戶-項目邊列表 [(user_id, item_id), ...]
    """
    # 使用Louvain算法檢測社群
    print("正在檢測社群...")

    # 分別對每個連通元件進行 Louvain，並整合
    connected_components = list(nx.connected_components(dtrain_user_item_graph))
    print(f"總共有 {len(connected_components)} 個連通元件")

    all_partition = community_louvain.best_partition(dtrain_user_item_graph, resolution=1, random_state=42)

    communities = all_partition
    print(f"總社群數量: {len(set(communities.values()))}")
    
    # 將節點分為用戶節點和項目節點
    users = [node for node in dtrain_user_item_graph.nodes() if dtrain_user_item_graph.nodes[node].get('bipartite') == 0]
    items = [node for node in dtrain_user_item_graph.nodes() if dtrain_user_item_graph.nodes[node].get('bipartite') == 1]
    # 為每個社群分配用戶和項目
    community_users = defaultdict(list)
    community_items = defaultdict(list)
    
    for node, community_id in communities.items():
        if node in users:
            community_users[community_id].append(node)
        elif node in items:
            community_items[community_id].append(node)
    
    valid_communities = [comm_id for comm_id in set(communities.values())
                        if len(community_users[comm_id]) > 0 and len(community_items[comm_id]) > 0]
    print("\nTop 社群資訊（前10）:")
    from collections import Counter
    comm_sizes = Counter(communities.values())
    for comm_id, size in comm_sizes.most_common(10):
        n_users = len(community_users[comm_id])
        n_items = len(community_items[comm_id])
        print(f"社群 {comm_id}: 總節點={size}, 使用者={n_users}, 電影={n_items}")

    
    if len(valid_communities) < 2:
        print("警告: 有效社群數量不足，無法進行跨社群曝光")
        return []
    
    print(f"有效社群數量: {len(valid_communities)}")
    
    
    # 隨機選擇兩個不同的社群
    if len(valid_communities) < n_selected_communities:
        print(f"警告: 有效社群數量({len(valid_communities)})小於請求數量({n_selected_communities})")
        selected_indices = valid_communities
    else:
        selected_indices = np.random.choice(valid_communities, 
                                            size=n_selected_communities, 
                                            replace=False)
        

    all_selected_items = []
    # 從被選到的社群創建曝光邊
    for i in range(len(selected_indices)):
        source_idx = selected_indices[i]

        # 計算社群中每個用戶的交互項目集合的多樣性得分(ILS)
        user_diversity_scores = {}
        for user in community_users[source_idx]:
            user = ''.join(filter(str.isdigit, user))
            user_items = rec_item_set[int(user)]
            if len(user_items) > 1:
                user_diversity_scores[user] = calculate_ILS(item_emb, user_items)
        
        if not user_diversity_scores:
            continue
        
        # 選擇ILS最低的用戶(最多樣化的用戶)
        diverse_users = sorted(user_diversity_scores.keys(), 
                                key=lambda u: user_diversity_scores[u])
        diverse_users = diverse_users[:min(n_diverse_users_per_community, len(diverse_users))]

        # 為每個多樣化用戶選擇歷史交互項目
        for user in diverse_users:
            neighbor_items = user_item_graph.neighbors(f"u{user}")
            # 選擇與用戶不同社群的項目
            user_items = [
                item for item in neighbor_items
                if item in communities and communities[item] != source_idx and item not in dtrain_user_item_graph.neighbors(f"u{user}")
            ]
            
            if len(user_items) == 0:
                continue
            
            # 隨機選擇一些項目
            selected_items = np.random.choice(user_items, 
                                            size=min(n_items_per_user, len(user_items)), 
                                            replace=False)
            for item_node_id in selected_items:
                item_node_id = ''.join(filter(str.isdigit, item_node_id))
                all_selected_items.append((int(user), int(item_node_id)))  

    print(f"已生成 {len(all_selected_items)} 條曝光邊")
    return all_selected_items
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# user_embeddings = torch.load("model/simulator/user_emb.pt", map_location=device, weights_only=True)
# item_embeddings = torch.load("model/simulator/item_emb.pt", map_location=device, weights_only=True)
# rec_item_set = [np.random.choice(range(item_embeddings.shape[0]), size=20, replace=False).tolist() for user in range(user_embeddings.shape[0])]
# user_item_graph = build_nx_graph()
# s = heuristic_exposure_strategy(user_item_graph, rec_item_set, item_embeddings)
# print(s)

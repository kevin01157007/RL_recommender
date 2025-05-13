import pandas as pd
import networkx as nx
import community.community_louvain as community_louvain
import matplotlib.pyplot as plt
import time
import numpy as np
# --- 1. 載入資料 ---
# 設定 MovieLens 1M 資料集的路徑 (請修改成你的實際路徑)

ratings_file = "data_split/train.dat"

# 讀取 ratings.dat，注意分隔符是 '::'
# 使用者ID::電影ID::評分::時間戳
try:
    ratings = pd.read_csv(ratings_file,
                        sep=',',
                        engine='python', # 需要 python engine 來處理 '::'
                        #   header=None,
                        names=['UserID', 'MovieID', 'Rating', 'Timestamp'],
                        encoding='ISO-8859-1') # 有些檔案可能需要指定編碼
except FileNotFoundError:
    print(f"錯誤：找不到檔案 {ratings_file}")
    print("請確認 ml_1m_path 是否設定正確，且 ratings.dat 存在於該路徑下。")
    exit()
ratings = ratings[ratings.Rating > 3]

print(f"成功載入 {len(ratings)} 筆評分資料。")
print(ratings.head())

# --- 2. 建立使用者-電影 雙邊圖 (Bipartite Graph) ---
print("\n建立使用者-電影 雙邊圖...")
start_time = time.time()

B = nx.Graph()

uid_map = {old: new for new, old in enumerate(sorted(ratings.UserID.unique()))}
mid_map = {old: new for new, old in enumerate(sorted(ratings.MovieID.unique()))}


ratings['UserID'] = ratings.UserID.map(uid_map)
ratings['MovieID'] = ratings.MovieID.map(mid_map)

# 添加節點，並標記節點類型 (bipartite=0 for users, bipartite=1 for movies)
users = sorted(ratings['UserID'].unique())
movies = sorted(ratings['MovieID'].unique())
# print(movies)
# # print(users)
# print(movies)
# 添加用戶節點
for i in range(len(users)):
    B.add_node(i, bipartite=0)

# 添加項目節點
for i in range(len(users),len(users)+len(movies)):
    B.add_node(i, bipartite=1)

# 添加邊 (使用者和電影之間的評分代表一條邊)
# 我們這裡不考慮評分值作為權重，僅代表有互動
movie_offset = len(users)
movie_id_map = {m: movie_offset + idx for idx, m in enumerate(movies)}
# print(movie_id_map)

edges = [(row['UserID'],
        movie_id_map[row['MovieID']])
        for _, row in ratings.iterrows()]
# print(edges[:10])
B.add_edges_from(edges)
end_time = time.time()
print(f"圖建立完成。耗時: {end_time - start_time:.2f} 秒")
print(f"節點數量: {B.number_of_nodes()} (Users: {len(users)}, Movies: {len(movies)})")
print(f"邊數量: {B.number_of_edges()}")

start_time = time.time()

# 檢查圖是否連通，Louvain 通常在連通圖或最大連通元件上執行
if nx.is_connected(B):
    print("圖是連通的。")
    graph_to_process = B
else:
    print("圖不是連通的。將在最大的連通元件上執行 Louvain。")
    # 找到最大的連通元件
    largest_cc = max(nx.connected_components(B), key=len)
    graph_to_process = B.subgraph(largest_cc).copy() # 使用子圖
    # 找出所有節點
    all_nodes = set(B.nodes())

    # 找出最大連通元件的節點
    largest_cc_nodes = set(largest_cc)

    # 取差集：這些是沒有被包含進最大連通元件的節點
    excluded_nodes = all_nodes - largest_cc_nodes

    print(f"\n不在最大連通元件中的節點數量：{len(excluded_nodes)}")
    print("其中前 10 個節點為：")
    print(list(excluded_nodes)[:10])
    print("\n前 10 個被排除節點的類型：")
    for node in list(excluded_nodes)[:10]:
        node_type = B.nodes[node].get('bipartite')
        type_str = "User" if node_type == 0 else "Movie"
        print(f"節點 {node} 類型: {type_str}")
    print(f"最大連通元件包含 {graph_to_process.number_of_nodes()} 個節點 和 {graph_to_process.number_of_edges()} 條邊。")

# 使用 best_partition 找到最佳的社群劃分
# 注意：python-louvain 的 best_partition 會將圖視為 unipartite 進行處理
# 它會回傳一個字典 {node: community_id}
partition = community_louvain.best_partition(graph_to_process, resolution=2, random_state=42)

end_time = time.time()
print(f"Louvain 執行完成。耗時: {end_time - start_time:.2f} 秒")

# --- 4. 分析結果 ---
num_communities = len(set(partition.values()))
print(f"\n偵測到的社群數量: {num_communities}")

# 計算模組度 (Modularity) - 衡量社群劃分品質的指標
modularity = community_louvain.modularity(partition, graph_to_process)
print(f"社群劃分的模組度 (Modularity): {modularity:.4f}")

# (可選) 顯示部分節點的社群歸屬
print("\n部分節點的社群歸屬範例:")
count = 0
for node, community_id in partition.items():
    node_type = "User" if graph_to_process.nodes[node]['bipartite'] == 0 else "Movie"
    print(f"  節點 {node} (類型: {node_type}) 屬於社群 {community_id}")
    count += 1
    if count >= 10: # 只顯示前 10 個
        break

# (可選) 分析每個社群的大小和組成
community_sizes = {}
community_users = {}
community_movies = {}
for node, comm_id in partition.items():
    community_sizes[comm_id] = community_sizes.get(comm_id, 0) + 1
    if graph_to_process.nodes[node]['bipartite'] == 0: # User
        community_users[comm_id] = community_users.get(comm_id, 0) + 1
    else: # Movie
        community_movies[comm_id] = community_movies.get(comm_id, 0) + 1

print("\n社群大小分佈 (Top 10):")
sorted_sizes = sorted(community_sizes.items(), key=lambda item: item[1], reverse=True)
for i, (comm_id, size) in enumerate(sorted_sizes[:10]):
     users_in_comm = community_users.get(comm_id, 0)
     movies_in_comm = community_movies.get(comm_id, 0)
     print(f"  社群 {comm_id}: 大小={size}, 使用者={users_in_comm}, 電影={movies_in_comm}")


# # (可選) 視覺化 (對於 MovieLens 1M 這種大小的圖可能很慢且混亂)
# print("\n(可選) 嘗試繪製圖與社群... (可能需要較長時間)")
# try:
#     pos = nx.spring_layout(graph_to_process, k=0.1, iterations=20) # 計算節點位置，耗時
#     plt.figure(figsize=(15, 15))
#     # 根據社群 ID 著色
#     cmap = plt.cm.get_cmap('viridis', num_communities)
#     nx.draw_networkx_nodes(graph_to_process, pos, partition.keys(), node_size=10,
#                            cmap=cmap, node_color=list(partition.values()))
#     nx.draw_networkx_edges(graph_to_process, pos, alpha=0.1)
#     plt.title("Louvain Communities in MovieLens 1M (Largest Component)")
#     plt.axis('off')
#     # plt.savefig("movielens_louvain.png") # 可以取消註解來儲存圖片
#     plt.show()
# except MemoryError:
#      print("記憶體不足，無法繪製完整圖形。")
# except Exception as e:
#      print(f"繪圖時發生錯誤: {e}")
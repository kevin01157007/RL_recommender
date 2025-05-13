import networkx as nx
import matplotlib.pyplot as plt
from community import community_louvain
import random

# 生成非連通圖（兩個完全獨立的社區）
n_nodes = 30
B = nx.Graph()
B.add_nodes_from(range(n_nodes))

# 社區1（節點0-14）
community1 = range(0, 15)
for u in community1:
    for v in community1:
        if u < v and random.random() < 0.3:
            B.add_edge(u, v, weight=1)

# 社區2（節點15-29）
community2 = range(15, 30)
for u in community2:
    for v in community2:
        if u < v and random.random() < 0.3:
            B.add_edge(u, v, weight=1)


# 確認圖是非連通的
print("圖是否連通:", nx.is_connected(B))  # 輸出 False
connected_components = list(nx.connected_components(B))
print(f"總共有 {len(connected_components)} 個連通元件")
# Louvain 檢測社區
partition = community_louvain.best_partition(B)

# 輸出社區劃分
from collections import defaultdict
community_dict = defaultdict(list)
for node, comm_id in partition.items():
    community_dict[comm_id].append(node)

print("\n社區劃分結果:")
for comm_id, nodes in community_dict.items():
    print(f"社區 {comm_id}: {nodes}")

# 可視化
pos = nx.spring_layout(B, seed=42)
plt.figure(figsize=(10, 6))
nx.draw_networkx_nodes(B, pos, node_size=100, cmap=plt.cm.tab20,
                       node_color=list(partition.values()))
nx.draw_networkx_edges(B, pos, alpha=0.5)
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.title("非連通圖的 Louvain 社區檢測")
plt.show()
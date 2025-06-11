import networkx as nx
import time

def build_dtrain_graph(inter):
    # --- 2. 建立使用者-電影 雙邊圖 (Bipartite Graph) ---
    print("\n建立Dtrain使用者-電影 雙邊圖...")
    start_time = time.time()

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

    end_time = time.time()
    print(f"圖建立完成。耗時: {end_time - start_time:.2f} 秒")
    print(f"節點數量: {B.number_of_nodes()} (Users: {len(users)}, Items: {len(items)})")
    print(f"邊數量: {B.number_of_edges()}")

    # 檢查圖是否連通
    if nx.is_connected(B):
        print("圖是連通的。")
    else:
        print("圖不是連通的。將個別偵測社群")
    
    return B
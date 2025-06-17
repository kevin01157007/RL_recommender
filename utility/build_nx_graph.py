import pandas as pd
import networkx as nx
# import time
def build_nx_graph():
    train_df = pd.read_csv('data/train.dat', sep=',', names=['UserID', 'MovieID', 'rating', 'timestamp'])
    val_df = pd.read_csv('data/val.dat', sep=',', names=['UserID', 'MovieID', 'rating', 'timestamp'])
    test_df = pd.read_csv('data/test.dat', sep=',', names=['UserID', 'MovieID', 'rating', 'timestamp'])

    try:
        ratings = pd.concat([train_df, val_df, test_df])
    except FileNotFoundError:
        print(f"錯誤：找不到檔案")
        print("請確認 ml_1m_path 是否設定正確，且 ratings.dat 存在於該路徑下。")
        exit()

    # print(f"成功載入 {len(ratings)} 筆評分資料。")
    # print(ratings.head())

    # --- 2. 建立使用者-電影 雙邊圖 (Bipartite Graph) ---
    # print("\n建立使用者-電影 雙邊圖...")
    # start_time = time.time()

    B = nx.Graph()

    # 添加節點，並標記節點類型 (bipartite=0 for users, bipartite=1 for movies)
    users = sorted(ratings['UserID'].unique())
    movies = sorted(ratings['MovieID'].unique())
    # print(movies)
    # # print(users)
    # print(movies)
    for uid in users:
        B.add_node(f"u{uid}", bipartite=0)

    for mid in movies:
        B.add_node(f"m{mid}", bipartite=1)


    edges = [(f"u{row['UserID']}", f"m{row['MovieID']}")
            for _, row in ratings.iterrows()]
    # print(edges[:10])
    B.add_edges_from(edges)
    # end_time = time.time()
    # print(f"圖建立完成。耗時: {end_time - start_time:.2f} 秒")
    # print(f"節點數量: {B.number_of_nodes()} (Users: {len(users)}, Movies: {len(movies)})")
    # print(f"邊數量: {B.number_of_edges()}")

    # start_time = time.time()

    # 檢查圖是否連通
    # if nx.is_connected(B):
    #     print("圖是連通的。")
    # else:
    #     print("圖不是連通的。將個別偵測社群")
    return B
# user_item_graph = build_nx_graph()
# if not user_item_graph.has_edge(0, 6890):
#     print(f"錯誤")
# else:
#     print("正常")
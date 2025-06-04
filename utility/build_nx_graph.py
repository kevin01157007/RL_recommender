import pandas as pd
import networkx as nx
import time
def build_nx_graph():
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

    # 檢查圖是否連通
    if nx.is_connected(B):
        print("圖是連通的。")
    else:
        print("圖不是連通的。將個別偵測社群")
    return B
# user_item_graph = build_nx_graph()
# if not user_item_graph.has_edge(0, 6890):
#     print(f"錯誤")
# else:
#     print("正常")
import pandas as pd

# ────────────────────────────────────────────────────────────────────────────────
# 1. Load ratings / movies / users (MovieLens‑1M)
# ────────────────────────────────────────────────────────────────────────────────
ratings = pd.read_csv('data_split/train.dat', sep=',', names=['user_id', 'movie_id', 'rating', 'timestamp'], engine='python', encoding='latin-1')
movies  = pd.read_csv('raw/ml-1m/movies.dat', sep='::', names=['movie_id', 'title', 'genres'],   engine='python', encoding='latin-1')
users   = pd.read_csv('raw/ml-1m/users.dat',  sep='::', names=['user_id', 'gender', 'age', 'occupation', 'zip'], engine='python', encoding='latin-1')

uid_map = {old: new for new, old in enumerate(sorted(ratings.user_id.unique()))}
mid_map = {old: new for new, old in enumerate(sorted(ratings.movie_id.unique()))}

# 反向映射 uid_map
reverse_uid_map = {new: old for old, new in uid_map.items()}

# 反向映射 mid_map
reverse_mid_map = {new: old for old, new in mid_map.items()}

ratings['user_id'] = ratings['user_id'].map(uid_map)
ratings['movie_id'] = ratings['movie_id'].map(mid_map)
# print(reverse_uid_map[948])
# print(reverse_mid_map[2037])

# print(uid_map[4277])
# print(mid_map[1193])

def batch_check_interactions(pairs, ratings):
    """
    批量檢查 ratings 中是否存在每個 (user_id, movie_id) 評分紀錄。

    參數:
        pairs (list of tuple): [(user_id1, movie_id1), (user_id2, movie_id2), ...]
        ratings (pd.DataFrame): 評分資料，含原始 user_id 和 movie_id

    回傳:
        list of bool: 對應每個 pair，若存在則為 True，否則 False
    """
    ratings_set = set(zip(ratings.user_id, ratings.movie_id))  # 建立快速查詢集合
    return [(u, i) in ratings_set for (u, i) in pairs]

n = [(4334, 873), (4334, 1003), (4334, 3113), (4795, 3063), (4795, 971), (4795, 996), (651, 1014), (651, 1207), (651, 1395), (3460, 2943), (3460, 2921), (3460, 930), (256, 1099), (256, 13), (256, 2521), (504, 1521), (504, 2240), (504, 2396), (162, 1814), (162, 1014), (162, 2806), (3125, 994), (3125, 1011), (3125, 2968), (519, 2006), (519, 2885), (519, 1481), (1691, 187), (1691, 1159), (1691, 2809), (3347, 1870), (3347, 545), (3347, 2841), (5278, 2056), (5278, 725), (5278, 1336), (5874, 1491), (5874, 2575), (5874, 2752)]
print(batch_check_interactions(n,ratings))

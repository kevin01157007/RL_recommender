import pandas as pd


ratings = pd.read_csv('../raw/ml-1m/ratings.dat', sep='::', names=['user_id', 'movie_id', 'rating', 'timestamp'], engine='python', encoding='latin-1')
movies  = pd.read_csv('../raw/ml-1m/movies.dat', sep='::', names=['movie_id', 'title', 'genres'],   engine='python', encoding='latin-1')
users   = pd.read_csv('../raw/ml-1m/users.dat',  sep='::', names=['user_id', 'gender', 'age', 'occupation', 'zip'], engine='python', encoding='latin-1')
# # 先過濾出評分大於 4 的評分記錄
# ratings = ratings[ratings['rating'] > 3]
# # print(len(ratings.user_id.unique()))
# # print(len(ratings.movie_id.unique()))
# # 計算每個使用者的評分數量
# user_rating_counts = ratings['user_id'].value_counts()

# movie_rating_counts = ratings['movie_id'].value_counts() #計算每部電影被評分幾次


# valid_users = user_rating_counts[user_rating_counts > 9].index # 獲取評分數量大於等於 10 的使用者 ID

# valid_movies = movie_rating_counts[movie_rating_counts > 5].index # 獲取被評分數量大於等於 4 的電影 ID

ratings_filtered = ratings[ratings['rating'] > 3].copy()
while True:
    old_shape = ratings_filtered.shape[0]
    user_counts = ratings_filtered['user_id'].value_counts() 
    movie_counts = ratings_filtered['movie_id'].value_counts()

    valid_users = user_counts[user_counts > 9].index # 獲取評分數量大於 9 的使用者 ID
    valid_movies = movie_counts[movie_counts > 3].index # 獲取被評分數量大於 3 的電影 ID

    ratings_filtered = ratings_filtered[
        ratings_filtered['user_id'].isin(valid_users) &
        ratings_filtered['movie_id'].isin(valid_movies)
    ]
    if ratings_filtered.shape[0] == old_shape:
        break

uid_map = {old: new for new, old in enumerate(sorted(ratings_filtered.user_id.unique()))}
mid_map = {old: new for new, old in enumerate(sorted(ratings_filtered.movie_id.unique()))}
ratings_filtered['user_id'] = ratings_filtered.user_id.map(uid_map)
ratings_filtered['movie_id'] = ratings_filtered.movie_id.map(mid_map)
# 使用這些有效的使用者 ID 來過濾原始的評分數據
# ratings_filtered = ratings[(ratings['user_id'].isin(valid_users)) & (ratings['movie_id'].isin(valid_movies))].copy()
print(len(ratings_filtered.user_id.unique()))
print(len(ratings_filtered.movie_id.unique()))
inter_df = ratings_filtered[['user_id', 'movie_id', 'rating', 'timestamp']].sort_values(['user_id','timestamp'])
test_indices = []

# 每位使用者個別處理
for uid, group in inter_df.groupby('user_id'):
    n = len(group)
    k = max(1, int(n * 0.2))  # 至少取 1 筆
    test_indices.extend(group.tail(k).index)

test_idx = pd.Index(test_indices)
test_df = inter_df.loc[test_idx].reset_index(drop=True)

train_val_df = inter_df.drop(test_idx).reset_index(drop=True)

# val: 每位 user 最後 20%
val_indices = []
for uid, group in train_val_df.groupby('user_id'):
    n = len(group)
    k = max(1, int(n * 0.1))
    val_indices.extend(group.tail(k).index)

val_idx = pd.Index(val_indices)
val_df = train_val_df.loc[val_idx].reset_index(drop=True)

train_df = train_val_df.drop(val_idx).reset_index(drop=True)

test_df.to_csv("test.dat", sep=',', index=False, header=False)
val_df.to_csv("val.dat", sep=',', index=False, header=False)
train_df.to_csv("train.dat", sep=',', index=False, header=False)

# print(test_df.head())
# print(train_df.head())

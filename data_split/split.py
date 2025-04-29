import pandas as pd


ratings = pd.read_csv('../raw/ml-1m/ratings.dat', sep='::', names=['user_id', 'movie_id', 'rating', 'timestamp'], engine='python', encoding='latin-1')
movies  = pd.read_csv('../raw/ml-1m/movies.dat', sep='::', names=['movie_id', 'title', 'genres'],   engine='python', encoding='latin-1')
users   = pd.read_csv('../raw/ml-1m/users.dat',  sep='::', names=['user_id', 'gender', 'age', 'occupation', 'zip'], engine='python', encoding='latin-1')

# 先過濾出評分大於 4 的評分記錄
ratings_above_4 = ratings[ratings['rating'] > 4]

# 計算每個使用者的評分數量
user_rating_counts = ratings_above_4['user_id'].value_counts()

# 獲取評分數量大於等於 10 的使用者 ID
valid_users = user_rating_counts[user_rating_counts > 10].index

# 使用這些有效的使用者 ID 來過濾原始的評分數據
ratings_filtered = ratings[ratings['user_id'].isin(valid_users)].copy()
inter_df = ratings_filtered[['user_id', 'movie_id', 'rating', 'timestamp']].sort_values(['user_id','timestamp'])
test_idx       = inter_df.groupby('user_id').tail(1).index
test_df        = inter_df.loc[test_idx].reset_index(drop=True)
test_df.to_csv("test.dat", sep=',', index=False)
train_df   = ratings_filtered.drop(test_idx).reset_index(drop=True)
train_df.to_csv("train.dat", sep=',', index=False)
# print(test_df.head())
# print(train_df.head())

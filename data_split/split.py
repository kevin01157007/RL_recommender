import pandas as pd


ratings = pd.read_csv('raw/ml-1m/ratings.dat', sep='::', names=['user_id', 'movie_id', 'rating', 'timestamp'], engine='python', encoding='latin-1')
movies  = pd.read_csv('raw/ml-1m/movies.dat', sep='::', names=['movie_id', 'title', 'genres'],   engine='python', encoding='latin-1')
users   = pd.read_csv('raw/ml-1m/users.dat',  sep='::', names=['user_id', 'gender', 'age', 'occupation', 'zip'], engine='python', encoding='latin-1')

inter_df = ratings[['user_id', 'movie_id', 'rating', 'timestamp']].sort_values(['user_id','timestamp'])
test_idx       = inter_df.groupby('user_id').tail(1).index
test_df        = inter_df.loc[test_idx].reset_index(drop=True)
test_df.to_csv("test.dat", sep=',', index=False)
train_df   = ratings.drop(test_idx).reset_index(drop=True)
train_df.to_csv("train.dat", sep=',', index=False)
print(test_df.head())
print(train_df.head())

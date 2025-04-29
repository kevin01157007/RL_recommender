import pandas as pd

# 讀取 rating.dat 文件，指定第一行為標題
df_ratings = pd.read_csv('train.dat', header=0)

# 計算每個使用者的評分數量
user_rating_counts = df_ratings['user_id'].value_counts()

# 過濾出評分不超過 k 的使用者
k = 10
users_with_few_ratings = user_rating_counts[user_rating_counts == k]

# 顯示這些使用者的 ID 和評分數量
print(f"評分不超過 {k} 的使用者:")
print(users_with_few_ratings)
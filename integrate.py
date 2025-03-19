import pandas as pd

# 讀取用戶資料
users = pd.read_csv('./dataset/ml-1m/ml-1m.user', sep='\t', header=None, engine='python', names=['userId', 'gender', 'age', 'occupation', 'zip'])

# 讀取電影資料
items = pd.read_csv('./dataset/ml-1m/ml-1m.item', sep='\t', header=None, engine='python', names=['itemId', 'title', 'genres'])

# 讀取互動資料
interactions = pd.read_csv('./dataset/ml-1m/ml-1m.inter', sep='\t', header=None, engine='python', names=['userId', 'itemId', 'rating', 'timestamp'])

# 設定評分閾值，篩選正向偏好
threshold = 4
interactions = interactions[interactions['rating'] >= threshold]

# 顯示整合後的資料
print(interactions.head())

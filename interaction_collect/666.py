import pandas as pd

# 讀檔
df = pd.read_csv('dtest.csv')  # 路徑相對於專案根目錄

# 建立 u -> [i, i, ...] 的映射
user_to_items = df.groupby('u')['i'].first()

# 查看 u=0 的所有 item
print(user_to_items.get(569))
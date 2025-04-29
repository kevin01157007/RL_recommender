import pandas as pd

# 讀取 test.dat 文件
df = pd.read_csv('test.dat', header=0)

# 創建應有的使用者 ID 列表
expected_users = set(range(1, 6041))  # 假設使用者 ID 從 1 到 6040

# 獲取實際的使用者 ID
actual_users = set(df['user_id'])

# 找出缺少的使用者
missing_users = expected_users - actual_users

# 顯示缺少的使用者
print("缺少的使用者 ID:", missing_users)
# 缺少的使用者 ID: {4486, 3598}
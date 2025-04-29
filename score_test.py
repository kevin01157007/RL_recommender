from typing import List

import numpy as np
import pandas as pd

import torch

# ... existing code ...
# score = simulator.score(30,1101)
# print(score)
user_emb = torch.load("user_emb.pt", map_location="cpu")   # [num_users, emb_dim]
item_emb = torch.load("item_emb.pt", map_location="cpu")   # [num_items, emb_dim]

# 2. 指定想看的 user / item index
u_idx = 1416
i_idx = 1176

# 3. 單一 (u, i) 的打分 = 兩向量點積
score = torch.matmul(user_emb[u_idx], item_emb.t())
score_df = pd.DataFrame(score.detach().numpy())
print(torch.topk(score,10))
# print(score_df.describe())
# score_df.to_csv("scores.csv", index=False)
# print(score)
# # 将 score 转换为 DataFrame，确保列名与评分数量匹配
# score_df = pd.DataFrame(score.detach().cpu().numpy(), columns=[f"Score_{i}" for i in range(score.shape[1])])

# # 将 DataFrame 写入 CSV 文件
# score_df.to_csv("scores.csv", index=False)  # 不写入行索引


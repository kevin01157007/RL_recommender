import collections
import math
import os
import os.path as osp
from tqdm import tqdm
from typing import List
import random
import time
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
pd.options.display.max_rows = 10
from sklearn import metrics
from tensorly import decomposition

import torch

# ... existing code ...
# score = simulator.score(30,1101)
# print(score)
user_emb = torch.load("user_emb.pt", map_location="cpu")   # [num_users, emb_dim]
item_emb = torch.load("item_emb.pt", map_location="cpu")   # [num_items, emb_dim]

# 2. 指定想看的 user / item index
u_idx = 2  # 第 0 個使用者
i_idx = 1241  # 第 1 部電影

# 3. 單一 (u, i) 的打分 = 兩向量點積
score = torch.dot(user_emb[u_idx], item_emb[i_idx])
print(score)   # .item() 變成 Python float
# # 将 score 转换为 DataFrame，确保列名与评分数量匹配
# score_df = pd.DataFrame(score.detach().cpu().numpy(), columns=[f"Score_{i}" for i in range(score.shape[1])])

# # 将 DataFrame 写入 CSV 文件
# score_df.to_csv("scores.csv", index=False)  # 不写入行索引


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
from torch.functional import tensordot
from torch import nn, optim, Tensor
import torch_geometric
from torch_geometric.data import Dataset, Data, download_url, extract_zip
from torch_geometric.nn import MessagePassing
from torch_geometric.typing import Adj
from LightGCNSimulator import LightGCNSimulator
from LightGCN import LightGCN
from LightGCNConv import LightGCNConv
from torch_geometric.data import Data, Dataset
from MovieLens import MovieLens
def personalized_topk(pred, K, user_indices, edge_index):
    """Computes TopK precision and recall.

    Args:
        pred: Predicted similarities between user and item.
        K: Number of items to rank.
        user_indices: Indices of users for each prediction in `pred`.
        edge_index: User and item connection matrix.

    Returns:
        Average Top K precision and recall for users in `user_indices`.
    """
    per_user_preds = collections.defaultdict(list)
    for index, user in enumerate(user_indices):
        # 获取该用户的前 K 个评分
        top_k_scores, _ = torch.topk(pred[index], K)  # 获取用户的前 K 个评分
        per_user_preds[user.item()].extend(top_k_scores.tolist())  # 转换为列表并扩展
    precisions = 0.0
    recalls = 0.0
    for user, preds in per_user_preds.items():
        while len(preds) < K:
            preds.append(random.choice(range(edge_index.shape[1])))
        top_ratings, top_items = torch.topk(torch.tensor(preds), K)
        correct_preds = edge_index[user, top_items].sum().item()
        total_pos = edge_index[user].sum().item()
        precisions += correct_preds / K
        recalls += correct_preds / total_pos if total_pos != 0 else 0
    num_users = len(user_indices.unique())
    return precisions / num_users, recalls / num_users
def trans_ml(dat, thres):
    """
    Transform function that assign non-negative entries >= thres 1, and non-
    negative entries <= thres 0. Keep other entries the same.
    """
    thres = thres[0]
    matrix = dat['edge_index']
    matrix[(matrix < thres) & (matrix > -1)] = 0
    matrix[(matrix >= thres)] = 1
    dat['edge_index'] = matrix
    return dat

model = torch.load('model.pth',weights_only=False)
root = os.getcwd()
movielens = MovieLens(root=root, transform=trans_ml)
data = movielens.get()
simulator = LightGCNSimulator(model=model, data=data)
user_indices = torch.linspace(start=0,
                                       end=20 - 1, steps=20).long()
item_indices = torch.linspace(start=0,
                                       end=3883 - 1, steps=3883).long()
# ... existing code ...
# score = simulator.score(30,1101)
# print(score)
score = simulator.score(100, item_indices)
topk = torch.topk(score, 20).indices.cpu().tolist()
print(topk)
# # 将 score 转换为 DataFrame，确保列名与评分数量匹配
# score_df = pd.DataFrame(score.detach().cpu().numpy(), columns=[f"Score_{i}" for i in range(score.shape[1])])

# # 将 DataFrame 写入 CSV 文件
# score_df.to_csv("scores.csv", index=False)  # 不写入行索引


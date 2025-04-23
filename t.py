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
rating_threshold = 3  #@param {type: "integer"}: Ratings equal to or greater than 3 are positive items.

config_dict = {
    "num_samples_per_user": 500,
    "num_users": 200,

    "epochs": 100,
    "batch_size": 128,
    "lr": 0.001,
    "weight_decay": 0.1,

    "embedding_size": 64,
    "num_layers": 5,
    "K": 10,
    "mf_rank": 8,

    "minibatch_per_print": 100,
    "epochs_per_print": 1,

    "val_frac": 0.2,
    "test_frac": 0.1,

    "model_name": "model.pth"
}
DATA_PATH = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
model = torch.load("model.pth", weights_only=False)
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


class MovieLens(Dataset):
    def __init__(self, root, transform=None, pre_transform=None,
            transform_args=None, pre_transform_args=None):
        """
        root = where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (process data).
        """
        super(MovieLens, self).__init__(root, transform, pre_transform)
        self.transform = transform
        self.pre_transform = pre_transform
        self.transform_args = transform_args
        self.pre_transform_args = pre_transform_args

    @property
    def raw_file_names(self):
        return "ml-1m.zip"

    @property
    def processed_file_names(self):
        return ["data_movielens.pt"]

    def download(self):
        # Download to `self.raw_dir`.
        download_url(DATA_PATH, self.raw_dir)

    def _load(self):
        print(self.raw_dir)
        # extract_zip(self.raw_paths[0], self.raw_dir)
        with zipfile.ZipFile(self.raw_paths[0], 'r') as zip_ref:
            zip_ref.extractall(self.raw_dir)
        unames = ['user_id', 'gender', 'age', 'occupation', 'zip']
        users = pd.read_table(self.raw_dir+'/ml-1m/users.dat',
                              sep='::', header=None, names=unames,
                              engine='python', encoding='latin-1')
        rnames = ['user_id', 'movie_id', 'rating', 'timestamp']
        ratings = pd.read_table(self.raw_dir+'/ml-1m/ratings.dat', sep='::',
                                header=None, names=rnames, engine='python',
                                encoding='latin-1')
        mnames = ['movie_id', 'title', 'genres']
        movies = pd.read_table(self.raw_dir+'/ml-1m/movies.dat', sep='::',
                               header=None, names=mnames, engine='python',
                               encoding='latin-1')
        dat = pd.merge(pd.merge(ratings, users), movies)

        return users, ratings, movies, dat

    def process(self):
        print('run process')
        # load information from file
        users, ratings, movies, dat = self._load()

        users = users['user_id']
        movies = movies['movie_id']

        num_users = config_dict["num_users"]
        if num_users != -1:
            users = users[:num_users]

        user_ids = range(len(users))
        movie_ids = range(len(movies))

        user_to_id = dict(zip(users, user_ids))
        movie_to_id = dict(zip(movies, movie_ids))

        # get adjacency info
        self.num_user = users.shape[0]
        self.num_item = movies.shape[0]

        # initialize the adjacency matrix
        rat = torch.zeros(self.num_user, self.num_item)

        for index, row in ratings.iterrows():
            user, movie, rating = row[:3]
            if num_users != -1:
                if user not in user_to_id: break
            # create ratings matrix where (i, j) entry represents the ratings
            # of movie j given by user i.
            rat[user_to_id[user], movie_to_id[movie]] = rating

        # create Data object
        data = Data(edge_index = rat,
                    raw_edge_index = rat.clone(),
                    data = ratings,
                    users = users,
                    items = movies)

        # apply any pre-transformation
        if self.pre_transform is not None:
            data = self.pre_transform(data, self.pre_transform_args)

        # apply any post_transformation
        # if self.transform is not None:
        #     # data = self.transform(data, self.transform_args)
        data = self.transform(data, [rating_threshold])

        # save the processed data into .pt file
        torch.save(data, osp.join(self.processed_dir, f'data_movielens.pt'))
        print('process finished')

    def len(self):
        """
        return the number of examples in your graph
        """
        # TODO: how to define number of examples
        return

    def get(self):
        """
        The logic to load a single graph
        """
        data = torch.load(osp.join(self.processed_dir, 'data_movielens.pt'), weights_only=False)
        return data

    def train_val_test_split(self, val_frac=0.2, test_frac=0.1):
        """
        Return two mask matrices (M, N) that represents edges present in the
        train and validation set
        """
        try:
            self.num_user, self.num_item
        except AttributeError:
            data = self.get()
            self.num_user = len(data["users"].unique())
            self.num_item = len(data["items"].unique())
        # get number of edges masked for training and validation
        num_train_replaced = \
            round((test_frac+val_frac)*self.num_user*self.num_item)
        num_val_show = round(val_frac*self.num_user*self.num_item)

        # edges masked during training
        indices_user = np.random.randint(0, self.num_user, num_train_replaced)
        indices_item = np.random.randint(0, self.num_item, num_train_replaced)

        # sample part of edges from training stage to be unmasked during
        # validation
        indices_val_user = np.random.choice(indices_user, num_val_show)
        indices_val_item = np.random.choice(indices_item, num_val_show)

        train_mask = torch.ones(self.num_user, self.num_item)
        train_mask[indices_user, indices_item] = 0

        val_mask = train_mask.clone()
        val_mask[indices_val_user, indices_val_item] = 1

        test_mask = torch.ones_like(train_mask)

        return train_mask, val_mask, test_mask
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


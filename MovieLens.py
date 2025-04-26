import os.path as osp
import os
from tqdm import tqdm
from typing import List
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
pd.options.display.max_rows = 10

import torch
from torch_geometric.data import Dataset, Data, download_url, extract_zip
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

class MovieLens(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(MovieLens, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return "ml-1m.zip"

    @property
    def processed_file_names(self):
        return ["data_movielens.pt"]

    def download(self):
        download_url(DATA_PATH, self.raw_dir)

    def _load(self):
        print(self.raw_dir)
        with zipfile.ZipFile(self.raw_paths[0], 'r') as zip_ref:
            zip_ref.extractall(self.raw_dir)

        unames = ['user_id', 'gender', 'age', 'occupation', 'zip']
        users = pd.read_table(self.raw_dir+'/ml-1m/users.dat', sep='::', header=None, names=unames, engine='python', encoding='latin-1')

        rnames = ['user_id', 'movie_id', 'rating', 'timestamp']
        ratings = pd.read_table(self.raw_dir+'/ml-1m/ratings.dat', sep='::', header=None, names=rnames, engine='python', encoding='latin-1')

        mnames = ['movie_id', 'title', 'genres']
        movies = pd.read_table(self.raw_dir+'/ml-1m/movies.dat', sep='::', header=None, names=mnames, engine='python', encoding='latin-1')

        return users, ratings, movies

    def process(self):
        print('run process')

        users, ratings, movies = self._load()

        # Encode users and movies to 0-based index
        unique_users = ratings['user_id'].unique()
        unique_movies = ratings['movie_id'].unique()

        user_mapping = {user_id: idx for idx, user_id in enumerate(unique_users)}
        movie_mapping = {movie_id: idx for idx, movie_id in enumerate(unique_movies)}

        num_users = len(unique_users)
        num_movies = len(unique_movies)

        # Build edges: users -> movies
        user_indices = ratings['user_id'].map(user_mapping).values
        movie_indices = ratings['movie_id'].map(movie_mapping).values
        ratings_values = ratings['rating'].values

        # edge_index = (2, num_edges)
        edge_index = torch.tensor([user_indices, movie_indices + num_users], dtype=torch.long)
        edge_attr = torch.tensor(ratings_values, dtype=torch.float)
        print(edge_index)
        data = Data(
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=num_users + num_movies,
            users=torch.tensor(user_indices, dtype=torch.long),
            movies=torch.tensor(movie_indices, dtype=torch.long),
        )

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(data, osp.join(self.processed_dir, "data_movielens.pt"))
        print('process finished')

    def len(self):
        return 1

    def get(self, idx=0):
        data = torch.load(osp.join(self.processed_dir, 'data_movielens.pt'), weights_only=False)
        return data
root = os.getcwd()
movielens = MovieLens(root=root)
data = movielens.get()
print(data.edge_index)

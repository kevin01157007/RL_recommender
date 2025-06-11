from collections import defaultdict
from utility.utilis import build_edge_index
import torch
import random
# train_inter = [(1,8),(2,9),(8,10),(2,8)]
# train_user_pos_dict = defaultdict(list)

# for u, i in train_inter:
#     train_user_pos_dict[u].append(i)
# a = random.choice(list(train_user_pos_dict.keys()))
# p = random.choice(train_user_pos_dict[u])
# print(list(train_user_pos_dict.keys()))

# inp = [(2,8),(8,9),(7,8),(7,8)]
# ind = [(12,8),(80,9),(6,8),(17,8)]
# b = build_edge_index(inp,3)
# c = build_edge_index(ind,4)
# e = torch.cat([b,c],dim=1)
# e = torch.unique(e, dim = 1)
# print(e)

d = [(7,8),(8,9),(9,10),(7,12)]
random.shuffle(d) 
print(d[0])
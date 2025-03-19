import torch
import os
import glob
import random
from recbole.quick_start import run_recbole
from recbole.model.general_recommender import LightGCN
from recbole.data.utils import create_dataset, data_preparation
from recbole.config.configurator import Config
import networkx as nx
import matplotlib.pyplot as plt
import community  # Louvain 方法

# 配置 RecBole 設置
config_dict = {
    'model': 'LightGCN',
    'dataset': 'ml-1m',
    'epochs': 20,
    'topk': 10,
    'loss_type': 'BPR',
    'learning_rate': 0.001,
    'weight_decay': 1e-4,
    'stopping_step': 10,
    'gpu_id': '0',  # 指定 GPU ID
    'nproc': 1,
}
config = Config(model='LightGCN', dataset='ml-1m', config_dict=config_dict)

# 訓練模型
run_recbole(config_dict=config_dict)

# 讀取數據
dataset = create_dataset(config)
train_data, valid_data, test_data = data_preparation(config, dataset)

# 設置設備
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 載入 LightGCN 模型
model = LightGCN(config, train_data.dataset).to(device)

# 加載訓練好的權重
# 根據檔案名稱排序，獲取最新的
checkpoint_dir = 'saved'
checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'LightGCN-*.pth'))

if not checkpoint_files:
    raise FileNotFoundError("找不到任何 LightGCN 的模型檔案，請先訓練模型！")

# 根據檔案名稱排序，獲取最新的
checkpoint_files.sort(key=os.path.getctime, reverse=True)  # 按照創建時間排序
latest_checkpoint = checkpoint_files[0]

print(f"載入最新的模型檔案: {latest_checkpoint}")

# 加載模型
checkpoint = torch.load(latest_checkpoint, map_location=device, weights_only=False)
model.load_state_dict(checkpoint['state_dict'])
model.eval()

# 獲取嵌入
user_all_embeddings, item_all_embeddings = model.forward()

# 測試用戶推薦
user_id = 1  # 例如用戶 1
num_users = user_all_embeddings.shape[0]
if user_id >= num_users:
    raise ValueError(f"用戶 ID {user_id} 超出了範圍 (最大值: {num_users-1})")

user_embedding = user_all_embeddings[user_id].to(device)

# 計算推薦分數
scores = torch.matmul(user_embedding, item_all_embeddings.T)
top_items = torch.argsort(scores, descending=True)[:10]  # 推薦前 10 名

print("推薦電影 ID:", top_items.tolist())


# # 創建一個無向圖
# G = nx.Graph()

# # 添加用戶節點（U1, U2, ...）
# num_users = user_all_embeddings.shape[0]
# user_nodes = [f"U{u}" for u in range(num_users)]
# G.add_nodes_from(user_nodes, bipartite=0)  # 設定為第一層（用戶）

# # 添加電影節點（I1, I2, ...）
# num_items = item_all_embeddings.shape[0]
# item_nodes = [f"I{i}" for i in range(num_items)]
# G.add_nodes_from(item_nodes, bipartite=1)  # 設定為第二層（電影）

# for user_id in range(num_users):
#     user_embedding = user_all_embeddings[user_id]
#     scores = torch.matmul(user_embedding, item_all_embeddings.T)
#     top_items = torch.argsort(scores, descending=True)[:5]  # 每個用戶連結前 5 個推薦電影

#     for item_id in top_items.tolist():
#         G.add_edge(f"U{user_id}", f"I{item_id}")  # 建立用戶-電影的推薦關係

# print(f"圖中節點數量: {G.number_of_nodes()}，邊數: {G.number_of_edges()}")

# # 取一部分用戶和電影
# sample_users = random.sample(user_nodes, 3)  # 抽樣 50 個用戶
# sample_items = set()  # 存放這些用戶關聯的電影

# for user in sample_users:
#     neighbors = list(G.neighbors(user))  # 取得該用戶的推薦電影
#     sample_items.update(neighbors[:3])  # 只取前 3 部電影

# # 取樣的節點
# sample_nodes = sample_users + list(sample_items)
# subgraph = G.subgraph(sample_nodes)  # 生成子圖

# # 設定節點顏色（用戶藍色，電影紅色）
# color_map = ["blue" if node.startswith("U") else "red" for node in subgraph.nodes]

# plt.figure(figsize=(10, 6))
# pos = nx.spring_layout(subgraph, seed=42)
# nx.draw(subgraph, pos, node_color=color_map, with_labels=True, font_size=20, node_size=400)
# plt.title("Sampled User-Item Graph")
# plt.show()

# run.py
import torch
from env import RecSimEnv          # 從你的 env.py 導入
from LightGCNRS import LightGCNRS  # 假設你有這個文件
from LightGCN import LightGCN      # 假設你有這個文件
import os
import pandas as pd
from simulator import SimpleSimulator 
# --- 配置 ---
rec_model_config = {
    "n_users": 500,
    "m_items": 2000,
    "embedding_size": 64,
    "num_layers": 3,
}

# SimpleSimulator 類的定義已移至 simulator.py

def load_dat_file_to_interactions(file_path, n_user_config, n_item_config):
    """
    從 .dat 文件加載交互數據，並返回 (user, item) 元組列表。
    同時檢查 user_id 和 movie_id 是否越界。
    """
    interactions = []
    print(f"Attempting to load interactions from: {file_path}")
    
        # 假設 .dat 文件格式與 LightGCN_pretrain.py 中讀取的 ratings 文件類似
        # 通常第一列是 user_id，第二列是 movie_id
        # 如果您的 .dat 文件沒有表頭且直接是 user_id, movie_id, ...
        # 並且是以逗號分隔的
    df = pd.read_csv(file_path, sep=',', header=None, names=['user_id', 'movie_id', 'rating', 'timestamp'], engine='python', encoding='latin-1')
    
    # 確保列存在
    if 'user_id' not in df.columns or 'movie_id' not in df.columns:
        print(f"錯誤: 文件 {file_path} 缺少 'user_id' 或 'movie_id' 列。")
        return []

    valid_interactions = 0
    skipped_out_of_bounds = 0
    for _, row in df.iterrows():

            user = int(row['user_id'])
            item = int(row['movie_id'])
            if 0 <= user < n_user_config and 0 <= item < n_item_config:
                interactions.append((user, item))
                valid_interactions += 1
            else:
                skipped_out_of_bounds += 1
        
    if skipped_out_of_bounds > 0:
        print(f"警告: 從 {file_path} 加載時，由於超出配置的 n_user/n_item 範圍，跳過了 {skipped_out_of_bounds} 筆互動。")
    
    if not interactions:
        print(f"警告: 從 {file_path} 加載的有效互動數據為空。")
    else:
        print(f"從 {file_path} 成功加載 {len(interactions)} 筆有效互動。")
   
    return interactions

if __name__ == "__main__":
    # --- 設置 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    n_user = rec_model_config["n_users"]
    n_item = rec_model_config["m_items"]

    # --- 初始化組件 ---

    # 1. 初始圖結構
    init_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
    print(f"Initial edge_index shape: {init_edge_index.shape}")

    # 檢查預訓練嵌入文件是否存在
    if not os.path.exists("user_emb.pt") or not os.path.exists("item_emb.pt"):
        print("Error: Pre-trained embeddings ('user_emb.pt', 'item_emb.pt') not found.")
        print("Please run LightGCN_pretrain.py first to generate embeddings.")
        exit()

    # 在創建 Simulator 和 Env 之前加載嵌入
    try:
        
        print("Loading pre-trained embeddings...")
        user_embeddings = torch.load("user_emb.pt", map_location=device, weights_only=True)
        item_embeddings = torch.load("item_emb.pt", map_location=device, weights_only=True)
        print("Embeddings loaded successfully.")
        if user_embeddings.shape[0] != n_user or item_embeddings.shape[0] != n_item:
            print(f"Warning: Loaded embedding dimensions ({user_embeddings.shape[0]} users, {item_embeddings.shape[0]} items) "
                  f"do not match config ({n_user} users, {n_item} items).")
            n_user = user_embeddings.shape[0]
            n_item = item_embeddings.shape[0]
             
            print(f"Adjusted n_user={n_user}, n_item={n_item} based on loaded embeddings.")
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        exit()


    # 2. 推薦模型
    lightgcn_model = LightGCN(
        num_users=n_user,
        num_items=n_item,
        emb_size=rec_model_config["embedding_size"],
        n_layers=rec_model_config["num_layers"]
    ).to(device)
    rec_model = LightGCNRS(n_user, lightgcn_model, device)
    print("LightGCN and LightGCNRS initialized.")

    # 3. Agent
    agent = 0

    # 4. 創建 Simulator 實例 (從導入的類創建)
    print("Creating Simulator...")
    simulator = SimpleSimulator(user_embeddings, item_embeddings, device)
    print("Simulator created.")

    # --- 修改：從 val.dat 和 test.dat 加載數據 ---
    val_dat_path = "data_split/val.dat"
    test_dat_path = "data_split/test.dat"
    
    print(f"Loading validation interactions from {val_dat_path} for RecSimEnv val_data...")
    val_interactions_for_env = load_dat_file_to_interactions(val_dat_path, n_user, n_item) # 使用調整後的 n_user, n_item
    
    print(f"Loading test interactions from {test_dat_path} for RecSimEnv test_data...")
    test_interactions_for_env = load_dat_file_to_interactions(test_dat_path, n_user, n_item) # 使用調整後的 n_user, n_item

    k_eval_for_env = 20 # 可以保留或根據需要配置

    # --- 創建環境 ---
    print("Creating RecSimEnv...")
    try:
        env = RecSimEnv(
            init_edge_index=init_edge_index,
            n_user=n_user, # 使用基於嵌入調整後的 n_user
            n_item=n_item, # 使用基於嵌入調整後的 n_item
            agent=agent,
            rec_model=rec_model,
            sim=simulator,
            device=device,
            val_data=val_interactions_for_env,    # <-- 傳遞從 val.dat 加載的數據
            test_data=test_interactions_for_env,  # <-- 傳遞從 test.dat 加載的數據
            k_eval=k_eval_for_env      
        )
        print("RecSimEnv created successfully.")
    except Exception as e:
        print(f"Error creating RecSimEnv: {e}")
        exit()

    print("Starting environment run...")

    env.run(n_round=5, k_rec=10)
    print("Environment run finished.")


    print("Script finished.")


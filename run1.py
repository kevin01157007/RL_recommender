# run.py
import torch
from env import RecSimEnv      
from LightGCNRS import LightGCNRS
from LightGCN import LightGCN      
import os
import pandas as pd
from simulator import SimpleSimulator 
rec_model_config = {
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # 1. 初始圖結構
    init_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
    if not os.path.exists("user_emb.pt") or not os.path.exists("item_emb.pt"):
        print("Error: Pre-trained embeddings ('user_emb.pt', 'item_emb.pt') not found.")
        print("Please run LightGCN_pretrain.py first to generate embeddings.")
        exit()

    n_user, n_item = None, None # 初始化為 None
    try:
        print("Loading pre-trained user embeddings...")
        user_embeddings = torch.load("user_emb.pt", map_location=device, weights_only=True)
        n_user = user_embeddings.shape[0]
        print(f"User embeddings loaded. n_user set to {n_user} based on user_emb.pt.")

        print("Loading pre-trained item embeddings...")
        item_embeddings = torch.load("item_emb.pt", map_location=device, weights_only=True)
        n_item = item_embeddings.shape[0]
        print(f"Item embeddings loaded. n_item set to {n_item} based on item_emb.pt.")
            
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        exit()
    
    if n_user is None or n_item is None:
        print("Error: Failed to determine n_user or n_item from embeddings.")
        exit()

    # 2. 推薦模型
    lightgcn_model = LightGCN(
        num_users=n_user, 
        num_items=n_item, 
        emb_size=rec_model_config["embedding_size"],
        n_layers=rec_model_config["num_layers"]
    ).to(device)
    rec_model = LightGCNRS(n_user, lightgcn_model, device)
    agent = 0
    # 4. 創建 Simulator 實例
    # print("Creating Simulator...") # 可以取消註釋以調試
    simulator = SimpleSimulator(user_embeddings, item_embeddings, device)
    # print("Simulator created.") # 可以取消註釋以調試

    # --- 從 val.dat 和 test.dat 加載數據 ---
    val_dat_path = "data_split/val.dat"
    test_dat_path = "data_split/test.dat"
    
    # print(f"Loading validation interactions from {val_dat_path} for RecSimEnv val_data...") # 可以取消註釋以調試
    val_interactions_for_env = load_dat_file_to_interactions(val_dat_path, n_user, n_item) 
    
   
    test_interactions_for_env = load_dat_file_to_interactions(test_dat_path, n_user, n_item)

    k_eval_for_env = 20 

    try:
        env = RecSimEnv(
            init_edge_index=init_edge_index,
            n_user=n_user, 
            n_item=n_item, 
            agent=agent,
            rec_model=rec_model,
            sim=simulator,
            device=device,
            val_data=val_interactions_for_env,    
            test_data=test_interactions_for_env,  
            k_eval=k_eval_for_env      
        )
       
    except Exception as e:
        print(f"Error creating RecSimEnv: {e}")
        exit()

    print("Starting environment run...")
    env.run(n_round=5, k_rec=10)
    print("Environment run finished.")
    print("Script finished.")


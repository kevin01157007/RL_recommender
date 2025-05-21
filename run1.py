# run.py
import torch
from env import RecSimEnv          # 從你的 env.py 導入
from LightGCNRS import LightGCNRS  # 假設你有這個文件
from LightGCN import LightGCN      # 假設你有這個文件
import os
from simulator import SimpleSimulator 
# --- 配置 ---
rec_model_config = {
    "n_users": 500,
    "m_items": 2000,
    "embedding_size": 64,
    "num_layers": 3,
}

# SimpleSimulator 類的定義已移至 simulator.py

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

    # Load validation data for post-simulation training
    val_interactions_path = "val_interactions.pt"
    test_data_for_env = []
    k_eval_for_env = 20 # Default k_eval, can be configured if needed or loaded from elsewhere
    if os.path.exists(val_interactions_path):
        try:
            print(f"Loading validation interactions from {val_interactions_path}...")
            # Assuming val_interactions.pt contains a list of tuples or a tensor [N, 2]
            loaded_val_data = torch.load(val_interactions_path, map_location=device)
            if isinstance(loaded_val_data, torch.Tensor) and loaded_val_data.dim() == 2 and loaded_val_data.size(1) == 2:
                # Convert tensor to list of tuples
                test_data_for_env = [(u.item(), i.item()) for u, i in loaded_val_data]
                print(f"Successfully loaded and processed {len(test_data_for_env)} validation interactions.")
            elif isinstance(loaded_val_data, list) and all(isinstance(item, tuple) and len(item) == 2 for item in loaded_val_data):
                # Ensure it's a list of (user, item) tuples
                test_data_for_env = loaded_val_data
                print(f"Successfully loaded {len(test_data_for_env)} validation interactions (as list).")
            else:
                print(f"Warning: {val_interactions_path} contains data in an unrecognized format. Expected a list of (user, item) tuples or a [N,2] tensor.")
                print("Post-simulation training will be skipped if validation data is not correctly loaded or is empty.")
        except Exception as e:
            print(f"Error loading validation interactions from {val_interactions_path}: {e}")
            print("Post-simulation training will be skipped due to error in loading validation data.")
    else:
        print(f"Warning: Validation interaction file '{val_interactions_path}' not found.")
        print("Post-simulation training will be skipped as no validation data is provided.")

    # --- 創建環境 ---
    print("Creating RecSimEnv...")
    try:
        env = RecSimEnv(
            init_edge_index=init_edge_index,
            n_user=n_user,
            n_item=n_item,
            agent=agent,
            rec_model=rec_model,
            sim=simulator,
            device=device,
            test_data=test_data_for_env, # Pass the loaded validation data
            k_eval=k_eval_for_env      # Pass k_eval
        )
        print("RecSimEnv created successfully.")
    except Exception as e:
        print(f"Error creating RecSimEnv: {e}")
        exit()

    print("Starting environment run...")

    env.run(n_round=5, k_rec=10)
    print("Environment run finished.")


    print("Script finished.")


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
            device=device
        )
        print("RecSimEnv created successfully.")
    except Exception as e:
        print(f"Error creating RecSimEnv: {e}")
        exit()

    print("Starting environment run...")
    try:
        env.run(n_round=5, k_rec=10)
        print("Environment run finished.")
    except Exception as e:
        print(f"Error during environment run: {e}")

    print("Script finished.")


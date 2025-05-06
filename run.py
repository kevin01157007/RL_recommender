# run.py
import torch
from env import RecSimEnv          # 從你的 env.py 導入
from LightGCNRS import LightGCNRS  # 假設你有這個文件
from LightGCN import LightGCN      # 假設你有這個文件
import os

# --- 配置 ---
rec_model_config = {
    "n_users": 6040,          # MovieLens-1M 的用戶數 (根據 LightGCN_pretrain.py)
    "m_items": 3706,          # MovieLens-1M 的物品數 (根據 LightGCN_pretrain.py)
    "embedding_size": 64,     # 嵌入維度
    "num_layers": 3,          # LightGCN 層數
}

if __name__ == "__main__":
    # --- 設置 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    n_user = rec_model_config["n_users"]
    n_item = rec_model_config["m_items"]

    # --- 初始化組件 ---

    # 1. 初始圖結構 (這裡用空的示例，你需要根據實際情況加載)
    #    env.py 需要一個 [2, E] 的 tensor，但你的範例 run.py 給的是 [U, I]
    #    這裡創建一個空的 [2, 0] tensor 作為起始，表示沒有初始交互
    #    或者，你可以加載 LightGCN_pretrain.py 的 train_edge_index
    init_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
    print(f"Initial edge_index shape: {init_edge_index.shape}")

    # 檢查預訓練嵌入文件是否存在
    if not os.path.exists("user_emb.pt") or not os.path.exists("item_emb.pt"):
        print("Error: Pre-trained embeddings ('user_emb.pt', 'item_emb.pt') not found.")
        print("Please run LightGCN_pretrain.py first to generate embeddings.")
        exit()

    # 2. 推薦模型
    lightgcn_model = LightGCN(
        num_users=n_user,
        num_items=n_item,
        emb_size=rec_model_config["embedding_size"],
        n_layers=rec_model_config["num_layers"]
    ).to(device)
    # 注意：這裡只是初始化了模型結構，env.py 會加載預訓練的嵌入，
    # 但 LightGCN 模型的權重（如果有的話，除了 embedding）需要另外加載或訓練。
    # LightGCNRS 似乎是一個包裝器
    rec_model = LightGCNRS(n_user, lightgcn_model, device)
    print("LightGCN and LightGCNRS initialized.")

    # 3. Agent (根據你的範例，似乎只是一個標識符)
    agent = 0

    # 4. Simulator (Sim) - 根據你的 env.py，__init__ 需要 sim，但 run 方法內部實現未使用
    #    如果你需要 sim，需要在此處創建並傳入 env
    simulator = None # 暫時設置為 None

    # --- 創建環境 ---
    print("Creating RecSimEnv...")
    try:
        env = RecSimEnv(
            init_edge_index=init_edge_index,
            n_user=n_user,
            n_item=n_item,
            agent=agent,
            rec_model=rec_model,
            sim=simulator,       # 傳入 simulator 對象，如果需要的話
            device=device
        )
        print("RecSimEnv created successfully.")
    except Exception as e:
        print(f"Error creating RecSimEnv: {e}")
        exit()

    # --- 運行模擬 ---
    print("Starting environment run...")
    try:
        env.run(n_round=5, k_rec=10) # 運行 5 個回合，每次推薦 10 個物品
        print("Environment run finished.")
    except Exception as e:
        print(f"Error during environment run: {e}")

    print("Script finished.")

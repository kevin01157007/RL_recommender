# run.py
import torch
from env import RecSimEnv      
from model.RS.LightGCNRS import LightGCNRS
from model.LightGCN.LightGCN import LightGCN    
import os
import pandas as pd
import utility.utilis as utilis
from model.simulator.simulator import SimpleSimulator 
rec_model_config = {
    "embedding_size": 64,
    "num_layers": 3,
}

if __name__ == "__main__":
    ratings_val = pd.read_csv('data/val.dat', sep=',', names=['user_id', 'movie_id', 'rating', 'timestamp'], engine='python', encoding='latin-1')
    ratings_test = pd.read_csv('data/test.dat', sep=',', names=['user_id', 'movie_id', 'rating', 'timestamp'], engine='python', encoding='latin-1')

    val_df         = ratings_val
    test_df        = ratings_test
    val_inter, test_inter = map(utilis.pairs_from, (val_df, test_df))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # 1. 初始圖結構
    init_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)

    user_embeddings = torch.load("model/simulator/user_emb.pt", map_location=device, weights_only=True)
    item_embeddings = torch.load("model/simulator/item_emb.pt", map_location=device, weights_only=True)

    n_user, n_item = user_embeddings.shape[0], item_embeddings.shape[0] # 初始化為 None

    # 2. 推薦模型
    lightgcn_model = LightGCN(
        num_users=n_user, 
        num_items=n_item, 
        emb_size=rec_model_config["embedding_size"],
        n_layers=rec_model_config["num_layers"]
    ).to(device)
    rec_model = LightGCNRS(n_user, lightgcn_model, device)
    # 4. 創建 Simulator 實例
    simulator = SimpleSimulator(user_embeddings, item_embeddings, device)
    
    k_eval_for_env = 80

    try:
        env = RecSimEnv(
            init_edge_index=init_edge_index,
            n_user=n_user, 
            n_item=n_item, 
            rec_model=rec_model,
            sim=simulator,
            device=device,
            k_eval=10      
        )
       
    except Exception as e:
        print(f"Error creating RecSimEnv: {e}")
        exit()

    print("Starting environment run...")
    env.run(n_round=11, k_rec=k_eval_for_env)
    print("Environment run finished.")
    print("Script finished.")


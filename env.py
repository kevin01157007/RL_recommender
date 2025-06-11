import torch
import pandas as pd
from utility.utilis import build_edge_index, train_val, test
import community.community_louvain as community_louvain
from utility.bulid_dtrain_graph import build_dtrain_graph
import time

class RecSimEnv:
    def __init__(self,
                 init_edge_index,  # tensor([2,E])  初始邊索引
                 n_user,
                 n_item,
                 rec_model,
                 sim,
                 device="cuda",
                 val_data=None,  # 新增：用於微調期間的驗證數據
                 test_data=None, # 用於微調後的最終測試
                 k_eval=20):      # 為評估一致性新增 k_eval
        self.device   = torch.device(device)
        self.n_user   = n_user
        self.n_item   = n_item

        # 1) 建立動態圖
        self.edge_index = init_edge_index.clone().to(self.device)

        self.rec_model = rec_model # 這是 LightGCNRS
        self.sim       = sim

        # 3) **載入預先儲存的 embeddings**
        self.user_emb = torch.load("model/simulator/user_emb.pt", map_location=self.device, weights_only=True)  # [n_user, d]
        self.item_emb = torch.load("model/simulator/item_emb.pt", map_location=self.device, weights_only=True)  # [n_item, d]
        
        self.k_eval = k_eval # 儲存 k_eval(TOPK)


    def run(self, n_round, k_rec):
        dtrain_new_interactions = [] # 這將是一個 (u,i) 元組的列表，累積所有輪次的互動
        seen = {u: set() for u in range(self.n_user)}

        for t in range(n_round):
            print(f"===== Time step{t} =====")
            current_round_interactions = [] # 當前timestep收集的互動
            for u in range(self.n_user):
                # 1) 取得 Top-k 推薦
                rec_items = self.rec_model.recommend(u, k=k_rec, exclude=seen[u]) # rec_items 是 I_i^t
               
                # rec_items  推薦的 item_ids
                for item_id_recommended in rec_items: # 直接遍歷 RS 推薦的物品
                    simulator_score = self.sim.score(u, item_id_recommended).item()
                    if simulator_score > 0.7:   
                        current_round_interactions.append((u, item_id_recommended))
                        seen[u].add(item_id_recommended) # 用戶喜歡過的物品
            dtrain_new_interactions.extend(current_round_interactions)

            #t = 0先不訓練，收集dtest、dval
            if t == 0:
                dtest = current_round_interactions[:len(current_round_interactions)]
                dval = current_round_interactions[len(current_round_interactions):]
                current_round_interactions = []
                dtrain_new_interactions = []
                        
            if current_round_interactions: 
                delta_edge_index = build_edge_index(current_round_interactions, self.n_user).to(self.device)
                if delta_edge_index.numel() > 0:
                    self.edge_index = torch.cat([self.edge_index, delta_edge_index], dim=1)
                    self.edge_index = torch.unique(self.edge_index, dim=1)

            if t > 0:
                dtrain_graph = build_dtrain_graph(dtrain_new_interactions)
                communities = community_louvain.best_partition(dtrain_graph, resolution=2, random_state=42)
                print(f"總社群數量: {len(set(communities.values()))}")


            # --- 每輪結束後進行訓練 ---
            if dtrain_new_interactions: 
                current_num_epochs = 10 + t * 10
                print(f"\n--- 第 {t} 輪後進行訓練，使用目前累積的 {len(dtrain_new_interactions)} 筆真實互動，訓練 {current_num_epochs} 個 epochs ---")
                self.train_model_on_collected_data(
                    training_interactions=dtrain_new_interactions,
                    val_interactions=self.val_data,
                    k_eval=self.k_eval,
                    num_epochs=current_num_epochs
                )
    
            print() # 所有輪次處理完畢後換行

            pd.DataFrame(current_round_interactions, columns=['u','i']).to_csv(f"new_interactions{t}.csv", index=False)
        
        # --- 最終測試評估 ---
        print(f"\n===== {n_round} 輪模擬結束後，於測試集上進行最終評估 =====")
        print(f"將使用初始測試集中的 {len(self.test_data)} 筆互動進行最終評估...")

        full_edge_index  = build_edge_index(dtrain_new_interactions + dval + dtest, self.n_user).to(self.device)

        model_to_evaluate = self.rec_model.model # LightGCN 實例
        model_to_evaluate.eval()
        test(model_to_evaluate, 
             num_items = self.n_item,
             batch_size = 512, 
             device = self.device, 
             train_inter = dtrain_new_interactions, val_inter = dval, test_inter = dtest, 
             train_edge_index = self.edge_index, full_edge_index = full_edge_index, 
             K = self.k_eval)
            
    def train_model_on_collected_data(self,
                                   training_interactions, # (u,i) 元組列表
                                   val_interactions,       
                                   num_epochs = 200,          
                                   batch_size = 512,        
                                   lr = 1e-4,               
                                   lambda_reg = 5e-4,        
                                   k_eval = 20,               
                                   patience = 20): 

        
        model_to_train = self.rec_model.model # 這是 LightGCN 實例
        optimizer = torch.optim.Adam(model_to_train.parameters(), lr=lr)
        

        val_edge_index   = build_edge_index(training_interactions + val_interactions, self.n_user).to(self.device)
        
        patience = 20
        print(f"在 {len(training_interactions)} 個互動上訓練 LightGCN，在 {len(val_interactions)} 個互動上進行驗證。")
        print(f"Epochs: {num_epochs}, Batch Size: {batch_size}, LR: {lr}, Lambda_reg: {lambda_reg}, K_eval: {k_eval}, Patience: {patience}")

        # LightGCN 在訓練和驗證期間前向傳播所用的 edge_index
        # 是 self.edge_index，它已經用所有 training_interactions 更新過了。
        current_graph_edge_index = self.edge_index.to(self.device)

        train_val(model = model_to_train, 
                  num_items = self.n_item, 
                  num_epochs = num_epochs, 
                  batch_size = batch_size, 
                  device = self.device, 
                  opt = optimizer, 
                  train_inter = training_interactions, val_inter = val_interactions, 
                  train_edge_index = current_graph_edge_index, val_edge_index = val_edge_index, 
                  patience = patience, K = k_eval)
        
        print("完成模擬後訓練。")
    

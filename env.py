import torch
import pandas as pd
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops
from torch_geometric.loader import NeighborLoader
from LightGCNRS import LightGCNRS
from LightGCN import LightGCN
from utilis import compute_bpr_loss_dataset, precision_recall_ndcg_at_k, sample_pos_neg, bpr_loss, build_edge_index
import time
import random

class RecSimEnv:
    def __init__(self,
                 init_edge_index,  # tensor([2,E])  初始邊索引
                 n_user,
                 n_item,
                 agent,
                 rec_model,
                 sim,
                 device="cuda",
                 val_data=None,  # 新增：用於微調期間的驗證數據
                 test_data=None, # 用於微調後的最終測試
                 k_eval=20):      # 為評估一致性新增 k_eval
        self.device   = torch.device(device)
        self.n_user   = n_user
        self.n_item   = n_item
        self.agent    = agent

        # 1) 建立動態圖
        self.edge_index = init_edge_index.clone().to(self.device)

        # 2) 建立推薦與 Simulator（可先保留）
        self.rec_model = rec_model # 這是 LightGCNRS
        self.sim       = sim

        # 3) **載入預先儲存的 embeddings**
        self.user_emb = torch.load("user_emb.pt", map_location=self.device)  # [n_user, d]
        self.item_emb = torch.load("item_emb.pt", map_location=self.device)  # [n_item, d]
        
        self.val_data = val_data if val_data is not None else []   # 儲存 val_data
        self.test_data = test_data if test_data is not None else [] # 儲存 test_data
        self.k_eval = k_eval # 儲存 k_eval
        # 假設 test_data 是一個 (user, item) 元組的列表，類似 pretrain.py 中的 val_inter 或 test_inter
        # 如果 bpr_loss 直接需要 lambda_reg，請確保其可用或傳遞它。utilis 中的 bpr_loss 有一個預設值。
        # self.lambda_reg = 1e-4 # 範例，如果 bpr_loss 直接需要（若不使用 utilis 中的那個）

    def run(self, n_round, k_rec):
        rec_items_list  = []
        new_interactions = [] # 這將是一個 (u,i) 元組的列表
        seen = {u: set() for u in range(self.n_user)}
        sim_scores_list = []

        for t in range(n_round):
            print(f"===== 時間步 {t} =====")
            current_round_interactions = []
            for u in range(self.n_user):
                # 1) 取得 Top-k 推薦
                rec_items = self.rec_model.recommend(u, k=k_rec, exclude=seen[u])
                rec_with_scores = [
                    (item_id, (self.user_emb[u] @ self.item_emb[item_id]).item())
                    for item_id in rec_items
                ]

               
                user_new_interactions = []
                # rec_items  推薦的 item_ids
                for item_id_recommended in rec_items: # 直接遍歷 RS 推薦的物品
                    simulator_score = self.sim.score(u, item_id_recommended).item()
                    if simulator_score > 0.7:   
                        user_new_interactions.append((u, item_id_recommended))
                
                current_round_interactions.extend(user_new_interactions)
                new_interactions.extend(user_new_interactions) # 累積所有新的互動
                seen[u].update(rec_items)

                # 收集推薦記錄
                rec_items_list.extend([(u, t, item_id, item_raw_score) for item_id, item_raw_score in rec_with_scores])
            # End of the for u in range(self.n_user) loop for round t
            
            # 在處理完一輪中的所有用戶後，更新全局 edge_index (每輪一次)
            if current_round_interactions: # 只有當這一輪有新的互動時才更新
                delta_edge_index = build_edge_index(current_round_interactions, self.n_user).to(self.device)
                
                if delta_edge_index.numel() > 0: # 確保 delta_edge_index 確實包含邊
                    self.edge_index = torch.cat([self.edge_index, delta_edge_index], dim=1)
                    self.edge_index = torch.unique(self.edge_index, dim=1)
                    message = f"時間步 {t}: 全局 edge_index 已更新。目前總邊數 (單向視角): {self.edge_index.size(1) // 2}"
                    print(f"\r{message:<100}", end="") # 更新並填充到100字符
                
          

        print() # 在循環結束後換行，為後續打印做準備


        pd.DataFrame(new_interactions, columns=['u','i']).to_csv('new_interactions.csv', index=False)
        

        # 模擬結束後，在收集到的 new_interactions 上訓練模型
        if new_interactions:
            train_data_for_fine_tuning = list(new_interactions) # 使用所有 new_interactions 進行訓練
            val_data_for_fine_tuning = self.test_data         # 使用 self.test_data (在 __init__ 中設置) 進行驗證

            print(f"收集到 {len(train_data_for_fine_tuning)} 筆新互動將用於訓練。")

            if val_data_for_fine_tuning: # 檢查列表是否為空
                print(f"將使用 {len(val_data_for_fine_tuning)} 筆來自初始 test_data 的互動進行驗證。")
            else:
                print("警告：初始 test_data (用於驗證) 為空。驗證集將為空。")
            self.train_model_on_collected_data(
                training_interactions=train_data_for_fine_tuning,
                val_interactions=self.val_data, 
                k_eval=self.k_eval 
            )

            # self.test_data 最終評估
            if self.test_data:
                print(f"\n完成微調後，在 {len(self.test_data)} 筆來自初始 test_data 的互動上進行最終測試評估...")
                model_to_evaluate = self.rec_model.model # LightGCN 實例
                model_to_evaluate.eval()
                with torch.no_grad():
                    # 使用 self.edge_index，因為它包含了模擬中收集到的所有互動
                    current_graph_edge_index = self.edge_index.to(self.device)
                    
                    test_prec, test_rec, test_ndcg = precision_recall_ndcg_at_k(
                        model_to_evaluate,
                        current_graph_edge_index, 
                        self.test_data,  # 使用 self.test_data
                        train_pairs=train_data_for_fine_tuning + self.val_data, # 從推薦中排除訓練和驗證集中的物品
                        K=self.k_eval
                    )
                    print(f"最終測試結果: P@{self.k_eval} {test_prec:.4f} R@{self.k_eval} {test_rec:.4f} NDCG@{self.k_eval} {test_ndcg:.4f}")
            else:
                print("\n警告：初始 test_data 為空，跳過最終測試評估。")

        elif not new_interactions:
            print("模擬期間未收集到新的互動。跳過模擬後訓練。")
            
        return self.edge_index # 或其他相關結果


    def train_model_on_collected_data(self,
                                   training_interactions, # (u,i) 元組列表
                                   val_interactions,       
                                   num_epochs=200,          
                                   batch_size=2048,        
                                   lr=1e-3,               
                                   lambda_reg=5e-4,        
                                   k_eval=20,               
                                   num_neg_per_interaction=10, # 在 pretrain 中是 num_neg_per_u，現在是每個互動
                                   patience=10): 

        model_to_train = self.rec_model.model # 這是 LightGCN 實例
        optimizer = torch.optim.Adam(model_to_train.parameters(), lr=lr)
        
        print(f"在 {len(training_interactions)} 個互動上訓練 LightGCN，在 {len(val_interactions)} 個互動上進行驗證。")
        print(f"Epochs: {num_epochs}, Batch Size: {batch_size}, LR: {lr}, Lambda_reg: {lambda_reg}, K_eval: {k_eval}, Negatives: {num_neg_per_interaction}, Patience: {patience}")

        loss_hist, val_loss_hist, val_prec_hist, val_rec_hist, val_ndcg_hist = [], [], [], [], []
        best_val_ndcg = -1.0 
        patience_counter = 0

        # LightGCN 在訓練和驗證期間前向傳播所用的 edge_index
        # 是 self.edge_index，它已經用所有 training_interactions 更新過了。
        current_graph_edge_index = self.edge_index.to(self.device)

        for epoch in range(1, num_epochs + 1):
            model_to_train.train()
            t0 = time.time()

            
            all_training_samples = []
            for _ in range(num_neg_per_interaction): # 這將通過調用 sample_pos_neg N 次為每個正樣本生成 N 個負樣本
                current_samples = sample_pos_neg(
                    training_interactions, # (u,i) 列表
                    self.n_user,
                    self.n_item,
                    num_negatives=1, # 目前 sample_pos_neg 中的這個參數是遺跡性的
                    seed=epoch + _ # 稍微改變種子以獲得不同的負樣本集
                )
                all_training_samples.append(current_samples)
            
            if not all_training_samples:
                print(f"Epoch {epoch:02d} | 未生成訓練樣本。跳過此 epoch 的訓練。")
                continue
                
            training_samples_tensor = torch.cat(all_training_samples, dim=0)
            training_samples_tensor = training_samples_tensor[torch.randperm(len(training_samples_tensor))].to(self.device)
            
            total_train_loss = 0
            processed_samples_count = 0
            for st in range(0, len(training_samples_tensor), batch_size):
                batch = training_samples_tensor[st:st + batch_size]
                u, p, n = batch[:, 0], batch[:, 1], batch[:, 2]
                
                optimizer.zero_grad()
                loss, _, _ = bpr_loss(model_to_train, u, p, n, current_graph_edge_index, lambda_reg=lambda_reg)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item() * u.size(0)
                processed_samples_count += u.size(0)
            
            avg_train_loss = total_train_loss / processed_samples_count if processed_samples_count > 0 else 0
            loss_hist.append(avg_train_loss)

            # 驗證
            model_to_train.eval()
            with torch.no_grad():
                val_loss = compute_bpr_loss_dataset(
                    model_to_train,
                    val_interactions, # 對 (pairs)
                    self.device,
                    current_graph_edge_index, # edge_index_train
                    self.n_user,
                    self.n_item,
                    num_negatives=1, 
                    exclude_pairs=training_interactions + val_interactions # 從負樣本選擇中排除訓練和驗證的正樣本
                )
                val_loss_hist.append(val_loss)

                prec, rec, ndcg = precision_recall_ndcg_at_k(
                    model_to_train,
                    current_graph_edge_index, # edge_index_train
                    val_interactions,         # test_pairs
                    train_pairs=training_interactions, # 從推薦中排除已見物品
                    K=k_eval
                )
                val_prec_hist.append(prec)
                val_rec_hist.append(rec)
                val_ndcg_hist.append(ndcg)

            elapsed_time = time.time() - t0
            print(f"Epoch {epoch:02d} | {elapsed_time:.1f}s | TrainLoss {avg_train_loss:.4f} | ValLoss {val_loss:.4f} | P@{k_eval} {prec:.4f} R@{k_eval} {rec:.4f} NDCG@{k_eval} {ndcg:.4f}")

            if ndcg > best_val_ndcg:
                best_val_ndcg = ndcg
                patience_counter = 0
                # 可選：儲存模型: torch.save(model_to_train.state_dict(), "best_recsim_finetuned_model.pth")
                # print(f"新的最佳 Val NDCG: {best_val_ndcg:.4f}。儲存模型。")
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"由於 Val NDCG 連續 {patience} 個 epochs 沒有改善，在 epoch {epoch} 提前停止。")
                break
        
        print("完成模擬後訓練。")
    

import torch
import pandas as pd
from utility.utilis import compute_bpr_loss_dataset, precision_recall_ndcg_at_k, sample_pos_neg, bpr_loss, build_edge_index
from utility.build_nx_graph import build_nx_graph
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
        
        self.val_data = val_data
        self.test_data = test_data
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
                        
            if current_round_interactions: 
                delta_edge_index = build_edge_index(current_round_interactions, self.n_user).to(self.device)
                if delta_edge_index.numel() > 0:
                    self.edge_index = torch.cat([self.edge_index, delta_edge_index], dim=1)
                    self.edge_index = torch.unique(self.edge_index, dim=1)

            dtrain_graph = build_dtrain_graph(dtrain_new_interactions)
            all_partition = community_louvain.best_partition(dtrain_graph, resolution=2, random_state=42)
            communities = all_partition
            print(f"總社群數量: {len(set(communities.values()))}")

            # --- 每輪結束後進行訓練 ---
            if dtrain_new_interactions: 
                current_num_epochs = 10 + t * 10
                print(f"\n--- 第 {t} 輪後進行訓練，使用目前累積的 {len(dtrain_new_interactions)} 筆真實互動，訓練 {current_num_epochs} 個 epochs ---")
                self.train_model_on_collected_data(
                    training_interactions=list(dtrain_new_interactions), 
                    val_interactions=self.val_data,      
                    k_eval=self.k_eval,                  
                    num_epochs=current_num_epochs 
                )
    
            print() # 所有輪次處理完畢後換行


            pd.DataFrame(dtrain_new_interactions, columns=['u','i']).to_csv('new_interactions.csv', index=False)
        

        # --- 最終測試評估 ---
        print(f"\n===== {n_round} 輪模擬結束後，於測試集上進行最終評估 =====")
        print(f"將使用初始測試集中的 {len(self.test_data)} 筆互動進行最終評估...")
        model_to_evaluate = self.rec_model.model # LightGCN 實例
        model_to_evaluate.eval()
        with torch.no_grad():
            # 使用最終的 self.edge_index，它包含所有輪次的所有互動
            final_graph_edge_index = self.edge_index.to(self.device)
            
            # 對於最終評估，'train_pairs' 應排除所有訓練階段看到的項目（所有 new_interactions)
            all_seen_interactions_for_exclusion = list(dtrain_new_interactions) + self.val_data
            
            test_prec, test_rec, test_ndcg = precision_recall_ndcg_at_k(
                model_to_evaluate,
                final_graph_edge_index,
                self.test_data,  # 在保留的 test_data 上評估
                train_pairs=all_seen_interactions_for_exclusion, # 從推薦中排除所有訓練和驗證互動
                K=self.k_eval
            )
            print(f"最終測試結果: P@{self.k_eval} {test_prec:.4f} R@{self.k_eval} {test_rec:.4f} NDCG@{self.k_eval} {test_ndcg:.4f}")
            
    def train_model_on_collected_data(self,
                                   training_interactions, # (u,i) 元組列表
                                   val_interactions,       
                                   num_epochs=200,          
                                   batch_size=2048,        
                                   lr=1e-4,               
                                   lambda_reg=5e-4,        
                                   k_eval=20,               
                                   patience=20): 

        model_to_train = self.rec_model.model # 這是 LightGCN 實例
        optimizer = torch.optim.Adam(model_to_train.parameters(), lr=lr)
        
        print(f"在 {len(training_interactions)} 個互動上訓練 LightGCN，在 {len(val_interactions)} 個互動上進行驗證。")
        print(f"Epochs: {num_epochs}, Batch Size: {batch_size}, LR: {lr}, Lambda_reg: {lambda_reg}, K_eval: {k_eval}, Patience: {patience}")

        loss_hist, val_loss_hist, val_prec_hist, val_rec_hist, val_ndcg_hist = [], [], [], [], []
        best_val_ndcg = -1.0 
        patience_counter = 0

        # LightGCN 在訓練和驗證期間前向傳播所用的 edge_index
        # 是 self.edge_index，它已經用所有 training_interactions 更新過了。
        current_graph_edge_index = self.edge_index.to(self.device)

        for epoch in range(1, num_epochs + 1):
            model_to_train.train()
            t0 = time.time()

            samples = sample_pos_neg(
                training_interactions, # (u,i) 列表
                self.n_user,
                self.n_item,
                seed=epoch # 稍微改變種子以獲得不同的負樣本集
            )
                
            samples = samples[torch.randperm(len(samples))].to(self.device)
            
            total_train_loss = 0
            processed_samples_count = 0
            for st in range(0, len(samples), batch_size):
                batch = samples[st:st + batch_size]
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
                    current_graph_edge_index, 
                    val_interactions,         
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
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"由於 Val NDCG 連續 {patience} 個 epochs 沒有改善，在 epoch {epoch} 提前停止。")
                break
        
        print("完成模擬後訓練。")
    

import torch
import pandas as pd
from utility.utilis import build_edge_index, train_val, test, same_seeds
import community.community_louvain as community_louvain
from utility.bulid_dtrain_graph import build_dtrain_graph
from utility.build_nx_graph import build_nx_graph
from exposure_method.heuristic_exposure import heuristic_exposure_strategy as heuristic_exposure
from utility.plot import plot
import random
import networkx as nx

class RecSimEnv:
    def __init__(self,
                 init_edge_index,  # tensor([2,E])  初始邊索引
                 n_user,
                 n_item,
                 rec_model,
                 sim,
                 device="cuda",
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


    def run(self, n_round, k_rec, gcn_retrain_every_n_rounds=1, gcn_simulation_retrain_epochs=10):
        same_seeds(0)
        test_prec_hist, test_rec_hist, test_ndcg_hist = [], [], []
        retrain_rounds = []
        user_item_graph = build_nx_graph()
        dtrain_new_interactions = [] # 這將是一個 (u,i) 元組的列表，累積所有輪次的互動
        dtest_df = pd.read_csv('interaction_collect/dtest.csv')
        dval_df = pd.read_csv('interaction_collect/dval.csv')
        dval = []
        dtest = []
        seen = {u: set() for u in range(self.n_user)}

        for u in range(self.n_user):
            dtest_user_to_items = dtest_df.groupby('u')['i'].first()
            dval_user_to_items = dval_df.groupby('u')['i'].first()
            dtest.append((u,dtest_user_to_items.get(u)))
            dval.append((u,dval_user_to_items.get(u)))
            seen[u].add(dtest_user_to_items.get(u))
            seen[u].add(dval_user_to_items.get(u))

        for t in range(n_round):
            print(f"===== Time step {t} =====")
            current_round_interactions = [] # 當前timestep收集的互動
            total_rec_item = [] #存放每個user的推薦列表

            for u in range(self.n_user):
                per_user_interation = []
                # while len(per_user_interation) < 2:
                    # 1) 取得 Top-k 推薦
                if t < 1:
                    rec_items = list(set(range(self.n_item)) - seen[u])
                    #如果推薦清單小於k_rec
                    if len(rec_items) > k_rec:
                        rec_items = random.sample(rec_items, k_rec)
                else:
                    rec_items = self.rec_model.recommend(u, k=k_rec, exclude=seen[u])
                # rec_items  推薦的 item_ids
                for item_id_recommended in rec_items: # 直接遍歷 RS 推薦的物品
                    seen[u].add(item_id_recommended) # 用戶喜歡過的物品
                    simulator_score = self.sim.score(u, item_id_recommended).item()
                    if simulator_score > 0.6:   
                        per_user_interation.append((u, item_id_recommended))
                            # if len(per_user_interation) >= 2 and t <= 0:
                            #     break
                    # if (t == 0 and len(per_user_interation) > 0) or t > 0:
                    #     break
                # if t < 1:
                #     b = int(len(per_user_interation)/2)
                #     dval.extend(per_user_interation[:b])
                #     dtest.extend(per_user_interation[b:])
                # else:
                total_rec_item.append(rec_items)
                current_round_interactions.extend(per_user_interation)
            #t = 0先不訓練，收集dtest、dval
            # if t <= 0:
            #     print("收集val有幾個:",len(dval))
            #     print("收集test有幾個:",len(dtest))
            #     pd.DataFrame(dtest, columns=['u','i']).to_csv("interaction_collect/dtest.csv", index=False)
            #     pd.DataFrame(dval, columns=['u','i']).to_csv("interaction_collect/dval.csv", index=False)
            # else: # This block is for t = 1, 2, ...
            print("收集到幾個互動:",len(current_round_interactions))

            dtrain_new_interactions.extend(current_round_interactions)
            dtrain_graph = build_dtrain_graph(dtrain_new_interactions)
            #產生曝光邊
            # exposure = rl_exposure(agent, dtrain_new_interactions, self.item_emb, self.user_emb, self.device)
            exposure = heuristic_exposure(dtrain_graph, user_item_graph, total_rec_item, self.item_emb)

            for u, item in exposure:
                seen[u].add(item)

            # 把曝光邊接起來
            current_round_interactions.extend(exposure)
            dtrain_new_interactions.extend(exposure)

            #去除重複(u,i)
            current_round_interactions = list(set(current_round_interactions))
            dtrain_new_interactions = list(set(dtrain_new_interactions))
            dtrain_new_interactions = list(set(dtrain_new_interactions) - set(dtest) - set(dval))

            #建立edge_index
            delta_edge_index = build_edge_index(current_round_interactions, self.n_user).to(self.device)
            if delta_edge_index.numel() > 0:
                self.edge_index = torch.cat([self.edge_index, delta_edge_index], dim=1)
                self.edge_index = torch.unique(self.edge_index, dim=1)
            
            #拿dtrain再做一次louvain
            dtrain_graph = build_dtrain_graph(dtrain_new_interactions)
            connected_components = list(nx.connected_components(dtrain_graph))
            print(f"總共有 {len(connected_components)} 個連通元件")
            communities = community_louvain.best_partition(dtrain_graph, resolution=1, random_state=42)
            print(f"總社群數量: {len(set(communities.values()))}")
        
            pd.DataFrame(current_round_interactions, columns=['u','i']).to_csv(f"interaction_collect/new_interactions{t}.csv", index=False)

            if t % gcn_retrain_every_n_rounds == 0:
                retrain_rounds.append(t)
                print(f"\n--- Triggering GCN retraining and evaluation after round {t} (gcn_retrain_every_n_rounds={gcn_retrain_every_n_rounds}) ---")
                current_num_epochs = gcn_simulation_retrain_epochs

                print(f"\n--- 第 {t} 輪後進行訓練，使用目前累積的 {len(dtrain_new_interactions)} 筆真實互動，訓練 {current_num_epochs} 個 epochs ---")
                self.train_model_on_collected_data(
                    training_interactions=dtrain_new_interactions,
                    val_interactions=dval,
                    batch_size = 1024,
                    k_eval=self.k_eval,
                    num_epochs=current_num_epochs
                )
                print() # 所有輪次處理完畢後換行
                # --- 最終測試評估 ---
                print(f"\n=====第 {t} 輪模擬結束後，於測試集上進行最終評估 =====")
                print(f"將使用初始測試集中的 {len(dtest)} 筆互動進行最終評估...")

                full_edge_index  = build_edge_index(dtrain_new_interactions + dval + dtest, self.n_user).to(self.device)

                model_to_evaluate = self.rec_model.model # LightGCN 實例
                model_to_evaluate.eval()
                prec_test, rec_test, ndcg_test = test(model_to_evaluate,
                                                    num_items = self.n_item,
                                                    batch_size = 1024,
                                                    device = self.device,
                                                    train_inter = dtrain_new_interactions, 
                                                    val_inter = dval, 
                                                    test_inter = dtest,
                                                    train_edge_index = self.edge_index, full_edge_index = full_edge_index,
                                                    K = self.k_eval)
                test_prec_hist.append(prec_test)
                test_rec_hist.append(rec_test)
                test_ndcg_hist.append(ndcg_test)
            else:
                print(f"\n--- Skipping GCN retraining and evaluation after round {t} (current round interactions: {len(current_round_interactions)}, total dtrain: {len(dtrain_new_interactions)}, gcn_retrain_every_n_rounds={gcn_retrain_every_n_rounds}) ---")
        plot(test_prec_hist, test_rec_hist, test_ndcg_hist, self.k_eval, retrain_rounds)
        user_emb, item_emb = self.rec_model.model.get_user_item(full_edge_index)
        # torch.save(model.state_dict(), "lightgcn_ml1m_fixed.pth")
        torch.save(user_emb.cpu(), "model/RS/user_emb.pt")
        torch.save(item_emb.cpu(), "model/RS/item_emb.pt")
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
    

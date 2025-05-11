import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_sim_score_distribution():
    """
    讀取 rec_items_sim_scores.csv 檔案並繪製 Sim_score 的分佈圖。
    """
    try:
        df_sim_scores = pd.read_csv('rec_items_sim_scores.csv')

        if 'sim_score' in df_sim_scores.columns and not df_sim_scores['sim_score'].empty:
            # 設定繪圖風格
            sns.set_style('whitegrid')

            # 繪製 Sim_score 的分佈直方圖
            plt.figure(figsize=(12, 7))
            sns.histplot(df_sim_scores['sim_score'], kde=True, bins=30, color='skyblue', edgecolor='black')
            plt.title('Sim_score 的分佈情況', fontsize=16)
            plt.xlabel('Sim_score (模擬器餘弦相似度分數)', fontsize=14)
            plt.ylabel('頻率', fontsize=14)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.show()

            # 打印描述性統計數據
            print("\nSim_score 的描述性統計數據:")
            print(df_sim_scores['sim_score'].describe())
        elif 'sim_score' not in df_sim_scores.columns:
            print("錯誤：'rec_items_sim_scores.csv' 檔案中缺少 'sim_score' 欄位。")
            print(f"檔案中可用的欄位有: {df_sim_scores.columns.tolist()}")
        else: # df_sim_scores['sim_score'].empty is True
            print("錯誤：'rec_items_sim_scores.csv' 檔案中的 'sim_score' 欄位是空的。")

    except FileNotFoundError:
        print("錯誤：找不到 'rec_items_sim_scores.csv' 檔案。")
        print("請確保您已經運行了模擬，並且該檔案已正確生成在當前工作目錄下。")
    except pd.errors.EmptyDataError:
        print("錯誤：'rec_items_sim_scores.csv' 檔案是空的。請檢查模擬的輸出。")
    except Exception as e:
        print(f"讀取或繪製檔案時發生錯誤：{e}")

if __name__ == '__main__':
    plot_sim_score_distribution()
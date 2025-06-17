import pandas as pd
import pickle
import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


USER_EMB = torch.load("model/simulator/user_emb.pt", map_location=DEVICE)
ITEM_EMB = torch.load("model/simulator/item_emb.pt", map_location=DEVICE)


EMB_DIM = USER_EMB.size(1)
NUM_USERS = USER_EMB.shape[0]
NUM_ITEMS = ITEM_EMB.shape[0]
# 讀取資料
train_df = pd.read_csv('data/train.dat', sep=',', names=['user_id', 'movie_id', 'rating', 'timestamp'])
val_df = pd.read_csv('data/val.dat', sep=',', names=['user_id', 'movie_id', 'rating', 'timestamp'])
test_df = pd.read_csv('data/test.dat', sep=',', names=['user_id', 'movie_id', 'rating', 'timestamp'])

movies = pd.read_csv('raw/ml-1m/movies.dat', sep='::', names=['movie_id', 'title', 'genres'], engine='python', encoding='latin-1')
users   = pd.read_csv('raw/ml-1m/users.dat',  sep='::', names=['user_id', 'gender', 'age', 'occupation', 'zip'], engine='python', encoding='latin-1')

combined = pd.concat([train_df, val_df, test_df])
# 讀取映射關係
with open('mapping/uid_map.pkl', 'rb') as f:
    uid_map = pickle.load(f)
with open('mapping/mid_map.pkl', 'rb') as f:
    mid_map = pickle.load(f)

# 創建反向映射（從新ID到舊ID）
reverse_uid_map = {v: k for k, v in uid_map.items()}
reverse_mid_map = {v: k for k, v in mid_map.items()}

def get_user_interactions(user_id):

    
    # 找到該用戶的所有交互記錄
    user_interactions = combined[combined['user_id'] == user_id].copy()
    
    
    # 將movie_id轉換回原始ID
    user_interactions['original_movie_id'] = user_interactions['movie_id'].map(reverse_mid_map)
    
    # 與movies資料合併
    result = pd.merge(
        user_interactions,
        movies,
        left_on='original_movie_id',
        right_on='movie_id',
        how='left'
    )
    
    # 選擇要顯示的欄位並重新命名
    result = result[['original_movie_id', 'title', 'genres', 'rating', 'timestamp']]
    result.columns = ['Movie ID', 'Title', 'Genres', 'Rating', 'Timestamp']
    result.to_csv("mapping/11.csv")
    return result

def convert_recommendations_to_original_ids(userid):

    # 使用之前載入的 reverse_mid_map 進行轉換
    with torch.no_grad():
        score = USER_EMB[userid] @ ITEM_EMB.t()
        TOP_K = 20
        s, topk_idx = torch.topk(score, TOP_K)
        recommendations = topk_idx.cpu().tolist()
        s = s.tolist()
        for i in s:
            s1 = (float(i) + 10) /24  
            # s2 = torch.cosine_similarity(USER_EMB[userid], ITEM_EMB[i], dim=0)
            print(s1)
    original_ids = [reverse_mid_map[item_id] for item_id in recommendations]
    recommendations_with_titles = pd.DataFrame({
        'Original ID': original_ids,
        'Title': [movies[movies['movie_id'] == mid]['title'].values[0] for mid in original_ids],
        'Genres': [movies[movies['movie_id'] == mid]['genres'].values[0] for mid in original_ids]
    })
    print(recommendations_with_titles)
# 使用範例
user_id = int(input("請輸入要查詢的用戶ID: "))
original_user_id = reverse_uid_map[user_id]
user_info = users[users['user_id'] == original_user_id]
print("\n用戶詳細資料：")
print(user_info[['user_id', 'gender', 'age', 'occupation', 'zip']].to_string(index=False))
result = get_user_interactions(user_id)
if result is not None:
    print(f"\n用戶 {user_id} 的交互記錄：")
    print(result)
convert_recommendations_to_original_ids(user_id)
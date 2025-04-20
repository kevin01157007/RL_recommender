import torch
class LightGCNSimulator:
    def __init__(self, model, data):
        self.model = model
        self.model.eval()
        self.data = data
    def get_user_embedding(self, user_id, model, data):
        all_users_items = model(model.embedding_user_item.weight.clone(),
                            data["edge_index"])
        all_users = all_users_items[:len(data["users"])]
        users_emb = all_users[user_id]
        return users_emb
    def get_item_embedding(self, item_id, model, data):
        all_users_items = model(model.embedding_user_item.weight.clone(),
                            data["edge_index"])
        all_items = all_users_items[len(data["users"]):]
        item_emb = all_items[item_id]
        return item_emb
    def score(self, user_id, item_id):
        users_emb = self.get_user_embedding(user_id, self.model, self.data)
        items_emb = self.get_item_embedding(item_id, self.model, self.data)
        
        # print("User Embedding:", users_emb)
        # print("Item Embedding:", items_emb)
        
        score_value = torch.matmul(users_emb, items_emb.t())
        # print("Score Value before Sigmoid:", score_value)
        
        return self.model.f(score_value)
import torch
import torch.nn.functional as F

class SimpleSimulator:
    def __init__(self, user_emb, item_emb, device):
        """
        Initializes the simulator with pre-trained embeddings.
        Args:
            user_emb (Tensor): User embeddings [n_users, emb_size].
            item_emb (Tensor): Item embeddings [n_items, emb_size].
            device: The torch device.
        """
        self.user_emb = user_emb.to(device)
        self.item_emb = item_emb.to(device)
        self.device = device

    def score(self, user_id, item_id, current_user_emb=None, current_item_emb=None):
    
        user_embeddings_to_use = current_user_emb if current_user_emb is not None else self.user_emb
        item_embeddings_to_use = current_item_emb if current_item_emb is not None else self.item_emb

        if user_id >= user_embeddings_to_use.shape[0] or item_id >= item_embeddings_to_use.shape[0]:
            print(f"Warning: Simulator - User {user_id} or Item {item_id} out of bounds for embeddings.")
            return torch.tensor(0.0, device=self.device)

        cosine_sim = torch.cosine_similarity(user_embeddings_to_use[user_id], item_embeddings_to_use[item_id], dim=0)
        # Map from [-1, 1] to [0, 1]
        score_val = (cosine_sim + 1) / 2

        return score_val

    def update_embeddings(self, user_emb, item_emb):
        self.user_emb = user_emb.to(self.device)
        self.item_emb = item_emb.to(self.device)

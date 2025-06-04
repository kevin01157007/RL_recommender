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

        with torch.no_grad():
            cosine_sim = torch.cosine_similarity(self.user_emb[user_id], self.item_emb[item_id], dim=0)
        # Map from [-1, 1] to [0, 1]
        score_val = (cosine_sim + 1) / 2

        return score_val
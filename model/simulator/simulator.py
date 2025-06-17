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

    def score(self, user_id, item_id):

        with torch.no_grad():
            s = self.user_emb[user_id] @ self.item_emb[item_id]
        score_val = (s + 10) / 24

        return score_val
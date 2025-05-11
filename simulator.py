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
        """
        Calculates an interaction score (e.g., probability) between a user and an item.
        This is then used to sample a binary interaction event (0 or 1) from a Bernoulli distribution.
        Args:
            user_id (int): The user ID.
            item_id (int): The item ID.
        Returns:
            Tensor: A scalar tensor representing the binary interaction event (0 or 1).
        """
        if user_id >= self.user_emb.shape[0] or item_id >= self.item_emb.shape[0]:
         
             print(f"Warning: Simulator - User {user_id} or Item {item_id} out of bounds for embeddings.")
             return torch.tensor(0.0, device=self.device)

        cosine_sim = torch.cosine_similarity(self.user_emb[user_id], self.item_emb[item_id], dim=0)
        # Map from [-1, 1] to [0, 1]
        score_val = (cosine_sim + 1) / 2

        return score_val
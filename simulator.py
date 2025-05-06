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
        Args:
            user_id (int): The user ID.
            item_id (int): The item ID.
        Returns:
            Tensor: A scalar tensor representing the interaction score (e.g., probability).
        """
        if user_id >= self.user_emb.shape[0] or item_id >= self.item_emb.shape[0]:
             # Handle potential out-of-bounds, return 0 probability
             print(f"Warning: Simulator - User {user_id} or Item {item_id} out of bounds for embeddings.")
             return torch.tensor(0.0, device=self.device)

        # Calculate dot product
        dot_product = (self.user_emb[user_id] * self.item_emb[item_id]).sum()
        # Apply sigmoid to get a probability-like score between 0 and 1
        score_val = torch.sigmoid(dot_product)
        return score_val
import torch
import numpy as np
from collections import defaultdict
import networkx as nx
import community as community_louvain  # python-louvain package

class RecommenderSimulator:
    def __init__(self, lightgcn_model, user_item_graph, train_data, test_data):
        """
        Initialize the simulator environment
        
        Args:
            lightgcn_model: Pre-trained LightGCN model to act as the simulator
            user_item_graph: Bipartite graph of user-item interactions
            train_data: Initial training data (user-item pairs)
            test_data: Fixed test set for evaluation
        """
        self.model = lightgcn_model
        self.graph = user_item_graph
        self.train_data = train_data
        self.test_data = test_data
        self.current_train = train_data.copy()
        
        # Initialize communities
        self._detect_communities()
        
        # Track simulation state
        self.time_step = 0
        self.history = {
            'precision': [],
            'recall': [],
            'ndcg': [],
            'num_communities': []
        }
    
    def _detect_communities(self):
        """Detect communities in the user-item graph using Louvain algorithm"""
        # Convert to networkx graph for community detection
        nx_graph = nx.Graph()
        
        # Add nodes (users and items)
        user_nodes = [f"u_{u}" for u in self.graph.users]
        item_nodes = [f"i_{i}" for i in self.graph.items]
        nx_graph.add_nodes_from(user_nodes, bipartite=0)
        nx_graph.add_nodes_from(item_nodes, bipartite=1)
        
        # Add edges
        for u, items in enumerate(self.current_train):
            for i in items:
                nx_graph.add_edge(f"u_{u}", f"i_{i}")
        
        # Detect communities
        self.communities = community_louvain.best_partition(nx_graph)
        self.num_communities = len(set(self.communities.values()))
        
        # Store community membership
        self.user_communities = defaultdict(list)
        self.item_communities = defaultdict(list)
        for node, comm_id in self.communities.items():
            if node.startswith('u_'):
                user_id = int(node[2:])
                self.user_communities[comm_id].append(user_id)
            else:
                item_id = int(node[2:])
                self.item_communities[comm_id].append(item_id)
    
    def step(self, exposure_edges=None):
        """
        Perform one simulation step:
        1. Generate recommendations using current model
        2. Simulate user feedback
        3. Update training data
        4. Retrain model
        5. Evaluate metrics
        
        Args:
            exposure_edges: List of additional (user, item) pairs to expose
            
        Returns:
            dict: Updated metrics and state
        """
        self.time_step += 1
        
        # 1. Generate recommendations (top-N for each user)
        recommendations = self._generate_recommendations()
        
        # 2. Simulate user feedback using LightGCN as oracle
        feedback = self._simulate_feedback(recommendations)
        
        # 3. Add controlled exposure edges if provided
        if exposure_edges:
            feedback.extend(exposure_edges)
            
        # 4. Update training data
        self.current_train.extend(feedback)
        
        # 5. Retrain model (can be optional or partial updates)
        self._update_model()
        
        # 6. Re-detect communities
        self._detect_communities()
        
        # 7. Evaluate metrics
        metrics = self._evaluate()
        
        # Update history
        self.history['precision'].append(metrics['precision'])
        self.history['recall'].append(metrics['recall'])
        self.history['ndcg'].append(metrics['ndcg'])
        self.history['num_communities'].append(self.num_communities)
        
        return metrics
    
    def _generate_recommendations(self, top_n=20):
        """Generate top-N recommendations for each user"""
        all_users = torch.arange(len(self.graph.users))
        all_ratings = self.model(all_users, None)  # Get all ratings
        
        recommendations = {}
        for user in range(len(self.graph.users)):
            # Get top-N items not already interacted with
            user_ratings = all_ratings[user]
            interacted = set(self.current_train[user])
            candidates = [i for i in range(len(user_ratings)) if i not in interacted]
            
            if candidates:
                top_items = torch.topk(user_ratings[candidates], min(top_n, len(candidates))).indices
                recommendations[user] = [candidates[i] for i in top_items]
            else:
                recommendations[user] = []
                
        return recommendations
    
    def _simulate_feedback(self, recommendations):
        """
        Simulate user feedback using the LightGCN model as oracle
        Returns list of (user, item) pairs that received positive feedback
        """
        feedback = []
        
        for user, items in recommendations.items():
            if not items:
                continue
                
            # Get predicted ratings for recommended items
            user_tensor = torch.tensor([user] * len(items))
            item_tensor = torch.tensor(items)
            pred_ratings = self.model(user_tensor, item_tensor)
            
            # Simulate feedback - users "click" items with probability based on rating
            prob = torch.sigmoid(pred_ratings)
            clicks = torch.bernoulli(prob)
            
            for item, click in zip(items, clicks):
                if click > 0.5:  # Threshold for positive feedback
                    feedback.append((user, item))
                    
        return feedback
    
    def _update_model(self):
        """Update the recommendation model with new training data"""
        # In practice you might want to do incremental training
        # Here we'll just retrain from scratch for simplicity
        # You can replace this with your LightGCN training code
        
        # For demo purposes, we'll just do a partial update
        pass
    
    def _evaluate(self):
        """Evaluate model performance on test set"""
        # Calculate precision, recall, NDCG
        # This would use your existing evaluation code
        
        # For demo, return dummy values
        return {
            'precision': np.random.uniform(0.05, 0.07),
            'recall': np.random.uniform(0.0015, 0.0017),
            'ndcg': np.random.uniform(0.06, 0.07)
        }
    
    def get_state(self):
        """Get current simulation state"""
        return {
            'time_step': self.time_step,
            'train_data_size': len(self.current_train),
            'num_communities': self.num_communities,
            'history': self.history
        }
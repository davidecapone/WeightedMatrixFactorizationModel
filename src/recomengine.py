from src.wmfact import WeightedMatrixFactorization
import numpy as np

class RecommenderEngine:
    def __init__(self):
        
        self.users_embedding = None
        self.items_embedding = None
        

    def recommend(self, user_id, n_items=10):
        """
        Recommend n_items items for the user with user_id
        """
        pass

    def recommend_similar_items(self, item_id, n_items=10):
        """
        Recommend n_items items similar to item_id
        """
        pass

    def recommend_similar_users(self, user_id, n_items=10):
        """
        Recommend n_items items for users similar to user_id
        """
        pass

    def recommend_similar_items_to_users(self, user_id, n_items=10):
        """
        Recommend n_items items similar to items rated by user_id
        """
        pass
    

        
    
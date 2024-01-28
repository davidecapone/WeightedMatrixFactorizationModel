from src.wmfact import WeightedMatrixFactorization
import numpy as np

class RecommenderEngine:
    def __init__(self, data):
        
        self.user_embedding = np.array([])
        self.item_embedding = np.array([])

        self.wmf = WeightedMatrixFactorization(data)
        self.data = None

    def learn(self):
        pass

        
    
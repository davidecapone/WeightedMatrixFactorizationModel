import numpy as np
import pandas as pd
from src.wmf import WeightedMatrixFactorization

DATA_PATH = "./data/"
PARAMS_PATH = "./"

if __name__ == "__main__":

    # Load data
    products = pd.read_csv(DATA_PATH + "products.csv")
    ratings = pd.read_csv(DATA_PATH + "ratings.csv")

    print("Number of products: {}".format(len(products)))
    
    # take a subset of the data
    ratings = ratings.sample(n=5000, random_state=1)
    print("Number of ratings: {}".format(len(ratings)))

    feedbacks = pd.pivot_table(ratings, values='rating', index=['user_id'], columns=['asin']).values
    print("Feedback shape: {}".format(feedbacks.shape))

    # Create model
    model = WeightedMatrixFactorization(
        feedbacks, 
        n_latents=100,
        n_iter=30,
        w_obs=1.0,
        w_unobs=0.1,
        lambda_reg=0.05,
        #seed=50
    )

    history = model.fit(
        method='WALS', 
        verbose=True
    )

    


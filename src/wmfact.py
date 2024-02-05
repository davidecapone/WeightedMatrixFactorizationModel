"""
This class implements the Weighted Matrix Factorization model using the Weighted Alternating Least Squares (WALS) algorithm. 
It provides functionalities for fitting the model, predicting ratings, and retrieving user and item embeddings.

It learns low-dimensional representations (embeddings) of users and items from observed ratings, 
and predicts ratings for user-item pairs based on these embeddings.

Usage:
>>> model = WeightedMatrixFactorization(feedbacks, n_latents=100, n_iter=20, lambda_reg=0.05)
>>> hist = model.fit(method='wals')
>>> user_emb, item_emb = model.get_embeddings()
>>> predicted_ratings = model.predict()
"""
import numpy as np
from numpy.linalg import solve
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

class WeightedMatrixFactorization():

  def __init__(self, feedbacks:np.ndarray, 
               seed:int=None,
               n_latents:int=100, 
               n_iter:int=20, 
               w_obs:float=1.0, 
               w_unobs:float=0.1,
               lambda_reg:float=0.05) -> None:
    
    # feedbacks: matrix of ratings (n_users x n_items)
    self.feedbacks = np.nan_to_num( np.array(feedbacks), 0 ) # all NaNs are replaced by 0

    # observed_data: boolean matrix of observed ratings (n_users x n_items)
    self.observed_data = ~np.isnan( feedbacks ) # True=observed ratings, False=unobserved ratings
    
    # getting the number of users and items:
    self.n_users, self.n_items = feedbacks.shape

    # hyperparameters:
    self.n_iter = n_iter  # number of iterations
    self.n_latents = n_latents # number of latent factors
    self.lambda_reg = lambda_reg # regularization parameter
    self.w_obs = w_obs # weight of observed 
    self.w_unobs = w_unobs # weight of unobserved
    
    if seed is not None: # set the seed for reproducibility?
      np.random.seed(seed)

  def fit(self, method:str='wals') -> dict:
    """
    Fits the model using the specified method and returns a dictionary containing
    the history of loss function values during training.

    Parameters:
    - method (str): The method to use for fitting the model. Currently supported methods are 'WALS' and 'SGD'.

    Returns:
    - hist (dict): A dictionary containing the history of loss function values during training.

    Raises:
    - NotImplementedError: If the specified method is not implemented.

    Usage:
    >>> model.fit(method='wals')
    >>> # or
    >>> model.fit()  # Default method is 'wals'
    """

    # random initialization of user and item embeddings:
    self.users_embedding = np.random.rand(self.n_users, self.n_latents)
    self.items_embedding = np.random.rand(self.n_items, self.n_latents)

    print(f"* Fitting the model with {method} method: n_iter = {self.n_iter}, n_latents = {self.n_latents}, lambda_reg = {self.lambda_reg} *")

    hist = {}     # history of loss function values
    match method: # select the method to use for fitting the model
      case 'wals':
        hist = self.__wals_method() 
      case _:
        raise NotImplementedError(f"Method {method} not implemented.")
      
    return hist
  
  def __wals_method(self) -> dict:
    """
    Perform Weighted Alternating Least Squares (WALS) algorithm.

    This method iterates over the specified number of iterations, updating user and item embeddings
    alternatively, and computes the loss function value at each iteration.

    Returns:
    - history (dict): A dictionary containing the loss function values for each iteration.
    """
    history = {}

    with tqdm(total=self.n_iter) as pbar:
      for i in range(self.n_iter):  # iterate over the number of iterations

        with ThreadPoolExecutor() as executor: # ** parallelize the updates of user and item embeddings **
          executor.submit(self.__update_users_embedding)
          executor.submit(self.__update_items_embedding)


        loss = np.sum(  # compute the loss function value at iteration i
          np.where(
              self.observed_data, # apply the formula only to observed ratings
              (self.feedbacks - self.users_embedding @ self.items_embedding.T) ** 2,
              0
          )
        )

        history[i] = loss # loss i-th iteration
        pbar.set_postfix(loss=f"{loss:.2f}")
        pbar.update(1)

    return history

  def __update_users_embedding(self) -> None:
    """
    Update the user matrix based on the observed data, weights, and regularization.

    This method iteratively updates the user embeddings using the observed feedback data,
    weights for observed and unobserved values, and regularization.

    Returns:
    - None

    ** This function is parallelized with the ThreadPoolExecutor in the __wals_method function. **
    """

    for user_idx in range( self.n_users ):  # iterate over all users

      weight_matrix = np.diag(  # diagonal matrix with weights for observed & unobserved ratings
        
          np.where(   # if observed, then w_obs/(# obs), else w_unobs/(# unobs)
              self.observed_data[user_idx, :],
              self.w_obs / sum(self.observed_data[user_idx, :]),    # normalize the weight for obs ratings
              self.w_unobs / sum(~self.observed_data[user_idx, :])  # Normalize the weight for unobs ratings
          )

      ) # --> shape: (n_items x n_items)

      # lambda_reg * I (n_latents x n_latents) is a diagonal matrix with lambda_reg in the diagonal and 0 elsewhere.
      regularization = self.lambda_reg * np.eye(self.n_latents) 

      self.users_embedding[user_idx,:] = np.linalg.solve(
        self.items_embedding.T @ weight_matrix @ self.items_embedding + regularization, # n_latents x n_latents
        self.items_embedding.T @ weight_matrix @ self.feedbacks[user_idx, :] # n_latents x 1
      )
      '''
      Explanation:
      - select the row in the users_embedding matrix corresponding to the current user_idx (shape: 1 x n_latents)
      - solve the linear system of equations Ax = b, 
        where A = items_embedding.T @ weight_matrix @ items_embedding + regularization (shape: n_latents x n_latents)
        and b = items_embedding.T @ weight_matrix @ feedbacks[user_idx, :] (shape: n_latents x 1)
      '''
    return
  
  def __update_items_embedding(self) -> None:
    """
    Update the item matrix based on the observed data, weights, and regularization.

    This method iteratively updates the item embeddings using the observed feedback data,
    weights for observed and unobserved values, and regularization.

    Returns:
    - None

    ** This function is parallelized with the ThreadPoolExecutor in the __wals_method function. **
    """

    for item_idx in range( self.n_items ):  # iterate over all items
      
      weight_matrix = np.diag(  # diagonal matrix with weights for observed & unobserved ratings
        
          np.where(   # if observed, then w_obs/(# obs), else w_unobs/(# unobs)
              self.observed_data[:,item_idx],
              self.w_obs / sum(self.observed_data[:, item_idx]),    # normalize the weight for obs ratings
              self.w_unobs / sum(~self.observed_data[:, item_idx])  # Normalize the weight for unobs ratings
          )

      ) # --> shape: (n_users x n_users)
      
      # lambda_reg * I (n_latents x n_latents) is a diagonal matrix with lambda_reg in the diagonal and 0 elsewhere.
      regularization = self.lambda_reg * np.eye(self.n_latents) 

      self.items_embedding[item_idx, :] = solve(
        self.users_embedding.T @ weight_matrix @ self.users_embedding + regularization,
        self.users_embedding.T @ weight_matrix @ self.feedbacks[:, item_idx]
      )
      '''
      Explanation:
      - select the row in the items_embedding matrix corresponding to the current item_idx (shape: 1 x n_latents)
      - solve the linear system of equations Ax = b, 
        where A = users_embedding.T @ weight_matrix @ users_embedding + regularization (shape: n_latents x n_latents)
        and b = users_embedding.T @ weight_matrix @ feedbacks[:, item_idx] (shape: n_latents x 1)
      '''
    return
  
  def compute_rmse(feedback_matrix, predicted_matrix):
    """
    Compute the Root Mean Squared Error (RMSE) between the predicted ratings and the actual ratings.

    Parameters:
    - feedback_matrix (numpy.ndarray): The actual ratings matrix with NaN values for non-ratings.
    - predicted_matrix (numpy.ndarray): The predicted ratings matrix.

    Returns:
    - rmse (float): The Root Mean Squared Error (RMSE) between the predicted and actual ratings.
    """
    
    mask = ~np.isnan(feedback_matrix) # boolean matrix, True=observed ratings, False=unobserved ratings (NaNs)
    squared_error = (predicted_matrix[mask] - feedback_matrix[mask]) ** 2 # squared error
    rmse = np.sqrt( np.mean(squared_error) )  # RMSE
    return rmse

  def predict(self):
    """
    Predicts ratings for user-item pairs based on the learned embeddings.

    Returns:
    - predicted_ratings (numpy.ndarray): Predicted ratings matrix, where each row represents a user
      and each column represents an item.

    Usage:
    >>> predicted_ratings = model.predict()
    """
    return self.users_embedding @ self.items_embedding.T
  
  def get_embeddings(self) -> tuple:
    """
    Return the user and item embeddings as a tuple.

    Returns:
    - embeddings (tuple): A tuple containing the user embeddings and item embeddings.

    Usage:
    >>> user_emb, item_emb = model.get_embeddings()
    """
    return self.users_embedding, self.items_embedding
  

  """def grid_search(self, params:dict) -> tuple:
    
    Perform grid search for the parameters
    
    best_params = None
    best_score = np.inf

    param_combinations = [dict(zip(params.keys(), values)) for values in itertools.product(*params.values())]

    for params in param_combinations:
      self.n_latents = params['n_latents']
      self.n_iter = params['n_iter']
      #self.w_obs = params['w_obs']
      #self.w_unobs = params['w_unobs']
      self.lambda_reg = params['lambda_reg']

      hist = self.fit(method='wals', verbose=True)
      score = hist[self.n_iter-1]

      if score < best_score:
        best_score = score
        best_params = params

    return best_params, best_score
    
    pass"""
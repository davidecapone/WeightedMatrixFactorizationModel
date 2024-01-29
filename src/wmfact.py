import numpy as np
from numpy.linalg import solve
import time
from concurrent.futures import ThreadPoolExecutor

class WeightedMatrixFactorization():

  def __init__(self, feedbacks:np.ndarray, 
               seed:int=None,
               n_latents:int=100, 
               n_iter:int=20, 
               w_obs:float=1.0, 
               w_unobs:float=0.1,
               lambda_reg:float=0.05) -> None:
    
    self.feedbacks = np.nan_to_num( np.array(feedbacks), 0 ) # replace NaNs with 0s
    self.observed_data = ~np.isnan( feedbacks ) # True --> observed ratings, 
                                                # False --> unobserved ratings
    
    self.n_users, self.n_items = feedbacks.shape # get num. of users and items from the feedback matrix
    self.n_iter = n_iter  # num. of iterations for the algorithm
    self.n_latents = n_latents # num. of latent factors
    self.w_obs = w_obs # weight of observed 
    self.w_unobs = w_unobs # weight of unobserved 
    self.lambda_reg = lambda_reg # regularization parameter
    
    if seed is not None: 
      np.random.seed(seed)  # seed for reproducibility

    # random values for the user and item matrices:
    self.users_embedding = np.random.rand(self.n_users, n_latents) # n_users x n_latents
    self.items_embedding = np.random.rand(self.n_items, n_latents) # n_items x n_latents

    self.verbose = False

  def fit(self, method:str='WALS', verbose:str=False) -> dict:
    """
    Fit the model
    """

    if verbose:
      self.verbose = True

    print(f"** fitting the model with {method} method **")
    start = time.process_time()   # start timer
    hist = {}                     # dict. for loss function values

    match method:
      case 'wals':
        hist = self.__wals_method(verbose) 
      case 'sgd':
        raise NotImplementedError(f"Method {method} not implemented.")
        #hist = self.__sgd_method(verbose)
      case _:
        raise NotImplementedError(f"Method {method} not implemented.")
      
    end = time.process_time()                       # end timer
    print(f"** done in {end-start:.2f} seconds **") # print elapsed time
    return hist
  
  def __sgd_method(self, verbose) -> None:
    # TODO: implement this method
    pass
  
  def __wals_method(self, verbose) -> dict:
    """
    Perform Weighted Alternating Least Squares
    """
    history = {} # to store the loss function values

    for i in range(self.n_iter):

      # parallelize the update of the user and item matrices:
      with ThreadPoolExecutor() as executor:
        executor.submit(self.__update_users_embedding)
        executor.submit(self.__update_items_embedding)

      # LOSS FUNCTION: 
      loss = np.sum(
        np.where(
            self.observed_data, # apply the formula only to observed ratings
            (self.feedbacks - self.users_embedding @ self.items_embedding.T) ** 2,
            0
        )
      )

      history[i] = loss # store loss function value at iteration i
      if self.verbose:
        print(f"Loss: {loss:.3f}, iteration: {i+1}/{self.n_iter}")

    return history

  def __update_users_embedding(self) -> None:
    """
    Update the user matrix
    """

    for user_idx in range(self.n_users):
      # iterate over the users

      # Weight matrix for observed and unobserved values
      weight_matrix = np.diag(
          np.where( 
              self.observed_data[user_idx, :],
              self.w_obs / sum(self.observed_data[user_idx, :]), # Normalize the weight for observed ratings
              self.w_unobs / sum(~self.observed_data[user_idx, :]) # Normalize the weight for unobserved ratings
          )
      )


      # lambda_reg*np.eye(self.n_latents) is a diagonal matrix with lambda_reg in the diagonal and 0 elsewhere.
      regularization = self.lambda_reg * np.eye(self.n_latents)
      
      # Solve the system of linear equations
      self.users_embedding[user_idx,:] = np.linalg.solve(
        self.items_embedding.T @ weight_matrix @ self.items_embedding + regularization, # n_latents x n_latents
        self.items_embedding.T @ weight_matrix @ self.feedbacks[user_idx, :] # n_latents x 1
      )
    return
  
  def __update_items_embedding(self) -> None:
    """
    Update the item matrix
    """

    for item_idx in range(self.n_items):

      # Weight matrix for observed and unobserved values
      # TODO: refactor this code, divide by zero warning...
      weight_matrix = np.diag(
          np.where(
              self.observed_data[:,item_idx],
              self.w_obs / sum(self.observed_data[:, item_idx]) , # Normalize the weight for observed ratings
              self.w_unobs / sum(~self.observed_data[:, item_idx]) # Normalize the weight for unobserved ratings
          )
      )

      # Regularization term
      regularization = self.lambda_reg * np.eye(self.n_latents)

      # Solve the system of linear equations using spsolve
      self.items_embedding[item_idx, :] = solve(
        self.users_embedding.T @ weight_matrix @ self.users_embedding + regularization,
        self.users_embedding.T @ weight_matrix @ self.feedbacks[:, item_idx]
      )
    return
  
  def predict(self, user_idx:int, item_idx:int) -> float:
    """
    Predict the rating for a given user and item
    """
    return self.users_embedding[user_idx, :] @ self.items_embedding[item_idx, :].T

  def predict_all(self) -> np.ndarray:
    """
    Predict the ratings for all users and items
    """
    return self.users_embedding @ self.items_embedding.T
  
  def save(self, path:str) -> None:
    """
    Save the model
    """

    with open(path, 'wb') as f:
      pickle.dump(self, f)
    return
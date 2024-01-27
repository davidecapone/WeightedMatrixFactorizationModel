import numpy as np
from numpy.linalg import solve

class WeightedMatrixFactorization():

  def __init__(self, feedback_matrix, 
               seed:int=None,
               n_latents:int=100, 
               n_iter:int=20, 
               w_obs:float=1.0, 
               w_unobs:float=0.1,
               lambda_reg:float=0.05) -> None:
    
    self.feedback_matrix = np.nan_to_num(np.array(feedback_matrix), 0) # Replace NaN values with 0
    self.observed_data = ~np.isnan(feedback_matrix) # Create a boolean matrix where True values are observed ratings, False values are unobserved ratings
    
    self.n_users, self.n_items = feedback_matrix.shape # number of users and items
    self.n_iter = n_iter # number of iterations
    self.n_latents = n_latents # number of latent factors
    self.w_obs = w_obs # weight of observed 
    self.w_unobs = w_unobs # weight of unobserved 
    self.lambda_reg = lambda_reg  # regularization parameter
    
    if seed is not None: 
      np.random.seed(seed)  # set the seed for reproducibility

    # initialize the user and item matrices with random values
    self.user_matrix = np.random.rand(self.n_users, n_latents)  
    self.item_matrix = np.random.rand(self.n_items, n_latents)  

  def fit(self, method:str='WALS', verbose:str=False) -> dict:

    # matrix factorization method:
    match method:
      case 'WALS':
        hist = self.__wals_method(verbose) 
      case _:
        raise NotImplementedError(f"Method {method} not implemented.")
    return hist
  
  def __wals_method(self, verbose) -> dict:
    
    history = {} # to store the loss function values

    for _ in range(self.n_iter):

      self.__update_user_matrix()
      self.__update_item_matrix()

      # calculate the loss function value
      loss = np.sum(
        np.where(
            self.observed_data,
            (self.feedback_matrix - self.user_matrix @ self.item_matrix.T) ** 2,
            0
        )
      )

      history[_] = loss # store the loss function value

      if verbose:
        print(f"Loss: {loss:.3f}, iteration: {_+1}/{self.n_iter}")

    return history

  def __update_user_matrix(self) -> None:
    
    for user_idx in range(self.n_users):

      # Weight matrix for observed and unobserved values
      weight_matrix = np.diag(
          np.where(
              self.observed_data[user_idx, :],
              self.w_obs / sum(self.observed_data[user_idx, :]), # Normalize the weight for observed ratings
              self.w_unobs / sum(~self.observed_data[user_idx, :]) # Normalize the weight for unobserved ratings
          )
      )

      # Regularization term
      regularization = self.lambda_reg * np.eye(self.n_latents)
      
      # Solve the system of linear equations
      self.user_matrix[user_idx,:] = solve(
          self.item_matrix.T @ weight_matrix @ self.item_matrix + regularization,
          self.item_matrix.T @ weight_matrix @ self.feedback_matrix[user_idx, :]
      )
  
    return
  
  def __update_item_matrix(self) -> None:

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
      self.item_matrix[item_idx, :] = solve(self.user_matrix.T @ weight_matrix @ self.user_matrix + regularization,
                                          self.user_matrix.T @ weight_matrix @ self.feedback_matrix[:, item_idx])
    return
  
  def get_users_embedding(self) -> np.array:
    return self.user_matrix
  
  def get_items_embedding(self) -> np.array:
    return self.item_matrix
  
"""
This class implements the Weighted Matrix Factorization model.
It provides functionalities for fitting the model, predicting ratings, and retrieving user and item embeddings.
"""

import numpy as np
from numpy.linalg import solve
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import pickle
import json
from datetime import datetime
import dill
import time

MODELS_PATH = 'models/'

class WeightedMatrixFactorization():

  def __init__(self, feedbacks:np.ndarray, 
               n_latents:int=100, 
               n_iter:int=15, 
               w_obs:float=1.0, 
               w_unobs:float=0.05,
               lambda_reg:float=0.05) -> None:
    
    # all NaNs in the feedback matrix are replaced with 0:
    self.feedbacks = np.nan_to_num( np.array(feedbacks), 0 ) 

    # boolean matrix of observed ratings (shape: same as feedbacks matrix, True=observed ratings, False=unobserved ratings):
    self.observed_data = ~np.isnan( feedbacks )
    
    # getting the number of users and items:
    self.n_users, self.n_items = feedbacks.shape

    # Hyperparameters:
    self.n_iter = n_iter          # number of iterations for the algorithm
    self.n_latents = n_latents    # number of latent factors
    self.lambda_reg = lambda_reg  # regularization parameter
    self.w_obs = w_obs            # weight of observed 
    self.w_unobs = w_unobs        # weight of unobserved

    # user and item embeddings:
    self.users_embedding = None
    self.items_embedding = None

  def fit(self, method:str='wals', seed:int=None, dump:bool=True) -> dict:
    """
    Fit the model using the specified method and return a dictionary containing the history of loss function values
    during training. The model and its history are saved to disk.

    Parameters:
    - method (str): The method to use for fitting the model.
    - seed (int): An optional seed value to set for reproducibility.
    - dump (bool): A flag to indicate whether to save the model and its history to disk. Default is True.

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
    np.random.seed(seed) if seed is not None else None
    self.users_embedding = np.random.rand(self.n_users, self.n_latents) # shape: (n_users x n_latents)
    self.items_embedding = np.random.rand(self.n_items, self.n_latents) # shape: (n_items x n_latents)

    print(f"* Fitting the model with {method} method: n_iter = {self.n_iter}, n_latents = {self.n_latents}, lambda_reg = {self.lambda_reg} *")
    # take the time before fitting the model:
    start_time = time.time()
    hist = {}     # history of loss function values

    match method: # select the method to use for fitting the model
      case 'wals':
        hist = self.__wals_method() 

      case 'sgd':
        hist = self.__sgd_method()

      case _:
        raise NotImplementedError(f"Method {method} not implemented.")

    print(f"\n-> Model fitting completed in {time.time()-start_time:.2f} seconds")

    if dump:
      # save the filename with method and params used:
      filename = f"wmf_{method}_nlat{self.n_latents}_niter{self.n_iter}_lambdareg{self.lambda_reg}.pkl"

      self.__save(MODELS_PATH + filename)
      print(f"\n-> Model saved to {MODELS_PATH + filename}")

      # dump the history to disk:
      with open(MODELS_PATH + filename.replace('.pkl', '.json'), 'w') as file:
        json.dump(hist, file)

      print(f"-> History saved to {MODELS_PATH + filename.replace('.pkl', '.csv')}\n")
    

    return hist
  
  def __wals_method(self) -> dict:
    """
    Perform Weighted Alternating Least Squares (WALS) algorithm.

    This method iterates over the specified number of iterations, updating user and item embeddings alternatively, 
    and computes the loss function value at each iteration.

    Returns:
    - history (dict): A dictionary containing the loss function values for each iteration.
    """

    history = {}
    with tqdm(total=self.n_iter) as pbar: # only for progress bar
      for i in range(self.n_iter):  # iterate over the number of iterations

        with ThreadPoolExecutor() as executor: # ! parallelising the updates of user and item embeddings !
          executor.submit(self.__update_users_embedding)
          executor.submit(self.__update_items_embedding)

        loss = np.sum(  # loss function value at iteration i
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
    Update the user matrix based on the observed data, weights.

    This method iteratively updates the user embeddings using the observed feedback data,
    weights for observed and unobserved values.

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
    Update the item matrix based on the observed data, weights.

    This method iteratively updates the item embeddings using the observed feedback data,
    weights for observed and unobserved values.

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
  
  def __sgd_method(self):
    """
    Perform Stochastic Gradient Descent (SGD) algorithm.
    This method updates the user and item embeddings iteratively using stochastic gradient descent.
    
    Returns:
    - history (dict): A dictionary containing the history of loss function values for each iteration.
    """
    history = {}
    self.learning_rate = 0.01  # learning rate for SGD

    for epoch in range(self.n_iter):  # iterate over the number of epochs
      total_loss = 0

      for i in range(self.n_users):
        for j in range(self.n_items):

          if self.feedbacks[i, j] > 0:  # check if the rating is observed

            dot_product = np.dot(self.users_embedding[i, :], self.items_embedding[j, :].T)
            error = self.feedbacks[i, j] - dot_product
            total_loss += error ** 2  # compute total loss

            # Update user and item embeddings using stochastic gradient descent
            user_gradient = -2 * error * self.items_embedding[j, :] + 2 * self.lambda_reg * self.users_embedding[i, :]
            item_gradient = -2 * error * self.users_embedding[i, :] + 2 * self.lambda_reg * self.items_embedding[j, :]

            self.users_embedding[i, :] -= self.learning_rate * user_gradient
            self.items_embedding[j, :] -= self.learning_rate * item_gradient
            print(f"Epoch {epoch+1}/{self.n_iter}, User {i+1}/{self.n_users}, Item {j+1}/{self.n_items}, Loss: {total_loss:.2f}", end='\r')

      # Save total loss for current epoch in history
      history[epoch] = total_loss

    return history
  
  def get_embeddings(self) -> tuple:
    """
    Return the user and item embeddings as a tuple.

    Returns:
    - embeddings (tuple): A tuple containing the user embeddings and item embeddings.

    Usage:
    >>> user_emb, item_emb = model.get_embeddings()
    """
    return self.users_embedding, self.items_embedding

  def predict_all(self):
    """
    Predicts ratings for user-item pairs based on the learned embeddings.

    Returns:
    - predicted_ratings (numpy.ndarray): Predicted ratings matrix, where each row represents a user
      and each column represents an item.

    Usage:
    >>> predicted_ratings = model.predict()
    """
    return self.users_embedding @ self.items_embedding.T

  def __save(self, filename:str) -> None:
    """
    Save the model to a file using pickle.

    Parameters:
    - filename (str): The name of the file to save the model to.

    Returns:
    - None

    Usage:
    >>> model.save('model.pkl')
    """

    # seroaize the model using dill:
    with open(filename, 'wb') as file:
      dill.dump(self, file)

  @staticmethod
  def load(filename:str) -> 'WeightedMatrixFactorization':
    """
    Load a model from a file using pickle.

    Parameters:
    - filename (str): The name of the file to load the model from.

    Returns:
    - model (WeightedMatrixFactorization): The loaded model.

    Usage:
    >>> model = WeightedMatrixFactorization.load('model.pkl')
    """
    with open(filename, 'rb') as file:
      model = pickle.load(file)
    return model


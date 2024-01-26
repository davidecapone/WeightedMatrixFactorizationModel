import numpy as np

class WeightedMatrixFactorization():
  """
  Weighted Matrix Factorization class.
  """

  def __init__(self, feedback_matrix, 
               n_latents:int=100, 
               n_iter:int=20, 
               w_obs:float=1.0, 
               w_unobs:float=0.1) -> None:
    """
    Initialize the Weighted Matrix Factorization.
    """
    self.feedback_matrix = feedback_matrix

    # get the number of users and items:
    self.n_users, self.n_items = self.feedback_matrix.shape

    self.n_iter = n_iter
    self.n_latents = n_latents

    # weights for observed and unobserved values:
    self.w_obs = w_obs
    self.w_unobs = w_unobs

    # filling random values in the two matrices:
    self.user_matrix = np.random.rand(self.n_users, self.n_latents)
    self.item_matrix = np.random.rand(self.n_items, self.n_latents)


  def fit(self, method='WALS') -> None:
    """
    Fit the model using the specified method.

    Args:
      method (str): the method to use for matrix factorization. Defaults to 'WALS'.

    Returns:
      None
    """

    match method:
      case 'WALS':
        pass
      case _:
        raise NotImplementedError(f"Method {method} not yet implemented.")

    return

  def update_user_matrix(self):
    """
    Update the user matrix.

    Args:
      None

    Returns:
      None
    """
    pass

  def update_item_matrix(self):
    """
    Update the item matrix.

    Args:
      None

    Returns:
      None
    """
    pass

  def loss(self):
    """
    Compute the loss for the current iteration.

    Args:
      None

    Returns:
      None
    """
    pass

  def predict(self):
    """
    Predict the ratings for the users and items.

    Args:
      None

    Returns:
      None
    """
    pass
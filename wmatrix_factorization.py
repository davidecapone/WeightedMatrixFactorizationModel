import numpy as np

class WeightedMatrixFactorization():
  """
  Weighted Matrix Factorization class.
  """

  def __init__(self, feedback_matrix, n_latents=100, n_iter=20) -> None:
    """
    Initialize the Weighted Matrix Factorization.
    """
    self.feedback_matrix = feedback_matrix
    self.n_latents = n_latents
    self.n_iter = n_iter

    # get the number of users and items:
    self.n_users, self.n_items = self.feedback_matrix.shape

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

    if method == 'WALS':
      pass
    else:
      pass

    return


  def predict(self):
    """
    Predict the ratings for the users and items.

    Args:
      None

    Returns:
      None
    """
    pass
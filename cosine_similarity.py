import numpy as np

def cosine_similarity(v1, v2) -> float:
    """
    Compute the cosine similarity between two numpy vectors.

    Args:
        v1 (numpy array): the first vector
        v2 (numpy array): the second vector

    Returns:
        float: the cosine similarity between v1 and v2
    """

    # cosine similarity is defined as (A . B) / (||A|| * ||B||)
    cosine = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return cosine
    
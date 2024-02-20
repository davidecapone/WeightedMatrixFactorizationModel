# WeightedMatrixFactorization Model for Recommender Systems

## About data
The MovieLens dataset is a widely used benchmark dataset for recommendation systems research and evaluation. It contains user-item ratings collected from the MovieLens website, where users rate movies on a scale of 1 to 5.

For this project, it has been decided to use the MovieLens 100k: the smallest version of the dataset containing 100000 ratings. This version is commonly used for initial experimentation and prototyping due to its smaller size and faster processing capabilities.

## Matrix Factorization Model
Let's consider a scenario with m users and items. Our objective with the recommendation system is to construct an m×n matrix, commonly known as the feedback matrix. This matrix represents ratings for each user-item pair. Note that, due to the limited number of available ratings, this matrix is very sparse.

Here's an example of a feedback matrix for a recommendation system with 3 users and 4 items (movies), where missing ratings are represented by '?' symbols:

|         | Item 1 | Item 2 | Item 3 | Item 4 |
|---------|--------|--------|--------|--------|
| User 1  |   5    |   ?    |   4    |   3    |
| User 2  |   ?    |   2    |   ?    |   ?    |
| User 3  |   4    |   3    |   ?    |   5    |



The goal of the Recommender system is to predict the missing ratings for each user-item pair marked with '?'.

By thinking of the feedback matrix as the product of two low-rank matrices, each with k latent factors, we simplify the essence of this problem. This simplification helps us frame the problem as a minimization task, where our goal is to optimize a function.

Our objective function is defined as follows:

```math
\min_{U,V}{\sum_{(i,j) \in obs}{(A_{ij}-U_i\dot{V_j^{T}})^2+\lambda(||U_i||^2+||V_j||^2)}}
```

In this equation:
- *obs* represents the set of *(i, j)* pairs for which ratings are known (i.e., the observed ratings).
- *Aij* is the rating assigned to item j by user i.
- *Ui* and *Vj* denote the user and item latent factor matrices, respectively.
- *λ* serves as a regularization parameter to mitigate overfitting.

To tackle this minimization problem, two common optimization algorithms can be used:


- *Stochastic Gradient Descent* (*SGD*): This method iteratively adjusts the parameters (user and item vectors) by moving in the direction of the negative gradient of the objective function.
- *Weighted Alternating Least Squares* (*WALS*): An optimization algorithm that alternates between updating the user and item matrices while keeping the other fixed. This method can also be parallelized as updates for different users/items can be performed independently.

Both of these optimization techniques are implemented in this project.

## Building a recommender system
Once the weighted matrix factorization model has learned thw two matrices *U* and *V* (the users embedding and the items embdedding), we can build our recommendation system. 


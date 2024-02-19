# Recommender System

## About data
The MovieLens dataset is a widely used benchmark dataset for recommendation systems research and evaluation. It contains user-item ratings collected from the MovieLens website, where users rate movies on a scale of 1 to 5.

For this project, it has been decided to use the MovieLens 100k: the smallest version of the dataset containing 100000 ratings. This version is commonly used for initial experimentation and prototyping due to its smaller size and faster processing capabilities.

## Recommender systems
Recommender systems can be broadly categorized into two types:
- **Content-Based**: these systems aim to recommend items to users by analyzing the content of the items and users' profiles. They match users with items based on similarities in item content and user preferences.
- **Collaborative filtering**: this approach relies on the principle that users who have similar preferences in the past are likely to have similar preferences in the future. Collaborative filtering algorithms recommend items to users based on the preferences of similar users. 

## Matrix factorization
Let's consider a scenario with m users and items. Our objective with the recommendation system is to construct an m×n matrix, commonly known as the feedback matrix. This matrix represents ratings for each user-item pair. Note that, due to the limited number of available ratings, this matrix is very sparse.

Here's an example of a feedback matrix for a recommendation system with 3 users and 4 items (movies), where missing ratings are represented by '?' symbols:

|         | Item 1 | Item 2 | Item 3 | Item 4 |
|---------|--------|--------|--------|--------|
| User 1  |   5    |   ?    |   4    |   3    |
| User 2  |   ?    |   2    |   ?    |   ?    |
| User 3  |   4    |   3    |   ?    |   5    |



The goal of the Recommender system is to predict the missing ratings for each user-item pair marked with '?'.

This problem can be seen as a minimisation task, where our **loss function** is defined as:
$$
\min_{p,q}{\sum_{(i,j) \in obs}{(A_{ij}-U_i\dot{V_j^{T}})^2+\lambda(||U_i||^2+||V_j||^2)}}
$$
where $obs$ is the set of $(i,j)$ pairs which are known (the observed ratings). In order to avoid overfitting we use the $\lambda$ for regularize the parameters.

```math
\min_{p,q}{\sum_{(i,j) \in obs}{(A_{ij}-U_i\dot{V_j^{T}})^2+\lambda(||U_i||^2+||V_j||^2)}}
```


To solve this minimization task, two common optimization algorithms can be used:
- Stochastic Gradient Descent (SGD): A popular optimization algorithm that iteratively updates the parameters (user and item vectors) in the direction of the negative gradient of the objective function.
- Weighted Alternating Least Squares (WALS): An optimization algorithm that alternates between updating the user and item matrices while keeping the other fixed. This method can also be parallelized as updates for different users/items can be performed independently.

In this project are implemented both of them, 

## References
- dw

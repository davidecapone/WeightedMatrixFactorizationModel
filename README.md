# Netlix Recommender System
*This repository contains the final project for the course of Information Retrival, University of Trieste (master degree in Data Science & Scientific Computing).* 

## About the dataset
The dataset provided here has been utilized in the Netflix Prize, an open competition organized by Netflix to identify the best algorithm for predicting user ratings on films. The grand prize of $1,000,000 was awarded to the BellKor's Pragmatic Chaos team, which emerged as the winner.

## Recommender systems

There are two types of Recommender systems:
- **Content-Based**: these systems try to match users with items based on items content and users profiles.
- **Collaborative filtering**: they rely on the assumption that similar users like similar items. Similarity measures between users and/or items are used to make recommendations.

## Matrix factorization
Let's assume to have m users and n items. The goal of our recommendation system is to build a $m \times n$ matrix (also called feedback matrix) in which we have a rating for each user-item pair. Since we only have a limited number of ratings, this matrix is very sparse.

|       |movie1|movie2|movie3|movie4|
|-------|:------:|:------:|:------:|:------:|
| user1 |1|?|3|?|
| user2 |1|4|?|?|
| user3 |?|?|3|1|

This is an example of feedback matrix in which we have 3 users and 4 items (movies). The couples user-item with no rating are labelled with '?'.
The task for the algorithm is to infer those values actually.

$$
\min_{p,q}{\sum_{(i,j) \in obs}{(A_{ij}-U_i\dot{V_j^{T}})^2+\lambda(||U_i||^2+||V_j||^2)}}
$$
where $obs$ is the set of $(i,j)$ pairs which are known (the observed ratings). In order to avoid overfitting we use the $\lambda$ for regularize the parameters.

For solving this minimization task we can use:
- Stochastic Gradient Descent (SGD)
- Weighted Alternated Least Square (WALS). 


## References
- Download the dataset: Netflix Movie Ratings ([Kaggle](https://www.kaggle.com/datasets/evanschreiner/netflix-movie-ratings?resource=download)).

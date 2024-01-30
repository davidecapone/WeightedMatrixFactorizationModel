# IRProject
This is the repository for the Information Retrieval's project (University of Trieste, Data Science & Scientif Computing Master Degree)

## Recommender systems

There are two types of Recommender systems:
- Content-Based systems: these systems try to match users with items based on items content and users profiles.
- Collaborative filtering: they rely on the assumption that similar users like similar items. Similarity measures between users and/or items are used to make recommendations.

## Matrix factorization
Let's assume to have m users and n items. The goal of our recommendation system is to build a mxn matrix (also called feedback matrix) in which we have a rating for each user-item pair. Since we only have a limited number of ratings, this matrix is very sparse.

|       |movie1|movie2|movie3|movie4|
|-------|------|------|------|------|
| user1 |1|?|3|?|
| user2 |1|4|?|?|
| user3 |?|?|3|1|

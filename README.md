#                                                 MOVIE RECOMMENDATION

![image](https://github.com/ThomasSabatier/An-Awesome-Movie-Recommender/streamlit_app/assets/godzilla_criterion.png?raw=true)

## Context

This repository contains the code for our project **MOVIE RECOMMENDATION SYSTEM**, developed during our [Data Scientist training](https://datascientest.com/en/data-scientist-course) at [DataScientest](https://datascientest.com/).
We have chosen to develop a movie recommendation system based on data sets from the reference database for the movie world (among others): IMDb, as well as from a movie rating site: Movie Lens, which provides additional information to IMDb data

## Objective
The goal of this project is to **build a system that will recommend to a specific users the movies he might like** based on the user history, personal preferences and similarities with other users.

## Datasets

Our initial work uses datasets made available by the educational community, namely
IMDb datasets : https://www.imdb.com/interfaces/
Movie Lens datasets: https://grouplens.org/datasets/movielens/20m/

According to our research and current collaborative filtering and content-based filtering methods, these datasets are sufficient for the design of a recommendation algorithm.
IMDb datasets provide a lot of information about the characteristics of movies. The Movie Lens datasets complement it by providing information about the users' behavior towards movies (rating, tags). The datasets complement each other to cover the two filtering approaches mentioned above.

![Dataset](https://user-images.githubusercontent.com/125690999/228502757-21c615e7-993a-482a-be9f-fc0aa66de28b.png)


## Project

We have worked on 3 approaches:
- The first approach, called content-based filtering, aims to suggest to a user one or more products with similarities to the products he likes. This approach focuses on the characteristics of the products to match them. In the case of a movie, these characteristics are, for example, the genre, the cast and the duration.

- The second approach, called collaborative filtering, focuses on the similarities in behavior between two users. It consists in determining similar habits between two users and suggesting to each one the movies (in our case) that the other one likes. This approach does not require any information about the movies.
This version of our recommendation system implemented a nearest neighbor model (the NearestNeighbors function of the sklearn library) with the cosine similarity calculation as a metric. The idea was to create vectors with the ratings given by each user to each movie, then to compute the cosine of the angles between each vector in order to determine, for a user U, which are the other closest users, i.e. with the closest rating schemes. Then, we extracted the average ratings of the movies that user U had never rated (so he has never seen, by simplification with respect to the data we have) and we recommended him the movies with the best averages.

![image](https://user-images.githubusercontent.com/125690999/226980185-2ff99c5e-4f07-4d90-a985-4e3c1ae944c8.png)


-  The third approach tested aims at optimizing the results with a hybrid recommender system combining a matrix factorization (to overcome the missing score problem) with a K-nearest neighbor search.
Matrix factorization by SVD allows decomposing a matrix into three simpler matrices: a left orthogonal matrix (U), a diagonal matrix containing the singular values (S) and a right orthogonal matrix (V). The number of significant singular values included in the S matrix represent the hidden factors in the original data and thus the latent dimension.
By doing the vector product of the three matrices obtained, we obtain a new matrix approximating the original matrix. In our case we obtain an approximation of the scores.                        

## Choise of Datasets
The implementation of each approach has required different processing of the datasets.



## Code

You can browse and run the [notebooks](./notebooks). You will need to install the dependencies (in a dedicated environment) :

```
pip install -r requirements.txt
```


## Streamlit App

**Add explanations on how to use the app.**

To run the app :

```shell
cd streamlit_app
conda create --name my-awesome-streamlit python=3.9
conda activate my-awesome-streamlit
pip install -r requirements.txt
streamlit run app.py
```

The app should then be available at [localhost:8501](http://localhost:8501).

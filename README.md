# An Awesome Movie Recommender

![image_name](./streamlit_app/assets/godzilla_criterion.png?raw=true)

## Context

This repository contains the code of a project initiated during my [Data Scientist training](https://datascientest.com/en/data-scientist-course) at [DataScientest](https://datascientest.com/).
It was originally developed with my former teammates : Michèle DUBUISSON and Antoine SIBILLE. I know ensure updates of the code on this GitHub.

## Objective

The goal of this project is to **build an (awesome) hybrid recommender system* that will embed collaborative filtering and content based-filtering.

## Datasets

- IMDb datasets : https://www.imdb.com/interfaces/
- Movie Lens datasets: https://grouplens.org/datasets/movielens/20m/

## Project

We have worked on 3 approaches:

- Collaborative filtering focuses on the similarities in behavior between two users. It consists in determining similar habits between several users and suggesting to each one the movies (in our case) that the other one likes. The model is based on matrix factorization by Singular Value Decomposition.

- Content-based filtering aims to suggest to a user one or more products with similarities to the products he likes. This approach focuses on the characteristics of the products to match them. In the case of a movie, these characteristics are, for example, the genre, the cast and the duration. The model is based on cosine similarity score between features vectors.

![image](https://user-images.githubusercontent.com/125690999/226980185-2ff99c5e-4f07-4d90-a985-4e3c1ae944c8.png)

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

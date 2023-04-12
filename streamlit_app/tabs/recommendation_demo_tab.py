
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# In[]:

### LIBRAIRIES ###

# Standard libraries
import sys
import os

# Third-party libraries
import streamlit as st
import numpy as np
from PIL import Image
from tmdbv3api import TMDb, Movie
import requests
from io import BytesIO

# Custom libraries
sys.path.append("..")
from main_recommandation_films import hey_honey_what_do_we_watch_tonight, user_lists
from main_recommandation_films import get_similar_matrix, content_recommendations
from main_recommandation_films import df_movie_lens, df_ml_relevant_tags


# In[]:

### VARIABLES ###

tab_title = "Collaborative filtering"
sidebar_name = "Demo"

tmdb = TMDb()
tmdb.api_key = '8ddc278f96b0a1a3fe8d231b7bc78012'
movie_api = Movie()

# In[]:
    
current_file_path = os.path.realpath(__file__) 
current_dir = os.path.dirname(current_file_path)

pkl_dir = os.path.join(current_dir, '..', '..', 'data', 'pkl')

best_params_file = os.path.join(pkl_dir, "best_params.pkl")
sim_movie_file = os.path.join(pkl_dir, "similar_movie.pkl")

# In[]:
    
### FONCTIONS ###

def display_movie_poster(tmdb_id):
    try:
        movie = movie_api.details(tmdb_id)
        poster_path = movie.poster_path
        poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}"
        response = requests.get(poster_url)
        poster = Image.open(BytesIO(response.content))
        st.image(poster, width=200)
    except Exception as e:
        st.write(f"Erreur lors de la récupération du poster: {e}")

def display_posters(df, title):
    with st.expander(title):
        #top_n = len(df)
        col_count = 3
        current_col = 0
        cols = st.columns(col_count)

        for index, movie in df.iterrows():
            tmdb_id = int(movie['tmdbId'])
            with cols[current_col]:
                display_movie_poster(tmdb_id)
                #st.write(movie['title'])
                #if st.button('Like', key=tmdb_id):
                 #   st.write("tmdb ID :", tmdb_id)
                current_col = (current_col + 1) % col_count


                
        
# In[]:

### RUN ###
def run():  
        
    ### USER SELECTION ###
    st.title(tab_title)
    
    
    list_of_users = user_lists(df_movie_lens)
    user_id = st.select_slider(
        "Choix de l'utilisateur",
        options = list_of_users)
    
    top_n = st.select_slider(
        "Nombre de films à afficher",
        options = np.arange(3,30,3))
    
    df_top_n_recommendations, df_rated_movies = hey_honey_what_do_we_watch_tonight(user_id, top_n, df_movie_lens)
    #st.write(df_rated_movies)
    #st.write(df_top_n_recommendations)
    similar_movie = get_similar_matrix(df_ml_relevant_tags['tag'].values, sim_movie_file)
    selected_movie_id = df_rated_movies.loc[0, 'tmdbId']
    content_based_recommendations = content_recommendations(selected_movie_id, similar_movie, df_ml_relevant_tags, top_n)
    #st.write("\n", content_based_recommendations)
    
    # Affichage des posters
    display_posters(df_rated_movies, "Films notés")
    display_posters(df_top_n_recommendations, "Films recommandés")
    display_posters(content_based_recommendations, "Films que vous aimerez aussi")
         
 
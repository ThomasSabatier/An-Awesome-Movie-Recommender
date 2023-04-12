import streamlit as st
from PIL import Image
import os

title = "AN AWESOME MOVIE RECOMMENDER"
sidebar_name = "Introduction"

current_file_path = os.path.realpath(__file__) 
current_dir = os.path.dirname(current_file_path)
assets_dir = os.path.join(current_dir, '..',  'assets')
cover_image = os.path.join(assets_dir, "godzilla_criterion.png")

def run():
    
    
    st.markdown(" ")
    st.title(title)
    st.markdown("---")
    #st.image(cover_image)
    st.image(Image.open("assets/godzilla_criterion.png"))
    st.markdown(" ")
    st.markdown(" ")
    st.markdown(
        """
        ### Context
        \n
        This repository contains the code for our project **MOVIE RECOMMENDATION SYSTEM**,
        which was developed during our [Data Scientist training](https://datascientest.com/en/data-scientist-course) 
        at [DataScientest](https://datascientest.com/). We chose to develop a movie recommendation system based on datasets 
        from various sources, including IMDb (a reference database for the movie world) 
        and Movie Lens (a movie rating site that provides additional information on user ratings).

        Here are the links to the datasets we used:
        - [Movie Lens](https://movielens.org/)
        - [IMDB](https://www.imdb.com/)
        
        ### Objective
        The goal of this project is to build a system that can recommend movies to specific users based on their history, 
        personal preferences, and similarities with other users. We explored two possible approaches to system recommender: 
        collaborative and content-based. The objective of our project is to implement these different approaches 
        and combine them into an hybrid model.
        \n
        """
    )
    st.markdown(" ")
    st.markdown(" ")
    st.markdown(" ")

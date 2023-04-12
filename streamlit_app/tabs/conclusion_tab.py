import streamlit as st
import os
from PIL import Image


title = "Conclusion & Prospects"
sidebar_name = "Conclusion"

current_file_path = os.path.realpath(__file__) 
current_dir = os.path.dirname(current_file_path)
assets_dir = os.path.join(current_dir, '..',  'assets')
cover_image = os.path.join(assets_dir, "Thats_all_folks.svg.png")

def run():
    st.markdown(
        """ /n
        """
    )
    st.title(title)
    st.markdown("---")

    st.markdown(
        """
        ###  Relevance of the approach and of the model.
        The literature on recommender systems is dense and the approaches we have tested have proven themselves in many industries.
        For instance, the Netflix algorithm, is based – among other things – on matrix factorization since a certain Simon Funk 
        had won third place in a competition aimed at improving the performance of their existing models, 
        by proposing a decomposition into singular values.
        
        ### When tuning goes wrong. When diversity goes good.
        This is a delicate topic that we have encountered with our model and read in the litterature.
        If the tuning SVD model gives better performance (RMSE, MAE) than a not tuned one, the recommendation appears to be
        sometimes off the profile of our user. The recommended movies are surprisingly "cinema d'auteur" oriented.
        So as cinema amateur we certainly approve this as diversity might be one of the weaknesses of recommender system,
        but low familiarity is also bad. In conclusion: output remains sensitive to hyperparameters
        
        ### Suggested improvements
        Now that we have successfully implemented a collaborative recommendation model based on the Surprise library, 
        and our data manipulation methods have been validated, we are well-equipped to integrate several other models into our Pipeline. 
        This will allow us to compare the performance of different algorithms, such as NMF, KNN, and others.
        Given more time, we would have explored additional techniques to address the cold start problem, 
        such as incorporating content-based and collaborative approaches. Furthermore, we could have experimented with deep learning methods, 
        such as time-series analysis, to capture the evolving nature of user ratings over time. We have access to the timestamp variable, 
        which can be used to address time-series analysis challenges.
        
        """
    )

    #st.image(cover_image)
    st.image(Image.open("assets/Thats_all_folks.svg.png"))
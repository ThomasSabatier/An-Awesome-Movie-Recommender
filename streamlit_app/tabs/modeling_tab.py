import streamlit as st
from PIL import Image

title = "Modeling"
sidebar_name = "Modeling"


def run():
    st.markdown(
        """ /n
        """
    )
    st.title(title)
    st.markdown("---")

    st.markdown(
        """
        ### Collaborative filtering
        
        The basic idea of collaborative filtering using Singular Value Decomposition (SVD) for making personalized recommendations 
        is to factorize the original user-item rating matrix into three smaller matrices using matrix decomposition. 
        These matrices capture the latent factors underlying user-item interactions, which can then be used to predict user ratings for unseen items.
        """)
        
    st.image(Image.open("assets/facto.png"))
    st.markdown(
        """
        According to the above schema, the decomposition of the original pivot table A containing ratings 
        of every movies by every single users gives:
        - a left orthogonal matrix (U)
        - a diagonal matrix containing singular values (D)
        - a right orthogonal matrix (V)
        
        By calculating the vector product of these three matrices, we can obtain a new approximated matrix along with its corresponding ratings. 
        ##### M = U/D/V^T
        
        #### ? Surprise Library ?
        After testing various techniques and libraries, we decided to use the Surprise library.
        It is a Python library specifically designed for building and evaluating recommender systems.
        The library provides a broad range of collaborative filtering algorithms, similarity metrics, and evaluation metrics.
        Additionally, it is built on top of NumPy and SciPy, and comes with built-in cross-validation and GridSearchCV functionality.
     
        #### Parameters tuning
        To optimize the hyperparameters of the SVD model, we employed GridSearchCV with cross-validation. 
        GridSearchCV allows us to test a range of hyperparameters, such as the number of latent factors, 
        learning rate, and regularization parameter, and select the optimal combination that yields the best performance.
        In our case, we have tuned the SVD algorithm with a 5 folds cross-validation, looking for the best params:
        - n_factors
        - n_epochs
        - lr_all
        - reg_all
        
        We selected RMSE to minimize the distance between the original ratings and the predicted ratings.
        """)
    st.image(Image.open("assets/RMSE.png"))
    st.markdown(
        """
        RMSE or MSE gives more weight to bigger errors so it's a good indicator on how stable is our prediction model.
        ##### Average RMSE score: 0.861
    
        """
    )
    
    
    
    
    st.markdown(
        """
        ### Content based filtering
        For the content based filtering, we use text mining technique and then apply cosinus similarity. 
        To apply cosine simularity, we have to create vectors with features from each movies and then calculate 
        cosinus from each vector angles to elect the closest movies with closest features.
        Then we removed movies that the user has already seen and recommend the ones with the best similarity scores.
        """
        )
    st.image(Image.open("assets/cosine_1.png"))
    st.image(Image.open("assets/cosine_2.jpg"))
    st.markdown(
        """
        So we combined all movie types from IMBD and MovieLens, genome tags from MovieLens with relevance > 0.8, IMDB average ratings and number of votes.
        Then we used the Term Frequency-Inverse Document Frequency or TF-IDF to determine the relevance of words according to some rules:
            first count the frequency or a word (or group of up to 3 words in our case)
            Then do inverse search to determine its specificity.
        """
        )
    st.image(Image.open("assets/td-idf.png"))
    st.image(Image.open("assets/td-idf 2.png"))
    st.markdown(
        """
        Last, the output of TF-IDF is entered to calculate the cosine similarity of each movie, output is a big matrix with a score between each movie.
        However, as there are occasionnaly similar similarity scores, we use average ratings and number of votes to distinguish them.
        This recommendation model is particularly relevant for cold start i.e when a new user is coming on board and no rating is available.\n
        \n
        """
    )

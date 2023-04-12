
import streamlit as st

from PIL import Image

title = "DATASET"
sidebar_name = "Dataset"

def run():

    st.markdown(
            """ /n
            """
        )

    st.title(title)

    st.markdown("---")

    st.markdown(
        """
        ### An unsupervised classification problem
        When it comes to unsupervised classification problems, there is no target variable per se. 
        Instead, we want to determine similarities between users or between movies. 
        This allows us to recommend movies to a specific user based on the ones liked or watched by other similar users.
        - This is the  _collaborative filtering!_ :star::star::star::star::star:
          
        \n 
        And if a user likes a particular movie, they will be recommended other movies that share similarities. 
        This is done by calculating a similarity score between the features of the liked movie and those of other movies in the dataset. 
        The system then recommends the movies with the highest similarity scores to the user.
        - This is _the content-based filtering!_ :thumbsup:
        
        ### Choice of dataset
       Now that we know the kind of approaches we want to implement, we can think about what data we need. Basically, we want to find similarity between users, and a simple way to do that would be to compare how they rate movies with others. Additionally, to determine if two movies are similar, we need to compare how many common features they share. We were able to collect this data using the following official datasets:
       - [IMDB](https://www.imdb.com/) - the world's most popular movie reference database.
       - [Movie Lens](https://movielens.org/) - run by GroupLens, a research lab at the University of Minnesota.
       Once these datasets are combined, they provide relevant data on users' ratings and movie features.
        
        ### Data cleaning and processing
        In the latest version of the project, we utilized the "small Movie Lens" dataset. 
        This dataset is light enough to be hosted on Github and includes an already filtered user and movie listing. 
        As a result, every user in the dataset has rated at least 20 movies, which is a decent threshold for collaborative filtering. 
        For content filtering, the most critical step was to filter the tags used as a feature 
        along with genres to select only the most relevant ones out of the 1000+ available.

        Linking the IMDB and Movie Lens datasets is possible using an additional table that links the IMDB and Movie Lens movies' IDs.
        
        ##### 'tt0' + imbdId = tconst
        """
    )
    
    # st.image("https://user-images.githubusercontent.com/125690999/228502757-21c615e7-993a-482a-be9f-fc0aa66de28b.png")
    st.image(Image.open("assets/Dataset.png"))
    
    st.markdown(
        """
        ### Pipeline for collaborative approach\n
    
        Matrix factorization, specifically Singular Value Decomposition (SVD), 
        is a technique used in recommendation systems to predict user preferences for items they have not yet seen or rated. 
        The technique involves decomposing the sparse evaluation matrix into two matrices of latent factors - a user factor matrix and an item factor matrix. 
        The latent factors capture the underlying relationships between users and items and are used to predict missing ratings. 
        SVD is useful in recommending products or items to users based on their past preferences and can be used to solve the problem of data sparsity.
        """
   )
   
    st.image(Image.open("assets/dataset_collaborative.png"))
    st.markdown(
        """
        ### Pipeline for content approach\n
    
        The content-based filtering approach aims to suggest to a user one or more products with similarities to the products he likes. 
        This approach is therefore interested in the characteristics of the products in order to relate them. 
        In the context of a film, these characteristics are, for example, the genre, the cast and the duration.\n
        We used this approach by restricting ourselves to genres and tags as well as the relevance of these tags.\n
        """
   )
    st.image(Image.open("assets/dataset-content.png"))    
    st.markdown(
        """
        ### \n
        
        """
   )        

import streamlit as st
from PIL import Image


title = "Datavisualization"
sidebar_name = "Datavisualization"


def run():
    st.markdown(
        """ /n
        """
    )
    st.title(title)
    st.markdown("---")

    st.markdown(
        """
        ### Movie Lens Rating :
        We analyzed the rating distributions of both datasets. 
        The MovieLens user ratings range from 0.5 to 5 in 0.5 increments, where half points are used less frequently than whole numbers.
        
        """
    )
    st.image(Image.open("assets/ratingMovieLens.png"))


    st.markdown(
        """
        ### IMDb average rating
        Movie averages given by IMDb are between 0 and 10 with an accuracy of one-tenth.
        The distribution of grades differs according to the type of rating.
        
        """
    )
    st.image(Image.open("assets/ratingIMDB.png"))

    st.markdown(
        """
        ### Ratings by users
        We also see that a small number of users voted for many films. 75% of users voted for a maximum of 155 films.
        
        """
    )

    st.image(Image.open("assets/RatingByUser.png"))


    st.markdown(
        """
        ### Ratings by movies
        We note that a small number of films have a lot of ratings compared to all the films, this could create an imbalance in the relevance and consideration of these votes.
        """
    )

    st.image(Image.open("assets/RatingByMovie.png"))

    st.markdown(
        """
        ### Genres
        The gender distributions for each of the datasets differ slightly. 
        Additionally, the number of genres is greater for the IMDb Dataset. 
        While the genre categories are not identical in the two databases, the two primary genres (comedy and drama) remain the same. 
        As each film can be associated with multiple genres, 
        it will be necessary to merge the genre lists for each film to simplify the analysis.
        
        """
    )
    st.image(Image.open("assets/Distribution_genres.jpg"))
    st.markdown(
        """
        ### Tags
        We observe that out of 1128 tags, only a small number are particularly relevant. 
        Therefore, grouping these tags could be useful in our model, as they hold significant importance regardless of the film.
        """
    )
    st.image(Image.open("assets/Relevance.png"))
    
   


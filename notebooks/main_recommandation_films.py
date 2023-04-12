#!/usr/bin/env python
# coding: utf-8

# <br>
# 
# # Movie recommender system : collaborative-content filtering
# 
# <br>
# 
# Hi everyone! Welcome to this Jupyter notebook on building a movie recommendation system! 
# 
# In today's world of seemingly endless movie choices, it can be difficult to decide what to watch next. This is where recommendation systems come in. By analyzing patterns in movie ratings, these systems can suggest movies that a user is likely to enjoy. In this notebook, we will explore the key concepts involved in building such a system. 
# 
# First we start with collaborative filtering and an hybrid nearest neighbors - matrix factorization approach. Collaborative filtering based on nearest neighbors allows us to identify similar users and recommend movies based on their past preferences. By comparing the movie ratings of different users, we can identify patterns and similarities, and use these to make recommendations for movies that a user hasn't yet seen. However, this approach has limitations like when there are sparse data. In these cases, matrix factorization can be used to fill in missing ratings in the matrix and make better predictions. 
# By combining collaborative filtering and matrix factorization, we can create a more robust and effective movie recommendation system that leverages the strengths of both techniques.
# 
# Finally, we'll introduce content-based filtering, which is used to address some of the limitations of collaborative filtering. We'll discuss each of these concepts in detail and demonstrate how they can be combined to create a personalized and effective movie recommendation system.
# 
# ##### The (super) team that built the system is composed of the following (super) members:
# * Michèle Dubuisson
# * Thomas Sabatier
# * Antoine Sibille
# 
# ##### The datasets files were taken from the following sources:
# * Movie Lens (small) dataset : https://grouplens.org/datasets/movielens/
# * IMDb dataset : https://www.imdb.com/interfaces/
# 
# <br>

# ### Librairies
# 

# In[2]:


# Regular librairies
import numpy as np
import pandas as pd
import random
import statistics
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

# Machine learning
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import GridSearchCV
from surprise.model_selection import train_test_split


# ## Collaborative filtering
# 

# ### 1. Pre-processing
Collaborative filtering requires finding patterns between users and items. 
Here, the pattern is based on similarities in movie ratings among users.
# In[3]:


# Load datasets into a DataFrame
df_ml_movies = pd.read_csv('../data/ml-latest-small/movies.csv')
df_ml_ratings = pd.read_csv('../data/ml-latest-small/ratings.csv')
 
# Split movie title and movie release year
df_ml_movies['year'] = df_ml_movies['title'].str.extract('(\d{4})', expand=False)
df_ml_movies['title'] = df_ml_movies['title'].str.replace('(\(\d{4}\))', '', regex=True).str.strip()

# Select the usefull features
df_ml_movies = df_ml_movies[['movieId', 'title', 'year', 'genres']]
df_ml_ratings = df_ml_ratings.merge(df_ml_movies, how='inner', on='movieId')
df_ml_ratings = df_ml_ratings[['userId', 'movieId', 'rating', 'title']]


# ### 2. Data exploration

# In[4]:


# Distribution of ratings by users
nb_ratings_by_user = df_ml_ratings['userId'].value_counts()

counts, bins = np.histogram(nb_ratings_by_user, bins=200) # Compute the histogram
cmap = plt.cm.Spectral
norm = plt.Normalize(counts.min(), counts.max()) # Normalize bar height

f,ax = plt.subplots(1,1,figsize=(10,4))
for count, x in zip(counts, bins[:-1]):
    plt.bar(x, count, width=bins[1]-bins[0], color=cmap(norm(count)))
plt.xlim(0, 400)
plt.ylim(0, 150)
plt.title('Distribution of ratings by users')
plt.xlabel('Number of ratings')
plt.ylabel('Number of users')
plt.show()

print(nb_ratings_by_user.describe())


# In[5]:


# Distribution of ratings by movies
nb_ratings_by_movie = df_ml_ratings['movieId'].value_counts()

counts, bins = np.histogram(nb_ratings_by_movie, bins=100) # Compute the histogram
cmap = plt.cm.Spectral
norm = plt.Normalize(counts.min(), counts.max()) # Normalize bar height

f,ax = plt.subplots(1,1,figsize=(10,4))
for count, x in zip(counts, bins[:-1]):
    plt.bar(x, count, width=bins[1]-bins[0], color=cmap(norm(count)))
plt.xlim(-2, 80)
plt.ylim(0, 1500)
plt.title('Distribution of ratings by movies')
plt.xlabel('Number of ratings')
plt.ylabel('Number of movies')
plt.show()

print(nb_ratings_by_movie.describe())


# ### 3. Model building

# #### Enter the matrix
We will now apply some filters to the data based on statistical properties. Specifically, for the user data, we will use a low threshold of the minimum number of rated movies, and a high threshold of the third quartile number of rated movies.
For the movie data, we will only set a low threshold based on the median number of ratings, as we have high sparsity in the data.
# In[6]:


# Convert the count Series into a DataFrame and filter the rows
df_nb_ratings_by_user = nb_ratings_by_user.to_frame(name='nb_of_ratings')
df_nb_ratings_by_user = df_nb_ratings_by_user.query('nb_of_ratings >= 35 and nb_of_ratings <= 170')
# Apply to the original matrix 
df_ml_ratings_filtered = df_ml_ratings[df_ml_ratings['userId'].isin(df_nb_ratings_by_user.index)]

# Convert the count Series into a DataFrame and filter the rows
df_nb_ratings_by_movie = nb_ratings_by_movie.to_frame(name='nb_of_ratings')
df_nb_ratings_by_movie = df_nb_ratings_by_movie.query('nb_of_ratings >=3')
# Apply to the original matrix 
df_ml_ratings_filtered = df_ml_ratings_filtered[df_ml_ratings_filtered['movieId'].isin(df_nb_ratings_by_movie.index)]

#matrix = df_ml_ratings_filtered.pivot_table(index = 'userId', columns ='movieId', values = 'rating').fillna(0)
matrix = df_ml_ratings.pivot_table(index = 'userId', columns ='movieId', values = 'rating').fillna(0)
matrix.head()


# #### Méthode 1 : Matrix Factorization (Truncated SVD)

# In[7]:


def matrix_factorization(matrix, n_components):
    
    sparse_matrix = csr_matrix(matrix)

   
    svd = TruncatedSVD(n_components=n_components)
    U_Sigma = svd.fit_transform(sparse_matrix)

 
    predicted_data = U_Sigma.dot(svd.components_)
    predicted_data = np.clip(predicted_data, 0.5, 5)  
    
   
    predicted_data = pd.DataFrame(predicted_data, index=matrix.index, columns=matrix.columns)

    return predicted_data


# In[8]:


def what_do_we_watch_tonight(user_id, matrix, n_neighbors, latent_dimension, n_movies=10, enable_neighbors=True):
    
    if enable_neighbors == True:
    # Create a matrix based on the user nearest neighbors
        matrix_knn = nearest_neighbors_matrix(user_id=user_id, matrix=matrix, n_neighbors=n_neighbors)
    else:
        matrix_knn = matrix
        
    # Matrix factorization
    df_predictions = matrix_factorization(matrix_knn, latent_dimension)

    # Sort the predictions for the user by rating descending order
    sorted_user_predictions = df_predictions.loc[user_id].sort_values(ascending=False).to_frame(name="prediction")

    # Get the original rating of the user
    user_data = matrix_knn.loc[user_id].to_frame(name="rating")

    # Merge true ratings and predicted ratings
    user_full = pd.merge(left = user_data, right = sorted_user_predictions, how='inner', on='movieId')
    user_full = pd.merge(left = user_full, right = df_ml_movies, how='inner', on='movieId')

    # Filter on movies that the user hasn't rated yet
    user_full = user_full.loc[user_full['rating'] == 0]

    # Recommendations
    recommendations = user_full.sort_values(by='prediction', ascending=False).head(n_movies)
    recommendations = recommendations[['title', 'year', 'prediction']]
    recommendations = recommendations.reset_index(drop=True)
    
    return recommendations


# #### Méthode 2 : Matrix Factorization (Surprise SVD)

# In[73]:


# Convert data into Surprise Dataset
reader = Reader(rating_scale=(0.5, 5))
data = Dataset.load_from_df(df_ml_ratings[['userId','movieId','rating']], reader)


# In[11]:


# Hyper-parameters tuning with cross-validation
param_grid = {'n_factors': [50, 100, 150],
              'n_epochs': [10, 20, 30], 
              'lr_all': [0.002, 0.005, 0.01],
              'reg_all':[0.05, 0.1, 0.2]}

gs = GridSearchCV(SVD, param_grid, measures=["rmse", "mae"], cv=3)

gs.fit(data)


# In[12]:


# Tuned SVD with best params from GridSearchCV
best_params = gs.best_params["rmse"]

algo = SVD(n_epochs=best_params['n_epochs'], 
           lr_all=best_params['lr_all'], 
           reg_all=best_params['reg_all'], 
           n_factors=best_params['n_factors'])

# Create and fit a trainset out of the full data dataset
trainset = data.build_full_trainset()
algo.fit(trainset)

best_params # Display best parameters


# In[85]:


def hey_honey_what_do_we_watch_tonight(user_id, top_n, trainset, algo): 
    
    # List of movies rated by a specific user
    rated_items = set(item_id for (item_id, _) in trainset.ur[random_user])
    
    # List of movies not rated yet by a specific user
    not_watched_yet = [item_id for item_id in trainset.all_items() if item_id not in rated_items]

    movie_titles = dict(zip(df_ml_ratings['movieId'], df_ml_ratings['title']))
    
    rated_movies = [(item_id, rating) for (item_id, rating) in trainset.ur[user_id]]
    sorted_rated_movies = sorted(rated_movies, key=lambda x: x[1], reverse=True)
    
     # Create rated movies string
    rated_movies_str = f"Top {top_n} des films déjà notés par l'utilisateur {user_id}:\n"
    rated_movies_str += "---------------------------------------\n"
    for item_id, rating in sorted_rated_movies[:top_n]:
        movie_id = trainset.to_raw_iid(item_id)  # Convert internal item_id to external movie_id
        rated_movies_str += f"{movie_id} | {movie_titles[movie_id]} | rating: {rating:.2f}\n"
    
    # Predictions
    predictions = [algo.predict(random_user, item_id) for item_id in not_watched_yet]
    sorted_predictions = sorted(predictions, key=lambda x: x.est, reverse=True)

    # Display top n recommendations
    recommendations_str = ""
    recommendations_str = f"Top {top_n} recommandations pour l'utilisateur {user_id}:\n"
    recommendations_str += "---------------------------------------\n"
    for pred in sorted_predictions[:top_n]:
        recommendations_str += f"{pred.iid} | {movie_titles[pred.iid]} | rating: {pred.est:.2f}\n"

    return recommendations_str, rated_movies_str


# In[96]:


# Randomly choose an user
user_ids = trainset.all_users()
random_user = random.choice(user_ids)

# Make top n recommendations and compare with the top n movies the user has rated
reco, rated = hey_honey_what_do_we_watch_tonight(random_user, 10, trainset, algo)
print(reco)
print(rated)

# Let's compare with predictions of the first method
#display(what_do_we_watch_tonight(random_user, matrix, n_neighbors=0, latent_dimension=100, n_movies=10, enable_neighbors=False))


# ## Content based-filtering

# ### Pre-processing

# In[192]:


# file data loads IMDB and ML combined
df_movies = pd.read_csv('../data/IMDb/df_movies.csv')

# taking care of NaN and data types
df_movies['year']=df_movies['year'].fillna(df_movies['startYear'].astype('int'))
df_movies['year']=df_movies['year'].astype('int')
df_movies['title_year']=df_movies['title_ml']+' ('+df_movies['year'].astype('str')+')'
df_movies = df_movies[['movieId', 'title_ml','year','genres_ml', 'title_year',
                       'averageRating','numVotes','genres_imdb','tconst']]


# In[193]:


# file data load ML tags
df_ml_genome_scores = pd.read_csv('../data/ml-20m/genome-scores.csv.gz')
df_ml_genome_tags = pd.read_csv('../data/ml-20m/genome-tags.csv')
df_ml_genome = df_ml_genome_scores.merge(df_ml_genome_tags, how='inner', on='tagId')


# In[200]:


# Defining a threshold to retain ML tags relevance
threshold = 0.8
df_relevant = df_ml_genome[df_ml_genome['relevance'] > threshold]

# Transpose retained ML tags into one line per movie
df_grouped_by_tags = pd.DataFrame(df_relevant.groupby('movieId')['tag'].apply(lambda x: "%s" % ' '.join(x)))
df_ml_relevant_tags=pd.merge(df_movies,df_grouped_by_tags, on='movieId', how='left')

# Taking care of NaN and 'no genre'
df_ml_relevant_tags['tag']=df_ml_relevant_tags['tag'].fillna('')
df_ml_relevant_tags['genres_imdb']=df_ml_relevant_tags['genres_imdb'].fillna('')
df_ml_relevant_tags['genres_ml']=df_ml_relevant_tags['genres_ml'].replace(to_replace = '(no genres listed)', value = '')

# Joining tags and genres
df_ml_relevant_tags['tag']=df_ml_relevant_tags['genres_ml'].str.replace('|', ' ', regex=False)+' '+df_ml_relevant_tags['tag']+' '+df_ml_relevant_tags['genres_imdb'].str.replace(',', ' ')
df_ml_relevant_tags['tag']=df_ml_relevant_tags['tag'].str.lower()


# In[203]:


# Creating word vector (for up to 3 words) and similar matrix
vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), stop_words='english')
tfidf_tag_matrix = vectorizer.fit_transform(df_ml_relevant_tags['tag'])
similar_movie = cosine_similarity(tfidf_tag_matrix, tfidf_tag_matrix)


# In[204]:


# Keeping movies indices
indices = pd.Series(df_ml_relevant_tags.index, index=df_ml_relevant_tags['title_year'])

# Building Content recommendation function
def content_recommendations(title, year, sim_matrix, indices):
    # Retrieving movie index
    title = title+' ('+year+')'
    idx = indices[title]

    # Retrieving similar movies
    sim_scores = list(enumerate(sim_matrix[idx]))
    sim_scores.remove(sim_scores[idx])

    # Sorting descending to find the closest similar movies
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # We keep the 20 closest and retrieve their respective indices
    sim_scores = sim_scores[0:20]
    movie_indices = [i[0] for i in sim_scores]

    # Cast the result as dataframe
    scores=pd.DataFrame(sim_scores)
    scores.columns=['index','score']

    # Retrieving details of each similar movies
    reco=df_ml_relevant_tags.iloc[movie_indices]

    # Final recommendation including IMDB avg rating and votes to segregate similar scores
    reco = pd.merge(reco,scores, left_on=reco.index, right_on='index',how='inner')
    reco=reco[['title_ml','year','averageRating','numVotes','score','tconst']]
    reco=reco.sort_values(by=['score','averageRating','numVotes'],ascending=[False,False,False])
    return reco


# In[205]:


content_recommendations('Innocence','2014',similar_movie,indices).head()


# In[206]:


df_ml_relevant_tags[df_ml_relevant_tags['title_ml']=='Innocence']


# In[ ]:





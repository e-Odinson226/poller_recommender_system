# Importing required libraries
import pandas as pd
from surprise import SVD
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate
from tensorflow.keras.models import Model
import numpy as np

# Load MovieLens dataset as an example
movies = pd.read_csv("movies.csv")  # Load movie data (movieId, title, genres)
ratings = pd.read_csv("ratings.csv")  # Load user ratings (userId, movieId, rating)

# Data Preprocessing: Convert genres into binary columns using one-hot encoding
genres_list = movies["genres"].str.get_dummies(
    sep="|"
)  # Create binary columns for each genre using one-hot encoding
movies = pd.concat(
    [movies, genres_list], axis=1
)  # Concatenate binary genre columns back to the 'movies' DataFrame

# Handling Missing Values: Fill missing values in the 'rating' column with the mean rating
mean_rating = ratings[
    "rating"
].mean()  # Calculate the mean rating from the 'rating' column
ratings["rating"].fillna(
    mean_rating, inplace=True
)  # Fill the missing values with the calculated mean rating

# Collaborative Filtering with Surprise
reader = Reader(
    rating_scale=(0.5, 5.0)
)  # Create a Surprise Reader with the rating scale (minimum and maximum ratings)
data = Dataset.load_from_df(
    ratings[["userId", "movieId", "rating"]], reader
)  # Load user ratings into a Surprise Dataset
trainset, testset = train_test_split(
    data, test_size=0.2
)  # Split the dataset into training and testing sets

svd = SVD()  # Create an SVD model for collaborative filtering
svd.fit(trainset)  # Train the SVD model using the training set

# Content-Based Filtering with TF-IDF
tfidf_vectorizer = TfidfVectorizer(
    stop_words="english"
)  # Create a TF-IDF vectorizer for content-based filtering
movies["genres"] = movies["genres"].str.replace(
    "|", " "
)  # Preprocess the 'genres' column (remove '|' separator)
tfidf_matrix = tfidf_vectorizer.fit_transform(
    movies["genres"]
)  # Create the TF-IDF matrix for content-based filtering
cosine_sim = linear_kernel(
    tfidf_matrix, tfidf_matrix
)  # Compute the cosine similarity between TF-IDF vectors


# Deep Learning Model for Collaborative Filtering (NeuMF) using TensorFlow-Keras
def create_neumf_model(num_users, num_movies, latent_dim=8, mlp_dims=[64, 32, 16]):
    # Define inputs for user and movie IDs
    user_input = Input(shape=(1,))
    movie_input = Input(shape=(1,))

    # Embedding layers to map user and movie IDs to low-dimensional vectors (embeddings)
    user_embedding_mf = Embedding(
        input_dim=num_users, output_dim=latent_dim, input_length=1
    )(user_input)
    movie_embedding_mf = Embedding(
        input_dim=num_movies, output_dim=latent_dim, input_length=1
    )(movie_input)

    user_embedding_mlp = Embedding(
        input_dim=num_users, output_dim=mlp_dims[0], input_length=1
    )(user_input)
    movie_embedding_mlp = Embedding(
        input_dim=num_movies, output_dim=mlp_dims[0], input_length=1
    )(movie_input)

    # Flatten the embedding outputs into 1D vectors
    user_flatten_mf = Flatten()(user_embedding_mf)
    movie_flatten_mf = Flatten()(movie_embedding_mf)

    user_flatten_mlp = Flatten()(user_embedding_mlp)
    movie_flatten_mlp = Flatten()(movie_embedding_mlp)

    # Matrix Factorization part
    mf_concat = Concatenate()([user_flatten_mf, movie_flatten_mf])
    mf_output = Dense(1)(mf_concat)

    # Multi-Layer Perceptron part
    mlp_concat = Concatenate()([user_flatten_mlp, movie_flatten_mlp])
    for dim in mlp_dims:
        mlp_concat = Dense(dim, activation="relu")(mlp_concat)
    mlp_output = Dense(1)(mlp_concat)

    # Combine Matrix Factorization and MLP parts using element-wise multiplication
    combined_output = Concatenate()([mf_output, mlp_output])
    prediction = Dense(1, activation="sigmoid")(combined_output)

    # Create the NeuMF model using the inputs and output layers
    model = Model(inputs=[user_input, movie_input], outputs=prediction)

    # Compile the model with 'adam' optimizer and binary cross-entropy loss
    model.compile(optimizer="adam", loss="binary_crossentropy")

    return model


# Mapping user and movie IDs to their corresponding indices for NeuMF
user_to_idx = {user_id: idx for idx, user_id in enumerate(ratings["userId"].unique())}
movie_to_idx = {
    movie_id: idx for idx, movie_id in enumerate(ratings["movieId"].unique())
}

# Converting user and movie IDs in the training data to numerical indices
X_train_user_indices = ratings["userId"].map(user_to_idx)
X_train_movie_indices = ratings["movieId"].map(movie_to_idx)

# Extracting target labels (ratings) from the training data (setting implicit feedback as binary labels)
y_train = (ratings["rating"] > 3.5).astype(int)

# Create and train the NeuMF model using the training data
neumf_model = create_neumf_model(
    num_users=len(user_to_idx), num_movies=len(movie_to_idx)
)
neumf_model.fit(
    [X_train_user_indices, X_train_movie_indices], y_train, epochs=10, batch_size=64
)


# Optimized Recommendation Function
def get_recommendations(user_id, num_recommendations=10):
    # Collaborative Filtering
    user_movies = ratings[ratings["userId"] == user_id]["movieId"]
    user_idx = user_to_idx[user_id]
    movie_indices = [movie_to_idx[movie_id] for movie_id in user_movies]
    user_ratings = [
        neumf_model.predict([[user_idx], [movie_idx]])[0][0]
        for movie_idx in range(len(movie_to_idx))
    ]
    unrated_movies = [
        movie_idx
        for movie_idx in range(len(movie_to_idx))
        if movie_idx not in movie_indices
    ]

    collab_recommendations = [
        (movieId, svd.predict(user_id, movieId).est)
        for movieId in movies["movieId"]
        if movieId not in user_movies
    ]
    collab_recommendations.sort(key=lambda x: x[1], reverse=True)

    similar_movies = list(enumerate(cosine_sim[movie_indices[-1]]))
    content_predictions = [
        (movies.iloc[i]["movieId"], score)
        for i, score in similar_movies
        if i not in movie_indices
    ]
    content_predictions.sort(key=lambda x: x[1], reverse=True)

    hybrid_recommendations = (
        collab_recommendations[:num_recommendations]
        + content_predictions[:num_recommendations]
    )
    hybrid_recommendations = (
        collab_recommendations[:num_recommendations]
        + content_predictions[:num_recommendations]
    )
    hybrid_recommendations.sort(key=lambda x: x[1], reverse=True)

    return [movie_id for movie_id, _ in hybrid_recommendations][:num_recommendations]


# Example Usage:
user_id = 1  # Replace with the user ID for whom you want to get recommendations
recommendations = get_recommendations(user_id)
print("Top Recommendations for User", user_id)
print(recommendations)

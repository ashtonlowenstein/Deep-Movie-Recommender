# Deep-Movie-Recommender

This is a recommender system project designed to be used with data from a popular movie cataloging website. It has a two-tower architecture where embedding vectors for query and item features are passed into dense layers, and the output is a single vector in the embedding space. The data is taken from this [kaggle set](https://www.kaggle.com/datasets/samlearner/letterboxd-movie-ratings-data?select=ratings_export.csv).

The scripts included in this repo include:
- Data cleaning and and feature engineering, which is done entirely by me using pandas
- Functions for preprocessing the data in the input pipeline for the neural network
- Definitions of the custom subclasses of keras.Model inspired by the guide [here](https://keras.io/keras_rs/examples/deep_recommender/)
- Model training and saving

This repo also includes the data that I started with (in CSV files) and the transforemd data (as TensorFlow Datasets).

The current version of this project is limited by some speed issues inherent to Keras (for example, loading a saved custom model is very slow), and training time.

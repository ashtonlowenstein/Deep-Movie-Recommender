import numpy as np
import pandas as pd
import ast
import keras
import os
import tensorflow as tf

os.environ["KERAS_BACKEND"] = "tensorflow"

from preprocessing_functions import genreEncoder, preprocess_movie, preprocess_joint

#################################################
#DATA CLEANING (AND A LITTLE FEATURE ENGINEERING)
#################################################

#Load, clean, and prep movie information
movie_df = pd.read_csv('/Users/ashtonlowenstein/Desktop/VS Code/Movie_Recommender/Movie_data/movie_data.csv')

movie_df.drop(['_id','image_url', 'imdb_link', 'production_countries', 'tmdb_link', 'imdb_id', 'tmdb_id', 'release_date', 'overview',
               'spoken_languages'], 
              axis=1, inplace=True)
movie_df['year_released'] = movie_df['year_released'].apply(lambda n : np.int64(n) if not np.isnan(n) else 0)
movie_df.dropna(subset='movie_title', inplace = True)
movie_df.dropna(subset = 'popularity', inplace = True)
for line in movie_df.itertuples():
    if (pd.isna(line[2]) and pd.isna(line[4]) and np.isnan(line[5]) and np.isnan(line[6])):
        movie_df.drop(labels = line[0], axis = 0, inplace = True)
movie_df['vote_average'] = movie_df['vote_average'].apply(lambda X: 0.0 if np.isnan(X) else X)
movie_df['vote_count'] = movie_df['vote_count'].apply(lambda X: 0.0 if np.isnan(X) else np.float64(X))
movie_df.dropna(subset='runtime', inplace = True)
movie_df.dropna(subset = 'movie_id', inplace = True)
movie_df['runtime'] = movie_df['runtime']
movie_df = movie_df[movie_df['runtime'] < 240]
movie_df = movie_df[movie_df['runtime'] > 0]
movie_df = movie_df[movie_df['vote_count'] > 1000]
movie_ids = movie_df['movie_id'].unique()
movie_dict = dict([(item, ind) for ind, item in enumerate(movie_df['movie_id'].unique())])
movie_df['movie_id_num'] = movie_df['movie_id'].apply(lambda id: movie_dict[id])
movie_df.drop('movie_id', axis=1, inplace=True)
movie_df['vote_count'] = movie_df['vote_count'].apply(lambda x: np.log(x))
movie_df['popularity'] = movie_df['popularity'].apply(lambda x: np.log(x))
vote_average_mean = movie_df['vote_average'].mean()
vote_average_std = movie_df['vote_average'].std()
movie_df['vote_average'] = movie_df['vote_average'].apply(lambda x: (x-vote_average_mean)/vote_average_std)
del vote_average_mean, vote_average_std

genre_ind = movie_df.columns.get_loc('genres')
unique_genres = set()
for line in movie_df.itertuples():
    unique_genres.update(set(
        ast.literal_eval(line[genre_ind+1])
        ))
unique_genres = list(unique_genres)
num_genres = len(unique_genres)
genre_dict = dict([(item, ind) for ind, item in enumerate(unique_genres)])

movie_df['genres_encoded'] = movie_df['genres'].apply(genreEncoder, args=(num_genres, genre_dict))
movie_df.drop('genres', axis = 1, inplace = True)
movie_df.reset_index(drop=True, inplace=True)
titles = movie_df['movie_title'].values
title_vectorizer = keras.layers.TextVectorization(
    max_tokens=10_000, output_sequence_length=16, dtype="int32"
)
title_vectorizer.adapt(titles)
movie_title_vectors = title_vectorizer(movie_df['movie_title'].values)
TITLE_TOKEN_COUNT = title_vectorizer.vocabulary_size()
movie_df.drop('movie_title', axis=1, inplace=True)
movie_df['movie_title_vec'] = pd.Series(data=np.arange(0,movie_df.index.max()+1, dtype=int))
movie_df['movie_title_vec'] = movie_df['movie_title_vec'].apply(lambda x: movie_title_vectors[x,:].numpy())

######################################

#Load, clean, and prep user information
user_df = pd.read_csv('/Users/ashtonlowenstein/Desktop/VS Code/Movie_Recommender/Movie_data/users_export.csv')

user_df.drop(['_id', 'num_ratings_pages', 'display_name'], axis = 1, inplace = True)
user_dict = dict([(item, ind) for ind, item in enumerate(user_df['username'])])
user_df['username_num'] = user_df['username'].apply(lambda name: user_dict[name])
user_df['num_reviews'] = user_df['num_reviews'].astype(float)

######################################

#Load, clean, and prep ratings information
ratings_df = pd.read_csv('/Users/ashtonlowenstein/Desktop/VS Code/Movie_Recommender/Movie_data/ratings_export.csv')

ratings_df.drop('_id', axis = 1, inplace = True)
ratings_df.dropna(subset='movie_id', inplace = True)
ratings_df = ratings_df[ratings_df['movie_id'].isin(movie_ids)]
ratings_df['movie_id_num'] = ratings_df['movie_id'].apply(lambda name: movie_dict[name])
ratings_df = ratings_df[ratings_df['user_id'].isin(user_df['username'])]
ratings_df['user_id_num'] = ratings_df['user_id'].apply(lambda name: user_dict[name])
ratings_df['rating_val'] = ratings_df['rating_val'].apply(lambda x: x/10.0)

######################################
######################################

#######################################
#FEATURE PREPROCESSING AND ENGINEERING
#######################################

num_users = user_df['username'].nunique()
num_movies = movie_df['movie_id_num'].nunique() 
num_languages = movie_df['original_language'].nunique()



joint_df = pd.merge(ratings_df, movie_df, on='movie_id_num')
joint_size = joint_df.shape[0]

del user_df

genresMat = np.zeros((joint_size, num_genres))
for line in joint_df.itertuples():
    genresMat[line[0], :] = line[12]
genresTens = keras.ops.convert_to_tensor(genresMat, dtype="int64")
del genresMat

titlesMat = np.zeros((joint_size, 16))
for line in joint_df.itertuples():
    titlesMat[line[0], :] = line[13]
titlesTens = keras.ops.convert_to_tensor(titlesMat, dtype="float64")
del titlesMat

languagesList = []
for line in joint_df.itertuples():
    languagesList.append(line[6])
languagesTens = keras.ops.convert_to_tensor(languagesList, dtype=str)
del languagesList

feedbackMat = np.zeros((num_users, num_movies), dtype=np.bool)
for line in ratings_df.itertuples():
    if line[2]:
        feedbackMat[line[5], line[4]] = 1
feedbackTens = keras.ops.convert_to_tensor(feedbackMat, dtype=bool)
del feedbackMat

ratings_df_dict = {'user_id' : joint_df['user_id_num'].to_numpy(dtype=np.int64).reshape((joint_df.shape[0],1))}
labels_tensor = joint_df['rating_val'].to_numpy(dtype=np.float64).reshape((joint_df.shape[0],1))
movie_df_int_dict = {key: value.to_numpy(np.int64)[:, tf.newaxis] for key, value in 
                 dict(joint_df[['movie_id_num', 'runtime', 'year_released']]).items()}
movie_df_float_dict = {key: value.to_numpy(np.float64)[:, tf.newaxis] for key, value in 
                 dict(joint_df[['popularity', 'vote_count', 'vote_average']]).items()}
joint_dict = dict()
joint_dict['user_id'] = ratings_df_dict['user_id']
joint_dict['movie_id_num'] = movie_df_int_dict['movie_id_num']
joint_dict['runtime'] = movie_df_int_dict['runtime']
joint_dict['year_released'] = movie_df_int_dict['year_released']
joint_dict['language'] = languagesTens
joint_dict['popularity'] = movie_df_float_dict['popularity']
joint_dict['vote_count'] = movie_df_float_dict['vote_count']
joint_dict['vote_average'] = movie_df_float_dict['vote_average']
joint_dict['movie_title_vec'] = titlesTens
joint_dict['genres_encoded'] = genresTens
joint_ds = tf.data.Dataset.from_tensor_slices((joint_dict, labels_tensor))

del titlesTens, genresTens, languagesTens, movie_df_int_dict, movie_df_float_dict, joint_dict, ratings_df_dict, labels_tensor

genresMat = np.zeros((num_movies, num_genres))
for line in movie_df.itertuples():
    genresMat[line[0], :] = line[8]
genresTens = keras.ops.convert_to_tensor(genresMat, dtype="int64")
del genresMat

titlesMat = np.zeros((num_movies, 16))
for line in movie_df.itertuples():
    titlesMat[line[0], :] = line[9]
titlesTens = keras.ops.convert_to_tensor(titlesMat, dtype="float64")
del titlesMat

languagesList = []
for line in movie_df.itertuples():
    languagesList.append(line[1])
languagesTens = keras.ops.convert_to_tensor(languagesList, dtype=str)
del languagesList

movie_df_int_dict = {key: value.to_numpy(np.int64)[:, tf.newaxis] for key, value in 
                 dict(movie_df[['movie_id_num', 'runtime', 'year_released']]).items()}
movie_df_float_dict = {key: value.to_numpy(np.float64)[:, tf.newaxis] for key, value in 
                 dict(movie_df[['popularity', 'vote_count', 'vote_average']]).items()}
movie_dict = dict()
movie_dict['movie_id_num'] = movie_df_int_dict['movie_id_num']
movie_dict['runtime'] = movie_df_int_dict['runtime']
movie_dict['year_released'] = movie_df_int_dict['year_released']
movie_dict['language'] = languagesTens
movie_dict['popularity'] = movie_df_float_dict['popularity']
movie_dict['vote_count'] = movie_df_float_dict['vote_count']
movie_dict['vote_average'] = movie_df_float_dict['vote_average']
movie_dict['movie_title_vec'] = titlesTens
movie_dict['genres_encoded'] = genresTens

MOVIES_DATASET = tf.data.Dataset.from_tensor_slices(movie_dict)

del joint_df, movie_df, ratings_df, titlesTens, genresTens, languagesTens, movie_dict, movie_df_float_dict, movie_df_int_dict, movie_title_vectors

#Create the Keras features and Feature Space for final structured preprocessing

item_runtime_feature = keras.utils.FeatureSpace.integer_hashed(
    num_bins = 24,
    output_mode="one_hot"
)
item_releaseYear_feature = keras.utils.FeatureSpace.integer_hashed(
    num_bins = 12,
    output_mode="one_hot"
)
item_language_feature = keras.utils.FeatureSpace.string_categorical(
    max_tokens=None,
    num_oov_indices=1,
    output_mode="one_hot"
)

movie_featureSpace = keras.utils.FeatureSpace(
    features={
        "runtime" : item_runtime_feature,
        "year_released" : item_releaseYear_feature,
        "language" : item_language_feature
    },
    output_mode='dict' #crucial that this is set to 'dict', otherwise the preprocessing functions don't work
)

movie_featureSpace.adapt(MOVIES_DATASET)

######################################
######################################

#########################################
#CREATING THE DATASETS USED FOR TRAINING
#########################################

MOVIES_DATASET = MOVIES_DATASET.map(lambda x: preprocess_movie(x, movie_featureSpace), num_parallel_calls=tf.data.AUTOTUNE, ).batch(1)

shuffled_joint = joint_ds.map(lambda x,y: preprocess_joint(x,y,movie_featureSpace), num_parallel_calls=tf.data.AUTOTUNE, ).shuffle(
    joint_ds.cardinality(), seed = 12, reshuffle_each_iteration=False
)

joint_cardinality = joint_ds.cardinality().numpy()
joint_train_num = int(joint_cardinality*0.8)
joint_test_num = joint_cardinality-joint_train_num

train_joint = shuffled_joint.take(joint_train_num).batch(500).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
test_joint = shuffled_joint.skip(joint_train_num).take(joint_test_num).batch(500).cache().prefetch(buffer_size=tf.data.AUTOTUNE)

train_joint.save('/Users/ashtonlowenstein/Desktop/VS Code/Movie_Recommender/Movie_data/train_joint')
test_joint.save('/Users/ashtonlowenstein/Desktop/VS Code/Movie_Recommender/Movie_data/test_joint')
MOVIES_DATASET.save('/Users/ashtonlowenstein/Desktop/VS Code/Movie_Recommender/Movie_data/MOVIES_DATASET')
tf.io.write_file('/Users/ashtonlowenstein/Desktop/VS Code/Movie_Recommender/Movie_data/feedback',feedbackTens)
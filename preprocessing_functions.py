import numpy as np
import ast
import tensorflow as tf

def genreEncoder(s: str, num_genres, genre_dict):
    '''
    The input s is a string that consists of a list of strings (the list of genre names is stored in a list, which is stored
    in the dataframe as a string/object). This function is to be used to multi-hot encode the genres and store that information in 
    a different vector for each movie.
    '''
    vec = np.zeros((num_genres))
    for g in ast.literal_eval(s):
        ind = genre_dict[g]
        vec[ind] = 1
    return vec

def preprocess_movie(x, movie_featureSpace):
    """
    Preprocess the joint ratings/movie dataset according to the feature transformations in the movie feature space.

    Args:
        x (tf.data.Datset): Movie dataset with columns *movie_id_num*, *runtime*, *popularity*, *vote_count*,
          *vote_average*, *year_released*, *language*, *movie_title_vec*, *genres_encoded*
          
    **kwargs:
        featureSpace (keras.utils.FeatureSpace)

    Returns:
        tf.data.Dataset: Transformed dataset with rescaled columns.
    """
    
    
    features = movie_featureSpace(
        {
            "runtime" : x['runtime'],
            "year_released" : x['year_released'],
            "language" : x['language']
        }
    )
    
    return {
        'movie_id_num' : x['movie_id_num'],
        'runtime' : tf.cast(features['runtime'], dtype=tf.float64),
        "popularity" : tf.cast(x["popularity"], dtype=tf.float64),
        "vote_count" : tf.cast(x["vote_count"], dtype=tf.float64),
        "vote_average" : tf.cast(x["vote_average"], dtype=tf.float64),
        "year_released" : tf.cast(features["year_released"], dtype=tf.float64),
        "language" : tf.cast(features["language"], dtype=tf.float64),
        "movie_title_vec" : tf.cast(x['movie_title_vec'], dtype=tf.float64),
        "genres_encoded" : tf.cast(x['genres_encoded'], dtype=tf.float64)
    }
    
def preprocess_joint(x, y, movie_featureSpace):
    """
    Preprocess the joint ratings/movie dataset according to the feature transformations in the movie feature space.

    Args:
        x (tf.data.Datset): Movie dataset with columns *movie_id_num*, *runtime*, *popularity*, *vote_count*,
          *vote_average*, *year_released*, *language*, *movie_title_vec*, *genres_encoded*
    
    Kwargs:
        featureSpace (keras.utils.FeatureSpace)

    Returns:
        tf.data.Dataset: Transformed dataset with rescaled columns.
    """
    
    features = movie_featureSpace(
        {
            "runtime" : x['runtime'],
            "year_released" : x['year_released'],
            "language" : x['language']
        }
    )
    
    #features = {k: tf.squeeze(v, axis=0) for k, v in features.items()}
    
    return ({
            'user_id' : tf.squeeze(x['user_id']),
            'movie_id_num' : x['movie_id_num'],
            'runtime' : tf.cast(features['runtime'], dtype=tf.float64),
            "popularity" : tf.cast(x["popularity"], dtype=tf.float64),
            "vote_count" : tf.cast(x["vote_count"], dtype=tf.float64),
            "vote_average" : tf.cast(x["vote_average"], dtype=tf.float64),
            "year_released" : tf.cast(features["year_released"], dtype=tf.float64),
            "language" : tf.cast(features["language"], dtype=tf.float64),
            "movie_title_vec" : tf.cast(x['movie_title_vec'], dtype=tf.float64),
            "genres_encoded" : tf.cast(x['genres_encoded'], dtype=tf.float64)
        },
        y
    )
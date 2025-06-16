import keras
import os
import tensorflow as tf
import keras_rs

os.environ["KERAS_BACKEND"] = "tensorflow"

@keras.saving.register_keras_serializable(package="Recommender")
class QueryModel(keras.Model):
    def __init__(
        self, 
        layer_sizes, 
        num_users, feedback, 
        embedding_dimension = 32
        ):
        """Construct a model for encoding user queries.

        Args:
          layer_sizes: A list of integers where the i-th entry represents the
            number of units the i-th layer contains.
          embedding_dimension: Output dimension for all embedding tables.
          num_users: Overall number of users whose reviews are under consideration.
          feedback: Implicit feedback matrix corresponding to the users and their reviews.
          embedding_dimension: Dimension for the query and item embedding space.
        """
        
        super().__init__()
        
        self.feedback = feedback

        self.user_embedding = keras.models.Sequential(
        [
          keras.layers.Embedding(
              num_users,
              embedding_dimension
          ),
          keras.layers.GlobalAveragePooling1D()
        ]
        )
        
        self.dense_layers = keras.Sequential()
        for layer_size in layer_sizes:
            self.dense_layers.add(keras.layers.Dense(layer_size, activation="relu"))
            
        self.dense_layers.add(keras.layers.Dense(embedding_dimension))
    
    def call(self, inputs):
        user_ids = inputs['user_id']
        feature_embedding = self.user_embedding(
            tf.gather(self.feedback, user_ids)
        )
        return self.dense_layers(feature_embedding)
    
    def get_config(self):
      config = super().get_config()
      config.update(
        {
          'feedback' : self.feedback,
          'user_embedding' : self.user_embedding,
          'dense_layers' : self.dense_layers
        }
      )
      
      return config
        
    @classmethod
    def from_config(cls, config):
      config["user_embedding"] = keras.saving.deserialize_keras_object(config['user_embedding'])
      config["dense_layers"] = keras.saving.deserialize_keras_object(config['dense_layers'])
      return cls(**config)

@keras.saving.register_keras_serializable(package="Recommender")
class CandidateModel(keras.Model):
    """Model for encoding candidates (movies)."""

    def __init__(
        self, 
        layer_sizes,
        TITLE_TOKEN_COUNT,
        num_genres, 
        embedding_dimension=32
        ):
        """Construct a model for encoding candidates (movies).

        Args:
          layer_sizes: A list of integers where the i-th entry represents the
            number of units the i-th layer contains.
          embedding_dimension: Output dimension for all embedding tables.
        """
        super().__init__()
        
        # Take all the title tokens for the title of the movie, embed each
        # token, and then take the mean of all token embeddings.
        self.movie_title_embedding = keras.Sequential(
            [
                keras.layers.Embedding(
                    # +1 for OOV token, which is used for padding
                    TITLE_TOKEN_COUNT + 1,
                    embedding_dimension,
                    mask_zero=False
                ),
                keras.layers.GlobalAveragePooling1D(),
            ]
        )
        
        # Take all the genres for the movie, embed each genre, and then take the
        # mean of all genre embeddings.
        self.movie_genre_embedding = keras.models.Sequential(
        [
            keras.layers.Embedding(
                num_genres + 1,
                embedding_dimension,
                mask_zero=False
            ),
            keras.layers.GlobalAveragePooling1D()
        ]
        )
        
        self.movie_language_embedding = keras.models.Sequential(
        [
            keras.layers.Embedding(
                24,
                embedding_dimension,
                mask_zero=False
            ),
            keras.layers.GlobalAveragePooling1D(),
        ]
        )
        
        self.movie_yearReleased_embedding = keras.models.Sequential(
        [
            keras.layers.Embedding(
                13,
                embedding_dimension,
                mask_zero=False
            ),
            keras.layers.GlobalAveragePooling1D()
        ]
        )
        
        self.movie_runtime_embedding = keras.models.Sequential(
        [
            keras.layers.Embedding(
                32,
                embedding_dimension,
                mask_zero=False
            ),
            keras.layers.GlobalAveragePooling1D()
        ]
        )
        
        self.dense_layers = keras.Sequential()
        
        # Use the ReLU activation for all but the last layer.
        for layer_size in layer_sizes:
            self.dense_layers.add(keras.layers.Dense(layer_size, activation="relu"))

        # No activation for the last layer.
        self.dense_layers.add(keras.layers.Dense(embedding_dimension, use_bias=False))
        
        # movies_ds has features: 'movie_id_num', 'runtime', 'popularity', 'vote_count', 'vote_average', 'year_released', 'language',
        # 'movie_title_vec', 'genres_encoded'
        
    def call(self, inputs):
        movie_title_vec = inputs["movie_title_vec"]
        movie_genres = inputs["genres_encoded"]
        movie_languages = inputs["language"]
        movie_yearReleased = inputs["year_released"]
        movie_runtime = inputs["runtime"]
        movie_popularity = inputs["popularity"]
        movie_voteCount = inputs["vote_count"]
        movie_voteAverage = inputs["vote_average"]
        feature_embedding = keras.ops.concatenate(
            [
                self.movie_title_embedding(movie_title_vec),
                self.movie_genre_embedding(movie_genres),
                self.movie_language_embedding(movie_languages),
                self.movie_yearReleased_embedding(movie_yearReleased),
                self.movie_runtime_embedding(movie_runtime),
                movie_popularity,
                movie_voteCount,
                movie_voteAverage
            ],
            axis=1,
        )
        return self.dense_layers(feature_embedding)
    
    def get_config(self):
        config = super().get_config()
        config.update(
            {
                'movie_title_embedding' : self.movie_title_embedding,
                'movie_genre_embedding' : self.movie_genre_embedding,
                'movie_language_embedding' : self.movie_language_embedding,
                'movie_yearReleased_embedding' : self.movie_yearReleased_embedding,
                'movie_runtime_embedding' : self.movie_runtime_embedding,
                'dense_layers' : self.dense_layers
            }
        )
        
        return config
    
    @classmethod
    def from_config(cls, config):
        config['movie_title_embedding'] = keras.saving.deserialize_keras_object(config['movie_title_embedding'])
        config['movie_genre_embedding'] = keras.saving.deserialize_keras_object(config['movie_genre_embedding'])
        config['movie_language_embedding'] = keras.saving.deserialize_keras_object(config['movie_language_embedding'])
        config['movie_yearReleased_embedding'] = keras.saving.deserialize_keras_object(config['movie_yearReleased_embedding'])
        config['movie_runtime_embedding'] = keras.saving.deserialize_keras_object(config['movie_runtime_embedding'])
        config['dense_layers'] = keras.saving.deserialize_keras_object(config['dense_layers'])
        
        return cls(**config)
    
@keras.saving.register_keras_serializable(package="Recommender")
class RetrievalModel(keras.Model):
    """Combined two-tower model."""

    def __init__(
        self,
        feedback,
        MOVIES_DATASET,
        TITLE_TOKEN_COUNT,
        num_genres,
        layer_sizes=[],
        embedding_dimension=32,
        retrieval_k=50,
    ):
        """Construct a two-tower model.

        Args:
          layer_sizes: A list of integers where the i-th entry represents the
            number of units the i-th layer contains.
          embedding_dimension: Output dimension for all embedding tables.
          retrieval_k: How many candidate movies to retrieve.
        """
        super().__init__()
        
        self.query_model = QueryModel(layer_sizes, embedding_dimension, feedback)
        self.candidate_model = CandidateModel(layer_sizes, embedding_dimension, TITLE_TOKEN_COUNT, num_genres)
        
        self.retrieval = keras_rs.layers.BruteForceRetrieval(
            k=retrieval_k, return_scores=False
        )
        
        self.update_candidates(MOVIES_DATASET)  # Provide an initial set of candidates
        
        self.loss_fn = keras.losses.MeanSquaredError()
        self.top_k_metric = keras.metrics.SparseTopKCategoricalAccuracy(
            k=retrieval_k, from_sorted_ids=True
        )
        
    def update_candidates(self, MOVIES_DATASET):
        self.retrieval.update_candidates(
            self.candidate_model.predict(MOVIES_DATASET)
        )
        
    def call(self, inputs, training=False):
        
        query_embeddings = self.query_model({
            "user_id" : inputs['user_id']
        })
        
        candidate_embeddings = self.candidate_model({
            "runtime" : inputs['runtime'],
            "popularity" : inputs['popularity'],
            "vote_count" : inputs['vote_count'],
            "vote_average" : inputs['vote_average'],
            "year_released" : inputs['year_released'],
            "language" : inputs['language'],
            "movie_title_vec" : inputs['movie_title_vec'],
            "genres_encoded" : inputs['genres_encoded']
        })

        result = {
            "query_embeddings": query_embeddings,
            "candidate_embeddings": candidate_embeddings,
        }
        if not training:
            # No need to spend time extracting top predicted movies during
            # training, they are not used.
            result["predictions"] = self.retrieval(query_embeddings)
        return result
        
    def evaluate(
        self,
        x=None,
        y=None,
        batch_size=None,
        verbose="auto",
        sample_weight=None,
        steps=None,
        callbacks=None,
        return_dict=False,
        **kwargs,
    ):
        """Overridden to update the candidate set.

        Before evaluating the model, we need to update our retrieval layer by
        re-computing the values predicted by the candidate model for all the
        candidates.
        """
        self.update_candidates()
        return super().evaluate(
            x,
            y,
            batch_size=batch_size,
            verbose=verbose,
            sample_weight=sample_weight,
            steps=steps,
            callbacks=callbacks,
            return_dict=return_dict,
            **kwargs,
        )
            
    def compute_loss(self, x, y, y_pred, sample_weight, training=True):
        query_embeddings = y_pred["query_embeddings"]
        candidate_embeddings = y_pred["candidate_embeddings"]

        labels = keras.ops.expand_dims(y, -1)
        # Compute the affinity score by multiplying the two embeddings.
        scores = keras.ops.sum(
            keras.ops.multiply(query_embeddings, candidate_embeddings),
            axis=1,
            keepdims=True,
        )
        return self.loss_fn(labels, scores, sample_weight)
    
    def compute_metrics(self, x, y, y_pred, sample_weight=None):
        if "predictions" in y_pred:
            # We are evaluating or predicting. Update `top_k_metric`.
            movie_ids = x["movie_id_num"]
            predictions = y_pred["predictions"]
            # For `top_k_metric`, which is a `SparseTopKCategoricalAccuracy`, we
            # only take top rated movies, and we put a weight of 0 for the rest.
            rating_weight = keras.ops.cast(keras.ops.greater(y, 0.9), "float32")
            sample_weight = (
                rating_weight
                if sample_weight is None
                else keras.ops.multiply(rating_weight, sample_weight)
            )
            self.top_k_metric.update_state(
                movie_ids, predictions, sample_weight=sample_weight
            )
            return self.get_metrics_result()
        else:
            # We are training. `top_k_metric` is not updated and is zero, so
            # don't report it.
            result = self.get_metrics_result()
            result.pop(self.top_k_metric.name)
            return result
        
    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "query_model": self.query_model,
                "candidate_model": self.candidate_model,
                "retrieval": self.retrieval
            }
        )
        return config

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "query_model": self.query_model,
                "candidate_model": self.candidate_model,
                "retrieval": self.retrieval
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        # Note that you can also use [`keras.saving.deserialize_keras_object`](/api/models/model_saving_apis/serialization_utils#deserializekerasobject-function) here
        config["query_model"] = keras.saving.deserialize_keras_object(config["query_model"])
        config["candidate_model"] = keras.saving.deserialize_keras_object(config["candidate_model"])
        config["retrieval"] = keras.saving.deserialize_keras_object(config["retrieval"])
        return cls(**config)

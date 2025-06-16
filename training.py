import tensorflow as tf
import keras
import keras_rs
import numpy as np
import os

os.environ["KERAS_BACKEND"] = "tensorflow"

from model_classes import QueryModel, CandidateModel, RetrievalModel

train_data = tf.data.Dataset.load('/Users/ashtonlowenstein/Desktop/VS Code/Movie_Recommender/Movie_data/train_joint')
test_data = tf.data.Dataset.load('/Users/ashtonlowenstein/Desktop/VS Code/Movie_Recommender/Movie_data/test_joint')
MOVIES_DATASET = tf.data.Dataset.load('/Users/ashtonlowenstein/Desktop/VS Code/Movie_Recommender/Movie_data/MOVIES_DATASET')

joint_cardinality = train_data.cardinality().numpy() + test_data.cardinality.numpy()
train_num = int(joint_cardinality*0.8)
test_num = joint_cardinality-train_num

train_data = train_data.take(train_num).batch(500).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
test_data = test_data.skip(train_num).take(test_num).batch(500).cache().prefetch(buffer_size=tf.data.AUTOTUNE)

feedback = tf.io.read_file('/Users/ashtonlowenstein/Desktop/VS Code/Movie_Recommender/Movie_data/feedback')

model = RetrievalModel(
    feedback=feedback,
    MOVIES_DATASET=MOVIES_DATASET,
    TITLE_TOKEN_COUNT=3331, #should read this from save
    num_genres=19, #should read this from save
    layer_sizes=[64]
)

model.compile(optimizer=keras.optimizers.Adagrad(0.05))

NUM_EPOCHS = 30
history = model.fit(
    train_data,
    validation_data=test_data,
    validation_freq=5,
    epochs=NUM_EPOCHS,
)

model.save('/Users/ashtonlowenstein/Desktop/VS Code/Movie_Recommender/model.keras')
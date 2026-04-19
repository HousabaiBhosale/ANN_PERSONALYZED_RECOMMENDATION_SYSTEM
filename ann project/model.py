import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate
from tensorflow.keras.optimizers import Adam

# Load dataset
ratings = pd.read_csv("dataset/ratings.csv")

# Encode userId & movieId
user_ids = ratings['userId'].unique().tolist()
movie_ids = ratings['movieId'].unique().tolist()

user2user_encoded = {x: i for i, x in enumerate(user_ids)}
movie2movie_encoded = {x: i for i, x in enumerate(movie_ids)}

ratings['user'] = ratings['userId'].map(user2user_encoded)
ratings['movie'] = ratings['movieId'].map(movie2movie_encoded)

num_users = len(user2user_encoded)
num_movies = len(movie2movie_encoded)

# Prepare data
X = ratings[['user', 'movie']].values
y = ratings['rating'].values / 5.0  # normalize

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
user_input = Input(shape=(1,))
movie_input = Input(shape=(1,))

user_embedding = Embedding(num_users, 50)(user_input)
movie_embedding = Embedding(num_movies, 50)(movie_input)

user_vec = Flatten()(user_embedding)
movie_vec = Flatten()(movie_embedding)

concat = Concatenate()([user_vec, movie_vec])

dense1 = Dense(128, activation='relu')(concat)
dense2 = Dense(64, activation='relu')(dense1)

output = Dense(1, activation='sigmoid')(dense2)

model = Model([user_input, movie_input], output)

model.compile(
    loss='mse',
    optimizer=Adam(learning_rate=0.001),
    metrics=['mae']
)

# Train
model.fit(
    [X_train[:, 0], X_train[:, 1]],
    y_train,
    epochs=5,
    batch_size=64,
    validation_data=([X_test[:, 0], X_test[:, 1]], y_test)
)

# Save model
model.save("model.h5")

# Save encoders
with open("user_encoder.pkl", "wb") as f:
    pickle.dump(user2user_encoded, f)

with open("movie_encoder.pkl", "wb") as f:
    pickle.dump(movie2movie_encoded, f)

print("✅ Model and encoders saved successfully!")
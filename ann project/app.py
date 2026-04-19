from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load model
model = load_model("model.h5", compile=False, safe_mode=False)

# Load encoders
with open("user_encoder.pkl", "rb") as f:
    user_encoder = pickle.load(f)

with open("movie_encoder.pkl", "rb") as f:
    movie_encoder = pickle.load(f)

# Load dataset
ratings = pd.read_csv("dataset/ratings.csv")
movies = pd.read_csv("dataset/movies.csv")

all_movie_ids = movies['movieId'].unique()

# Evaluate approximate MAE on a sample of ratings
try:
    sample_size = min(len(ratings), 2000)
    sample_df = ratings.sample(n=sample_size, random_state=42)
    valid_users = []
    valid_movies = []
    actuals = []
    for _, row in sample_df.iterrows():
        u_enc = user_encoder.get(row['userId'])
        m_enc = movie_encoder.get(row['movieId'])
        if u_enc is not None and m_enc is not None:
            valid_users.append(u_enc)
            valid_movies.append(m_enc)
            actuals.append(row['rating'])
    
    if len(valid_users) > 0:
        val_preds = model.predict([np.array(valid_users), np.array(valid_movies)], verbose=0)
        val_preds = val_preds.flatten() * 5.0
        app_mae = f"{np.mean(np.abs(val_preds - np.array(actuals))):.2f}"
    else:
        app_mae = "—"
except Exception as e:
    print(f"Failed to calculate MAE: {e}")
    app_mae = "—"

# Home route (frontend)
@app.route("/")
def index():
    return render_template("index.html")


# Predict rating
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    user_id = data['userId']
    movie_id = data['movieId']

    user_enc = user_encoder.get(user_id)
    movie_enc = movie_encoder.get(movie_id)

    if user_enc is None or movie_enc is None:
        return jsonify({"error": "Invalid user or movie"})

    pred = model.predict([np.array([user_enc]), np.array([movie_enc])])
    rating = float(pred[0][0] * 5)
    
    movie_df = movies[movies['movieId'] == movie_id]
    if not movie_df.empty:
        movie = movie_df.iloc[0]
        movie_data = {
            "title": movie['title'],
            "genres": movie.get('genres', '') if 'genres' in movie else ''
        }
    else:
        movie_data = None

    return jsonify({
        "predicted_rating": rating,
        "userId": user_id,
        "movieId": movie_id,
        "movie": movie_data
    })


# Recommend movies
@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    user_id = data['userId']
    n = data.get('n', 10)
    genre_filter = data.get('genre', '').lower()

    if user_id not in user_encoder:
        return jsonify({"error": "User not found"})

    user_enc = user_encoder[user_id]

    watched = ratings[ratings['userId'] == user_id]['movieId'].values
    unseen = [m for m in all_movie_ids if m not in watched]
    
    if genre_filter and 'genres' in movies.columns:
        valid_movies = movies[movies['genres'].str.lower().str.contains(genre_filter, na=False)]['movieId'].values
        unseen = [m for m in unseen if m in valid_movies]

    predictions = []

    for movie_id in unseen[:300]:
        movie_enc = movie_encoder.get(movie_id)
        if movie_enc is None:
            continue

        pred = model.predict([np.array([user_enc]), np.array([movie_enc])])
        predictions.append((movie_id, float(pred[0][0])))

    predictions.sort(key=lambda x: x[1], reverse=True)

    top = predictions[:n]

    result = []
    for movie_id, score in top:
        movie_df = movies[movies['movieId'] == movie_id]
        if not movie_df.empty:
            movie = movie_df.iloc[0]
            result.append({
                "movieId": int(movie_id),
                "title": movie['title'],
                "genres": movie.get('genres', '') if 'genres' in movie else '',
                "predicted_rating": score * 5
            })

    return jsonify({
        "recommendations": result,
        "count": len(result),
        "userId": user_id
    })


# Rate movie
@app.route('/rate', methods=['POST'])
def rate():
    global ratings

    data = request.json

    new = pd.DataFrame({
        "userId": [data['userId']],
        "movieId": [data['movieId']],
        "rating": [data['rating']]
    })

    ratings = pd.concat([ratings, new], ignore_index=True)

    return jsonify({
        "message": "Rating added",
        "userId": data['userId'],
        "movieId": data['movieId'],
        "rating": data['rating']
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "mae": app_mae})


@app.route('/movies/popular', methods=['GET'])
def popular():
    n = int(request.args.get('n', 8))
    top_movie_ids = ratings['movieId'].value_counts().head(n).index
    result = []
    for movie_id in top_movie_ids:
        movie_df = movies[movies['movieId'] == movie_id]
        if not movie_df.empty:
            movie = movie_df.iloc[0]
            result.append({
                "movieId": int(movie_id),
                "title": movie['title'],
                "genres": movie.get('genres', '') if 'genres' in movie else '',
                "score": ratings[ratings['movieId'] == movie_id]['rating'].mean()
            })
    return jsonify({"popular": result})


@app.route('/movies/search', methods=['GET'])
def search():
    q = request.args.get('q', '').lower()
    if not q:
        return jsonify({"results": [], "count": 0})
    
    matches = movies[movies['title'].str.lower().str.contains(q, na=False)].head(20)
    result = []
    for _, movie in matches.iterrows():
        result.append({
            "movieId": int(movie['movieId']),
            "title": movie['title'],
            "genres": movie.get('genres', '') if 'genres' in movie else ''
        })
    return jsonify({"results": result, "count": len(result)})


@app.route('/user/<int:user_id>/ratings', methods=['GET'])
def user_ratings(user_id):
    user_ratings = ratings[ratings['userId'] == user_id]
    result = []
    for _, row in user_ratings.iterrows():
        movie_df = movies[movies['movieId'] == row['movieId']]
        title = movie_df.iloc[0]['title'] if not movie_df.empty else "Unknown"
        result.append({
            "movieId": int(row['movieId']),
            "title": title,
            "rating": float(row['rating'])
        })
    return jsonify({"ratings": result})


if __name__ == "__main__":
    app.run(debug=True)
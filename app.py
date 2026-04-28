from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import pickle
import os
from tensorflow.keras.models import load_model

app = Flask(__name__)

# ✅ Robust Path Handling
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_path(rel_path):
    return os.path.join(BASE_DIR, rel_path)

# ✅ Load model
model_path = get_path("model.h5")
model = load_model(model_path, compile=False)

# Load encoders
with open(get_path("user_encoder.pkl"), "rb") as f:
    user_encoder = pickle.load(f)

with open(get_path("movie_encoder.pkl"), "rb") as f:
    movie_encoder = pickle.load(f)

# ✅ Safe dataset loading
try:
    ratings = pd.read_csv(get_path("dataset/ratings.csv"))
    movies = pd.read_csv(get_path("dataset/movies.csv"))
except Exception as e:
    print("Dataset load error:", e)
    ratings = pd.DataFrame()
    movies = pd.DataFrame()

all_movie_ids = movies['movieId'].unique() if not movies.empty else []

# ✅ Reduced load for Render (fixed)
try:
    sample_size = min(len(ratings), 500)
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
        val_preds = model.predict(
            [np.array(valid_users), np.array(valid_movies)], verbose=0
        )
        val_preds = val_preds.flatten() * 5.0
        mae_val = float(np.mean(np.abs(val_preds - np.array(actuals))))
        app_mae = f"{mae_val:.2f}"
        app_accuracy = f"{(1.0 - (mae_val/5.0))*100:.1f}%"
    else:
        app_mae = "—"
        app_accuracy = "—"

except Exception as e:
    print(f"Failed to calculate MAE: {e}")
    app_mae = "—"
    app_accuracy = "—"


# Home route
@app.route("/")
def index():
    return render_template("index.html")


# Predict rating
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    user_id = data['userId']
    movie_id_input = data['movieId']

    user_enc = user_encoder.get(user_id)
    if user_enc is None:
        return jsonify({"error": "Invalid user"})

    if isinstance(movie_id_input, str) and ',' in movie_id_input:
        movie_ids = [int(m.strip()) for m in movie_id_input.split(',')]
    elif isinstance(movie_id_input, list):
        movie_ids = [int(m) for m in movie_id_input]
    else:
        movie_ids = [int(movie_id_input)]

    # Collect valid encodings for batch prediction
    valid_mids = []
    valid_encs = []
    for mid in movie_ids:
        enc = movie_encoder.get(mid)
        if enc is not None:
            valid_mids.append(mid)
            valid_encs.append(enc)

    results = []
    if valid_encs:
        # ✅ Batch Prediction
        u_arr = np.full(len(valid_encs), user_enc)
        m_arr = np.array(valid_encs)
        batch_preds = model.predict([u_arr, m_arr], verbose=0).flatten()

        user_rating_count = len(ratings[ratings['userId'] == user_id])
        
        for mid, score in zip(valid_mids, batch_preds):
            rating = float(score * 5.0)
            confidence = min(99.9, 50.0 + (user_rating_count * 0.5) - abs(rating - 3.0)*2.0)
            
            movie_df = movies[movies['movieId'] == mid]
            movie_data = {
                "title": movie_df.iloc[0]['title'] if not movie_df.empty else f"Movie #{mid}",
                "genres": movie_df.iloc[0].get('genres', '') if not movie_df.empty else ''
            }

            results.append({
                "predicted_rating": rating,
                "confidence_score": round(confidence, 1),
                "userId": user_id,
                "movieId": mid,
                "movie": movie_data
            })

    # Add back errors for invalid movies
    invalid_mids = [mid for mid in movie_ids if mid not in valid_mids]
    for mid in invalid_mids:
        results.append({"movieId": mid, "error": "Invalid movie", "predicted_rating": 0})

    if len(results) == 1:
        return jsonify(results[0])
    return jsonify({"predictions": results})


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
        valid_movies = movies[
            movies['genres'].str.lower().str.contains(genre_filter, na=False)
        ]['movieId'].values
        unseen = [m for m in unseen if m in valid_movies]

    # Limit to top 400 potential candidates for stability
    unseen = unseen[:400]
    
    # Collect valid encodings for batch prediction
    valid_mids = []
    valid_encs = []
    for mid in unseen:
        enc = movie_encoder.get(mid)
        if enc is not None:
            valid_mids.append(mid)
            valid_encs.append(enc)

    if not valid_encs:
        return jsonify({"recommendations": [], "count": 0, "userId": user_id})

    # ✅ Batch Prediction (Fast & Stable)
    u_arr = np.full(len(valid_encs), user_enc)
    m_arr = np.array(valid_encs)
    batch_preds = model.predict([u_arr, m_arr], verbose=0).flatten()

    predictions = list(zip(valid_mids, batch_preds))
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
                "genres": movie.get('genres', ''),
                "predicted_rating": float(score * 5.0)
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


@app.route('/login', methods=['POST'])
def login():
    data = request.json
    user_id = data.get('userId')

    if not user_id:
        return jsonify({"error": "User ID required"}), 400

    try:
        uid = int(user_id)
        if uid in user_encoder:
            return jsonify({"status": "success", "userId": uid})
        else:
            return jsonify({"error": "User not found"}), 404
    except ValueError:
        return jsonify({"error": "Invalid format"}), 400


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "mae": app_mae, "accuracy": app_accuracy})


# ✅ Get popular movies (new)
@app.route('/movies/popular', methods=['GET'])
def popular_movies():
    n = request.args.get('n', default=8, type=int)
    if ratings.empty or movies.empty:
        return jsonify({"popular": []})

    # Calculate popularity stats
    pop_stats = ratings.groupby('movieId').agg({'rating': ['mean', 'count']})
    pop_stats.columns = ['avg_rating', 'count']

    # Filter for movies with at least 5 ratings and sort by average
    popular_ids = pop_stats[pop_stats['count'] > 5].sort_values(by='avg_rating', ascending=False).head(n).index.tolist()

    result = []
    for mid in popular_ids:
        m_info = movies[movies['movieId'] == mid]
        if not m_info.empty:
            result.append({
                "movieId": int(mid),
                "title": m_info.iloc[0]['title'],
                "genres": m_info.iloc[0].get('genres', ''),
                "predicted_rating": float(pop_stats.loc[mid, 'avg_rating'])
            })
    return jsonify({"popular": result})


# ✅ Get user ratings history (new)
@app.route('/user/<int:uid>/ratings', methods=['GET'])
def user_ratings(uid):
    if ratings.empty:
        return jsonify({"ratings": []})

    user_data = ratings[ratings['userId'] == uid]
    if user_data.empty:
        return jsonify({"ratings": []})

    # Merge with movies to get titles
    user_data = user_data.merge(movies, on='movieId', how='left')

    results = []
    for _, row in user_data.iterrows():
        results.append({
            "movieId": int(row['movieId']),
            "title": str(row['title']) if pd.notna(row['title']) else f"Movie #{row['movieId']}",
            "rating": float(row['rating']),
            "genres": row.get('genres', '') if pd.notna(row.get('genres')) else ''
        })
    return jsonify({"ratings": results})


# ✅ Search movies (new)
@app.route('/movies/search', methods=['GET'])
def search_movies():
    q = request.args.get('q', '').lower()
    if not q or movies.empty:
        return jsonify({"results": [], "count": 0})

    matches = movies[movies['title'].str.lower().str.contains(q, na=False)].head(20)

    results = []
    for _, row in matches.iterrows():
        results.append({
            "movieId": int(row['movieId']),
            "title": row['title'],
            "genres": row.get('genres', '')
        })
    return jsonify({"results": results, "count": len(results)})


# ✅ Upload new ratings (new)
@app.route('/upload', methods=['POST'])
def upload_dataset():
    global ratings
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and file.filename.endswith('.csv'):
        try:
            new_df = pd.read_csv(file)
            if not all(col in new_df.columns for col in ['userId', 'movieId', 'rating']):
                return jsonify({"error": "Invalid CSV format. Required columns: userId, movieId, rating"}), 400

            ratings = pd.concat([ratings, new_df], ignore_index=True)
            return jsonify({"message": f"Successfully appended {len(new_df)} rows to dataset."})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return jsonify({"error": "Only CSV files allowed"}), 400


# ✅ Final run config (Render compatible)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
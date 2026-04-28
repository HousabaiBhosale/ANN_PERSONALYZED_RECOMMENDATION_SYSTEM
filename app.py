import os
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import pickle

# Use tflite_runtime for extreme memory efficiency
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    # Fallback for local testing if tflite_runtime isn't installed
    try:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter
    except ImportError:
        print("Error: tflite_runtime or tensorflow must be installed")

app = Flask(__name__)

# ✅ Robust Path Handling
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_path(rel_path):
    return os.path.join(BASE_DIR, rel_path)

# ✅ Load TFLite Model
interpreter = Interpreter(model_path=get_path("model.tflite"))
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def tflite_predict(user_ids, movie_ids):
    """Runs prediction using TFLite engine."""
    user_ids = np.array(user_ids, dtype=np.float32).reshape(-1, 1)
    movie_ids = np.array(movie_ids, dtype=np.float32).reshape(-1, 1)
    
    # In many ANN models, input 0 is user, input 1 is movie
    # We must match the input indices from input_details
    # Usually: index 0 = user, index 1 = movie
    
    results = []
    for i in range(len(user_ids)):
        interpreter.set_tensor(input_details[0]['index'], user_ids[i:i+1])
        interpreter.set_tensor(input_details[1]['index'], movie_ids[i:i+1])
        interpreter.invoke()
        pred = interpreter.get_tensor(output_details[0]['index'])
        results.append(pred[0][0])
    return np.array(results)

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

# ✅ Calculate metrics for Home Page
try:
    sample_size = min(len(ratings), 100)
    sample_df = ratings.sample(n=sample_size, random_state=42)
    valid_u, valid_m, valid_a = [], [], []

    for _, row in sample_df.iterrows():
        u_enc = user_encoder.get(row['userId'])
        m_enc = movie_encoder.get(row['movieId'])
        if u_enc is not None and m_enc is not None:
            valid_u.append(u_enc)
            valid_m.append(m_enc)
            valid_a.append(row['rating'])

    if valid_u:
        val_preds = tflite_predict(valid_u, valid_m) * 5.0
        mae_val = float(np.mean(np.abs(val_preds - np.array(valid_a))))
        app_mae = f"{mae_val:.2f}"
        app_accuracy = f"{(1.0 - (mae_val/5.0))*100:.1f}%"
    else:
        app_mae, app_accuracy = "—", "—"
except:
    app_mae, app_accuracy = "—", "—"

# Global Error Handler
@app.errorhandler(Exception)
def handle_exception(e):
    print(f"Server Error: {e}")
    return jsonify({"error": str(e)}), 500

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        user_id = int(data['userId'])
        movie_id_input = data['movieId']

        user_enc = user_encoder.get(user_id)
        if user_enc is None:
            return jsonify({"error": f"User ID {user_id} not found"}), 404

        if isinstance(movie_id_input, str) and ',' in movie_id_input:
            mids = [int(m.strip()) for m in movie_id_input.split(',')]
        elif isinstance(movie_id_input, list):
            mids = [int(m) for m in movie_id_input]
        else:
            mids = [int(movie_id_input)]

        v_mids, v_encs = [], []
        for mid in mids:
            enc = movie_encoder.get(mid)
            if enc is not None:
                v_mids.append(mid)
                v_encs.append(enc)

        results = []
        if v_encs:
            u_list = [user_enc] * len(v_encs)
            batch_preds = tflite_predict(u_list, v_encs)
            
            user_count = len(ratings[ratings['userId'] == user_id]) if not ratings.empty else 0
            for mid, score in zip(v_mids, batch_preds):
                rating = float(score * 5.0)
                conf = min(99.9, 50.0 + (user_count * 0.5) - abs(rating - 3.0)*2.0)
                m_df = movies[movies['movieId'] == mid] if not movies.empty else pd.DataFrame()
                
                results.append({
                    "predicted_rating": rating,
                    "confidence_score": round(conf, 1),
                    "userId": user_id,
                    "movieId": mid,
                    "movie": {"title": m_df.iloc[0]['title'], "genres": m_df.iloc[0].get('genres', '')} if not m_df.empty else None
                })

        invalid = [m for m in mids if m not in v_mids]
        for m in invalid:
            results.append({"movieId": m, "error": "Invalid movie ID", "predicted_rating": 0})

        return jsonify(results[0] if len(results) == 1 else {"predictions": results})
    except Exception as e:
        return handle_exception(e)

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.json
        user_id = int(data['userId'])
        n = int(data.get('n', 10))
        genre = str(data.get('genre', '')).lower()

        if user_id not in user_encoder:
            return jsonify({"error": "User not found"}), 404

        user_enc = user_encoder[user_id]
        watched = ratings[ratings['userId'] == user_id]['movieId'].values if not ratings.empty else []
        unseen = [m for m in all_movie_ids if m not in watched]

        if genre and not movies.empty:
            v_movies = movies[movies['genres'].str.lower().str.contains(genre, na=False)]['movieId'].values
            unseen = [m for m in unseen if m in v_movies]

        unseen = unseen[:100] # Very fast limit
        v_mids, v_encs = [], []
        for mid in unseen:
            enc = movie_encoder.get(mid)
            if enc is not None:
                v_mids.append(mid)
                v_encs.append(enc)

        if not v_encs:
            return jsonify({"recommendations": [], "userId": user_id})

        u_list = [user_enc] * len(v_encs)
        batch_preds = tflite_predict(u_list, v_encs)
        
        preds = sorted(zip(v_mids, batch_preds), key=lambda x: x[1], reverse=True)[:n]
        res = []
        for mid, score in preds:
            m_df = movies[movies['movieId'] == mid]
            res.append({
                "movieId": int(mid),
                "title": m_df.iloc[0]['title'],
                "genres": m_df.iloc[0].get('genres', ''),
                "predicted_rating": float(score * 5.0)
            })
        return jsonify({"recommendations": res, "userId": user_id})
    except Exception as e:
        return handle_exception(e)

@app.route('/rate', methods=['POST'])
def rate():
    try:
        global ratings
        data = request.json
        uid, mid, r = int(data['userId']), int(data['movieId']), float(data['rating'])
        new = pd.DataFrame({"userId": [uid], "movieId": [mid], "rating": [r]})
        ratings = pd.concat([ratings, new], ignore_index=True)
        return jsonify({"message": "Rating added"})
    except Exception as e: return handle_exception(e)

@app.route('/login', methods=['POST'])
def login():
    try:
        uid = int(request.json.get('userId'))
        if uid in user_encoder: return jsonify({"status": "success", "userId": uid})
        return jsonify({"error": "User not found"}), 404
    except: return jsonify({"error": "Invalid format"}), 400

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "mae": app_mae, "accuracy": app_accuracy})

@app.route('/movies/popular', methods=['GET'])
def popular_movies():
    n = request.args.get('n', default=8, type=int)
    if ratings.empty or movies.empty: return jsonify({"popular": []})
    pop = ratings.groupby('movieId').agg({'rating': ['mean', 'count']})
    pop.columns = ['avg', 'cnt']
    p_ids = pop[pop['cnt'] > 5].sort_values(by='avg', ascending=False).head(n).index.tolist()
    res = []
    for mid in p_ids:
        m = movies[movies['movieId'] == mid]
        if not m.empty:
            res.append({"movieId": int(mid), "title": m.iloc[0]['title'], "genres": m.iloc[0].get('genres', ''), "predicted_rating": float(pop.loc[mid, 'avg'])})
    return jsonify({"popular": res})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
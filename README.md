# CineANN — Neural Movie Recommender System

![CineANN Logo](https://img.shields.io/badge/AI-Neural_Network-gold?style=for-the-badge)
![Flask](https://img.shields.io/badge/Backend-Flask-lightgrey?style=for-the-badge)
![TensorFlow](https://img.shields.io/badge/ML-TensorFlow-orange?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**CineANN** is a sophisticated movie recommendation platform powered by an Artificial Neural Network (ANN). It leverages deep neural embeddings to understand user preferences and predict movie ratings with high precision, offering a premium, vintage-inspired cinematic user experience.

---

## 🌟 Key Features

- **🎯 Neural Prediction Engine**: Predicts the exact rating (1-5 stars) a user would give to any movie using deep learning embeddings.
- **🎬 Personalized Recommendations**: Generates a Top-N list of movies tailored to a specific User ID, with optional genre filtering.
- **📈 Real-time Analytics**: Interactive dashboards featuring MAE (Mean Absolute Error) tracking and prediction distribution charts via Chart.js.
- **🔍 Full-text Search**: Easily find movies and their unique IDs from the MovieLens dataset.
- **📥 Dynamic Dataset Management**: Upload new rating CSVs to expand the model's knowledge base in real-time.
- **⭐ Interactive Rating System**: Submit new ratings directly through the UI to update user profiles.
- **🔥 Popularity Tracking**: Automatic calculation of trending films based on global user activity.
- **📱 Premium UI/UX**: A "Lumina Noir" aesthetic featuring smooth animations, canvas-based backgrounds, and a responsive design.

---

## 🛠️ Tech Stack

### Backend
- **Language**: Python 3.x
- **Framework**: Flask
- **WSGI Server**: Gunicorn (Production ready)
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: TensorFlow 2.x, Keras, Scikit-Learn
- **Serialization**: Pickle (for User/Movie Encoders)

### Frontend
- **Structure**: Semantic HTML5
- **Styling**: Vanilla CSS3 (Custom design system with vintage "Sepia" palette)
- **Interactivity**: JavaScript (Async Fetch API)
- **Visualizations**: Chart.js
- **Typography**: Google Fonts (Playfair Display, Bebas Neue, IBM Plex Mono)

---

## 🧠 Deep Dive: How the ANN Works

The core of **CineANN** is a multi-layered Artificial Neural Network (ANN) that performs **Collaborative Filtering** through deep embeddings. Unlike traditional algorithms that use simple similarity scores, this system learns the latent relationships between users and movies.

### 1. The Embedding Layer (Latent Space)
Every `userId` and `movieId` is passed through an **Embedding Layer**. 
- These layers convert categorical IDs into dense, high-dimensional vectors (50 dimensions).
- During training, the network adjusts these vectors so that users with similar tastes (and movies with similar attributes) are positioned close together in this "Latent Space."

### 2. Feature Fusion (Concatenation)
The user vector and movie vector are concatenated into a single 100-dimensional vector. This allows the network to evaluate the specific interaction between *that* user and *that* movie simultaneously.

### 3. Deep Learning (Fully Connected Layers)
The fused vector passes through a series of **Dense (Linear) Layers**:
- **Layer 1 (128 Neurons)**: Extracts complex, non-linear patterns (e.g., "This user likes 90s action but only if it's rated above 3 stars").
- **Layer 2 (64 Neurons)**: Refines these features for the final decision.
- **Activation**: We use **ReLU** (Rectified Linear Unit) to introduce non-linearity, allowing the model to learn complex preferences.

### 4. The Output Layer (Rating Prediction)
The final neuron uses a **Sigmoid Activation Function**, which outputs a value between 0 and 1. We scale this value by **5.0** to provide a final predicted rating (e.g., 4.2 stars).

### 5. Production Optimization: The NumPy Inference Engine
In a typical production environment, loading the heavy TensorFlow framework for every request is slow and memory-intensive. 
- **CineANN** uses a custom-built **Pure NumPy Inference Engine**. 
- We extracted the trained weights from the Keras model and replicated the neural logic using matrix multiplication (`@` operator). 
- This results in **sub-millisecond prediction times** and near-zero memory footprint.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Pip (Python Package Manager)

### Installation

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd "ann project (3)"
   ```

2. **Install dependencies**:
   ```bash
   pip install -r "ann project/requirements.txt"
   ```

3. **Run the Application**:
   ```bash
   python "ann project/app.py"
   ```
   The server will start at `http://127.0.0.1:5000`.

---

## 📂 Project Structure

```text
.
├── ann project/
│   ├── app.py              # Main Flask application & API endpoints
│   ├── model.py            # ANN training & architecture definition
│   ├── model.h5            # Pre-trained Keras model weights
│   ├── requirements.txt    # Project dependencies
│   ├── templates/          # Frontend HTML files
│   ├── dataset/            # MovieLens 100k data (CSV)
│   ├── user_encoder.pkl    # Serialized user ID mapping
│   └── movie_encoder.pkl   # Serialized movie ID mapping
├── Procfile                # Deployment configuration for Heroku/Render
└── README.md               # Project documentation
```

---

## 📊 API Endpoints

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/predict` | `POST` | Predict rating for a User-Movie pair |
| `/recommend` | `POST` | Get Top-N recommendations for a user |
| `/rate` | `POST` | Submit a new rating |
| `/movies/popular`| `GET` | Get trending movies |
| `/movies/search` | `GET` | Search movies by title |
| `/upload` | `POST` | Upload new ratings CSV |
| `/health` | `GET` | Check system and model status |

---

## 📜 License
This project is licensed under the MIT License.

---
*Developed with ❤️ using Artificial Neural Networks.*

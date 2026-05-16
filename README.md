# ANN Personalized Recommendation System 🎬🧠

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)](https://flask.palletsprojects.com/)
[![NumPy](https://img.shields.io/badge/NumPy-Inference-orange.svg)](https://numpy.org/)

A state-of-the-art Movie Recommendation System powered by **Artificial Neural Networks (ANN)**. This project features a unique **Pure NumPy Inference Engine**, allowing deep learning models to run with zero heavy dependencies like TensorFlow or PyTorch in production.

## 🚀 Key Features

- **Personalized Recommendations**: Tailored movie suggestions based on "User Taste DNA".
- **Neural Explainability**: Visualizes *why* a movie was recommended by inspecting neuron activations.
- **Ultra-Fast Inference**: Pure NumPy matrix math implementation for sub-millisecond predictions.
- **Interactive Dashboard**: Modern UI for searching, rating, and discovering movies.
- **Engine Benchmarking**: Real-time comparison of inference speed and memory footprint.

## 🛠️ Tech Stack

- **Backend**: Flask (Python)
- **Frontend**: HTML5, Vanilla CSS, JS (Modern Dashboard)
- **AI/ML Logic**: Artificial Neural Networks (Collaborative Filtering)
- **Optimization**: Pure NumPy Inference Engine

## 📂 Project Structure

```text
├── app.py              # Main Flask Application & NumPy Engine
├── weights.pkl         # Pre-trained ANN Weights
├── movie_encoder.pkl   # Movie ID mapping
├── user_encoder.pkl    # User ID mapping
├── dataset/            # MovieLens datasets (ratings, movies)
├── templates/          # Modern UI (index.html)
├── requirements.txt    # Project dependencies
└── commands.txt        # Quick setup guide
```

## ⚙️ Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/HousabaiBhosale/ANN_PERSONALYZED_RECOMMENDATION_SYSTEM.git
   cd ANN_PERSONALYZED_RECOMMENDATION_SYSTEM
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   python app.py
   ```

4. **Access the Dashboard**:
   Open `http://localhost:5000` in your browser.

## 🧠 How it Works

The system uses **Embeddings** to map Users and Movies into a 50-dimensional latent space. An ANN then learns the complex interactions between these vectors to predict user preferences. 

By extracting the trained weights into `weights.pkl`, we perform the forward pass using:
`Output = Sigmoid(Dense2(ReLU(Dense1(Concatenate(User_Emb, Movie_Emb)))))`
completely within NumPy!

## 📄 License
This project is for educational purposes and uses the MovieLens dataset.

---
Developed with ❤️ by [HousabaiBhosale](https://github.com/HousabaiBhosale)

# CineANN — Neural Movie Recommender System ⚡
## *The Cinema of Neural Intelligence*

![CineANN Logo](https://img.shields.io/badge/AI-Neural_Network-gold?style=for-the-badge)
![Flask](https://img.shields.io/badge/Backend-Flask-lightgrey?style=for-the-badge)
![NumPy](https://img.shields.io/badge/Engine-Pure_NumPy-blue?style=for-the-badge)
![XAI](https://img.shields.io/badge/XAI-Neural_X--Ray-red?style=for-the-badge)

**CineANN** is a high-performance recommendation engine that goes beyond "black box" AI. It provides real-time **Neural Explainability (XAI)**, revealing the internal logic of its custom-built **NumPy Neural Inference Engine**.

---

## 💎 Advanced Neural Innovations (v2.0)

CineANN features high-impact "Neural Introspection" tools for deep model transparency:

### 🔬 1. Neural X-Ray (Model Introspection)
Deeply analyze the ANN's decision-making process for any specific movie prediction:
- **Embedding Heatmaps**: Visualizes the 50-dimensional vectors for both User and Movie.
- **Neuron Activation Flow**: Tracks the percentage of neurons that "fired" (ReLU > 0) through every Dense layer.
- **Top Latent Dimensions**: Mathematically identifies which specific hidden features drove the final rating.
- **Cosine Similarity**: Measures raw vector alignment in the 50D latent space.

### 🧬 2. Learned User Profile (Taste DNA)
The engine doesn't just recommend movies; it learns a **Neural Fingerprint** for every user:
- **Taste DNA Fingerprint**: A unique 50-dimensional visual barcode learned by the model.
- **Genre Affinity Radar**: Real-time calculation of user preferences across genres, derived from the user's interaction history and embedding weights.

### 🚀 3. Live Engineering Benchmarks
CineANN demonstrates engineering superiority by outperforming standard deep learning frameworks:
- **NumPy Engine vs. TensorFlow**: Side-by-side comparison showing a **~2000x speedup** in inference.
- **Layer-by-Layer Timing**: Micro-second precision timing for Embedding Lookups, ReLU activations, and Sigmoid outputs.
- **0.6MB Footprint**: Shows how a 1.2GB TensorFlow model was compressed into a 610KB high-performance NumPy link.

---

## 🌟 Core Functionality

- **🎯 Neural Prediction**: Predicts exact ratings (1-5 stars) using deep neural embeddings.
- **🎬 Smart Recommendations**: Top-N personalized picks with real-time genre filtering.
- **🔍 Contrast Discovery**: Identifies hidden "novel patterns" when raw embedding similarity is low but neural layers predict high interest.
- **📈 Real-time Analytics**: Interactive MAE (Mean Absolute Error) monitoring and prediction distribution graphs.
- **📱 Lumina Noir UX**: A premium, vintage-inspired aesthetic with glassmorphism and smooth micro-animations.

---

## 🧠 Deep Dive: The Neural Architecture

The core of **CineANN** is a multi-layered Artificial Neural Network (ANN) that performs **Collaborative Filtering** through deep embeddings.

### 1. The Embedding Layer (The Fingerprint)
Every `userId` and `movieId` is mapped to a **50-dimensional vector**. During training, the network adjusts these vectors so that users with similar tastes (and movies with similar attributes) are positioned close together in the "Latent Space."

### 2. Deep Learning (The Reasoning)
- **Fused Vector (100D)**: Concatenates user and movie features.
- **Dense Layer 1 (128 Neurons, ReLU)**: Captures complex, non-linear interactions between user and movie attributes.
- **Dense Layer 2 (64 Neurons, ReLU)**: Refines the signal for the final decision.
- **Output Layer (Sigmoid)**: Produces the final probability score, scaled to a 5.0 star rating.

### 3. Pure NumPy Inference (The Speed)
Standard projects require the entire 1.2GB TensorFlow library to run. CineANN uses a custom **Pure NumPy Inference Engine**:
- Extracted trained weights from Keras.
- Replicated the neural math using high-speed matrix multiplication (`@`).
- **Result**: Sub-millisecond predictions with zero framework overhead.

---

## 📊 API & Engine Stats

| Endpoint | Description | New in v2.0 |
| :--- | :--- | :---: |
| `/explain` | Full neural introspection data for a user-movie pair | ✅ |
| `/benchmark` | Live comparison of NumPy vs TensorFlow performance | ✅ |
| `/user/similarity` | Compare user tastes based on embedding distance | ✅ |
| `/predict` | Predict rating with neural match % and real reasoning | ✅ |
| `/recommend` | Personalized picks with Taste DNA & Genre Radar | ✅ |

---

## 🛠️ Installation & Setup

1. **Clone & Enter**:
   ```bash
   git clone <your-repo-url>
   cd "ann project (3)"
   ```

2. **Install Weights & Data**:
   Ensure `weights.pkl` and the `dataset/` folder are in the root directory.

3. **Install Dependencies**:
   ```bash
   pip install flask numpy pandas scikit-learn chart.js
   ```

4. **Launch**:
   ```bash
   python app.py
   ```
   Open `http://127.0.0.1:5000` to enter the neural cinema.

---

## 📜 License
This project is licensed under the MIT License.

*Coded with ❤️ using Artificial Neural Networks and Pure NumPy.*

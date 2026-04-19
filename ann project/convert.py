import pandas as pd
import os

# Create dataset folder if not exists
os.makedirs("dataset", exist_ok=True)

# Load ratings
ratings = pd.read_csv("ml-100k/u.data", sep="\t",
                      names=["userId", "movieId", "rating", "timestamp"])

ratings = ratings.drop("timestamp", axis=1)

# Load movies
movies = pd.read_csv("ml-100k/u.item", sep="|", encoding="latin-1", header=None)
movies = movies[[0, 1]]
movies.columns = ["movieId", "title"]

# Save CSV files
ratings.to_csv("dataset/ratings.csv", index=False)
movies.to_csv("dataset/movies.csv", index=False)

print("✅ Dataset converted successfully!")
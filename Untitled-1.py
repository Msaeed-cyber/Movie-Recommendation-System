"""
🎬 AI Intern – Task 5: Movie Recommendation System (User-Based Filtering)
📂 Dataset: MovieLens 100k
📌 Libraries: pandas, sklearn, numpy, seaborn, matplotlib
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt

# 1️⃣ Load datasets using absolute paths
ratings = pd.read_csv("C:/Users/lenovo/Desktop/New folder/ml-100k/u.data", sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
movies = pd.read_csv("C:/Users/lenovo/Desktop/New folder/ml-100k/u.item", sep='|', encoding='latin-1', names=[
    'movie_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL',
    'unknown', 'Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy',
    'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
    'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
])

# 2️⃣ Merge ratings and movie titles
movie_data = pd.merge(ratings, movies[['movie_id', 'title']], left_on='item_id', right_on='movie_id')

# 3️⃣ Create user-item matrix
user_item_matrix = movie_data.pivot_table(index='user_id', columns='title', values='rating').fillna(0)

# 4️⃣ Build the NearestNeighbors model using cosine similarity
model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
model_knn.fit(user_item_matrix)

# 5️⃣ Recommendation function
def recommend_movies_for_user(user_id, user_item_matrix, model, k=3):
    user_index = user_id - 1  # Adjust for zero-based index
    distances, indices = model.kneighbors([user_item_matrix.iloc[user_index]], n_neighbors=k + 1)

    neighbor_indices = indices.flatten()[1:]  # exclude self
    similar_users = user_item_matrix.iloc[neighbor_indices]
    mean_ratings = similar_users.mean(axis=0)

    user_ratings = user_item_matrix.iloc[user_index]
    unseen_movies = user_ratings[user_ratings == 0].index
    recommendations = mean_ratings[unseen_movies].sort_values(ascending=False).head(k)
    
    return recommendations

# 6️⃣ Take user input and show recommendations
try:
    user_input = int(input("🎯 Enter a User ID (1–943): "))
    if 1 <= user_input <= 943:
        print(f"\n🎥 Top 3 Recommendations for User {user_input}:\n")
        recs = recommend_movies_for_user(user_input, user_item_matrix, model_knn, k=3)
        for movie, score in recs.items():
            print(f"✅ {movie}  (Predicted Rating: {score:.2f})")
    else:
        print("❌ User ID must be between 1 and 943.")
except ValueError:
    print("❌ Invalid input. Please enter a numeric user ID.")

# 7️⃣ Visualize the similarity matrix for first 20 users
print("\n📊 Showing User Similarity Heatmap (first 20 users)...")

sim_matrix = cosine_similarity(user_item_matrix[:20])
plt.figure(figsize=(10, 8))
sns.heatmap(sim_matrix, xticklabels=range(1, 21), yticklabels=range(1, 21),
            cmap="YlGnBu", annot=True, fmt=".2f", linewidths=0.5)
plt.title("User Similarity Matrix (Top 20 Users)", fontsize=14)
plt.xlabel("User ID")
plt.ylabel("User ID")
plt.tight_layout()
plt.show()

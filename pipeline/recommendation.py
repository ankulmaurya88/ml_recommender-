import os
import numpy as np
import pandas as pd

MODEL_DIR = "models/"
SIMILARITY_MATRIX_PATH = os.path.join(MODEL_DIR, "similarity_matrix.npy")
RESTAURANT_INDEX_PATH = os.path.join(MODEL_DIR, "restaurant_index.csv")
RESTAURANT_RATINGS_PATH = "/home/arvind/ml_recommender/data/raw/processed_zomato.csv"  # Expected columns: Restaurant Name, Aggregate rating

def recommend_restaurants(restaurant_name, top_n=5):
    try:
        if not os.path.exists(SIMILARITY_MATRIX_PATH):
            raise FileNotFoundError(f"Similarity matrix not found at {SIMILARITY_MATRIX_PATH}")
        if not os.path.exists(RESTAURANT_INDEX_PATH):
            raise FileNotFoundError(f"Restaurant index not found at {RESTAURANT_INDEX_PATH}")
        if not os.path.exists(RESTAURANT_RATINGS_PATH):
            raise FileNotFoundError(f"Restaurant ratings file not found at {RESTAURANT_RATINGS_PATH}")

        # Load data
        similarity_matrix = np.load(SIMILARITY_MATRIX_PATH)
        restaurant_index = pd.read_csv(RESTAURANT_INDEX_PATH)
        ratings_df = pd.read_csv(RESTAURANT_RATINGS_PATH)

        # Normalize input name for case-insensitive match
        restaurant_name = restaurant_name.lower()
        restaurant_index['restaurant_name_lower'] = restaurant_index['restaurant_name'].str.lower()

        if restaurant_name not in restaurant_index['restaurant_name_lower'].values:
            print(f"Restaurant '{restaurant_name}' not found.")
            return []

        idx = restaurant_index[restaurant_index['restaurant_name_lower'] == restaurant_name].index[0]
        similarities = similarity_matrix[idx]

        # Top similar indices excluding itself
        similar_indices = similarities.argsort()[::-1][1:top_n+1]

        # Create a mapping for quick rating lookup
        ratings_df['Restaurant Name'] = ratings_df['Restaurant Name'].astype(str)
        rating_map = dict(zip(ratings_df['Restaurant Name'], ratings_df['Aggregate rating']))

        recommendations = []
        for i in similar_indices:
            name = restaurant_index.iloc[i]['restaurant_name']
            rating = rating_map.get(name, 0)
            recommendations.append({
                'restaurant_name': name,
                'similarity': similarities[i],
                'rating': rating
            })

        # Sort recommendations by rating (descending)
        recommendations.sort(key=lambda x: x['rating'], reverse=True)
        return recommendations

    except FileNotFoundError as fnf_error:
        print(fnf_error)
        return []
    except Exception as e:
        print(f"An error occurred during recommendation: {e}")
        return []

if __name__ == "__main__":
    res_name = input("Enter restaurant name: ").strip()
    recs = recommend_restaurants(res_name)
    if recs:
        print(f"\nTop recommendations similar to '{res_name}':")
        for rec in recs:
            print(f"{rec['restaurant_name']} - Rating: {rec['rating']:.1f}, Similarity: {rec['similarity']:.4f}")
    else:
        print("No recommendations found.")

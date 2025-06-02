import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os
import scipy.sparse

CLEANED_DATA_PATH = "/home/arvind/ml_recommender/data/processed/cleaned_zomato.csv"  # updated to your cleaned data
MODEL_DIR = "models/"
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
TFIDF_MATRIX_PATH = os.path.join(MODEL_DIR, "tfidf_matrix.npz")

def build_tfidf_model():
    df = pd.read_csv(CLEANED_DATA_PATH)

    # Combine relevant text columns for vectorization, e.g., cuisines + locality
    df['combined_text'] = df['cleaned_cuisines'].fillna('') + ' ' + df['cleaned_locality'].fillna('')

    # Group by restaurant name to combine text per restaurant
    grouped = df.groupby('cleaned_restaurant_name')['combined_text'].apply(lambda x: ' '.join(x)).reset_index()

    vectorizer = TfidfVectorizer(max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(grouped['combined_text'])

    os.makedirs(MODEL_DIR, exist_ok=True)
    # Save vectorizer
    with open(VECTORIZER_PATH, 'wb') as f:
        pickle.dump(vectorizer, f)
    # Save tfidf matrix
    scipy.sparse.save_npz(TFIDF_MATRIX_PATH, tfidf_matrix)

    # Save restaurant name index for lookup
    grouped['cleaned_restaurant_name'].to_csv(os.path.join(MODEL_DIR, "restaurant_index.csv"), index=False)

    print(f"Saved vectorizer and TF-IDF matrix to {MODEL_DIR}")

if __name__ == "__main__":
    build_tfidf_model()

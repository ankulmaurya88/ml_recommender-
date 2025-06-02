import os
import pickle
import scipy.sparse
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

MODEL_DIR = "models/"
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
TFIDF_MATRIX_PATH = os.path.join(MODEL_DIR, "tfidf_matrix.npz")
RESTAURANT_INDEX_PATH = os.path.join(MODEL_DIR, "restaurant_index.csv")
SIMILARITY_MATRIX_PATH = os.path.join(MODEL_DIR, "similarity_matrix.npy")

def build_similarity_matrix():
    try:
        if not os.path.exists(TFIDF_MATRIX_PATH):
            raise FileNotFoundError(f"TF-IDF matrix not found at {TFIDF_MATRIX_PATH}")

        print("Loading TF-IDF matrix...")
        tfidf_matrix = scipy.sparse.load_npz(TFIDF_MATRIX_PATH)

        print("Calculating cosine similarity matrix...")
        similarity_matrix = cosine_similarity(tfidf_matrix)

        os.makedirs(MODEL_DIR, exist_ok=True)
        np.save(SIMILARITY_MATRIX_PATH, similarity_matrix)
        print(f"Saved similarity matrix at {SIMILARITY_MATRIX_PATH}")

    except FileNotFoundError as fnf_error:
        print(fnf_error)
    except Exception as e:
        print(f"An error occurred while building similarity matrix: {e}")

if __name__ == "__main__":
    build_similarity_matrix()


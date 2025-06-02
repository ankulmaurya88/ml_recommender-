from pipeline import ingestion, preprocessing, vectorization, similarity, recommendation

def run_full_pipeline():
    print("Starting data ingestion...")
    ingestion.load_raw_data()

    print("Starting preprocessing...")
    preprocessing.preprocess_text_columns()

    print("Starting vectorization...")
    vectorization.build_tfidf_model()

    print("Calculating similarity matrix...")
    similarity.build_similarity_matrix()

    print("Pipeline finished successfully!")

if __name__ == "__main__":
    run_full_pipeline()
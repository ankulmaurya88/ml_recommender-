import os
import pandas as pd
import subprocess

RAW_DATA_DIR = "data/raw"
RAW_DATA_PATH = os.path.join(RAW_DATA_DIR, "zomato.csv")
PROCESSED_DATA_PATH = "data/processed/clean_reviews.csv"

def download_kaggle_dataset():
    dataset_name = "harishkumardatalab/global-zomato-dataset"
    os.makedirs(RAW_DATA_DIR, exist_ok=True)

    try:
        print("Starting dataset download from Kaggle...")
        # Using subprocess.run for better control and error capture
        result = subprocess.run(
            ["kaggle", "datasets", "download", "-d", dataset_name, "-p", RAW_DATA_DIR, "--unzip"],
            capture_output=True,
            text=True,
            check=True,
        )
        print("Download completed successfully.")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error during download: {e.stderr}")
        raise RuntimeError("Failed to download Kaggle dataset. Make sure kaggle.json is configured.") from e
    

# download_kaggle_dataset()


def load_raw_data():
    try:
        file_path = "/home/arvind/ml_recommender/notebooks/data/raw/zomato.csv"
        print(f"Loading raw data from {file_path} ......")
        reviews = pd.read_csv(file_path, encoding='latin1')
        print(f"Loaded raw data with shape: {reviews.shape}")

        PROCESSED_DATA_PATH = '/home/arvind/ml_recommender/data/raw/processed_zomato.csv'
        os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
        reviews.to_csv(PROCESSED_DATA_PATH, index=False)
        print(f"Saved raw data copy to {PROCESSED_DATA_PATH}")
        
    except FileNotFoundError as e:
        print(f"File not found: {e.filename}")
        raise RuntimeError("Raw dataset file missing. Download might have failed.") from e
    except pd.errors.ParserError as e:
        print(f"Pandas parsing error: {str(e)}")
        raise RuntimeError("Error parsing the CSV file.") from e
    

if __name__ == "__main__":
    try:
        download_kaggle_dataset()
        load_raw_data()
        print("Dataset is ready for further processing.")
    except Exception as err:
        print(f"Pipeline failed: {err}")




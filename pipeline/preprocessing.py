import pandas as pd
import os
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure necessary NLTK resources are downloaded (only once)
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

try:
    lemmatizer = WordNetLemmatizer()
except LookupError:
    nltk.download('wordnet')
    lemmatizer = WordNetLemmatizer()

INPUT_PATH = "/home/arvind/ml_recommender/data/raw/processed_zomato.csv"
OUTPUT_PATH = "/home/arvind/ml_recommender/data/processed/cleaned_zomato.csv"

def clean_text(text):
    if pd.isnull(text):
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove digits
    text = re.sub(r'\d+', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenize and remove stopwords
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words]
    
    # Lemmatize tokens
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    
    return ' '.join(tokens)

def preprocess_text_columns():
    try:
        print(f"Loading input data from {INPUT_PATH} ...")
        df = pd.read_csv(INPUT_PATH)

        # Columns to clean
        text_columns = ['Restaurant Name', 'Cuisines', 'City', 'Locality']

        for col in text_columns:
            if col in df.columns:
                print(f"Cleaning column: {col}")
                cleaned_col_name = f'cleaned_{col.lower().replace(" ", "_")}'
                df[cleaned_col_name] = df[col].astype(str).apply(clean_text)
            else:
                print(f"Warning: Column '{col}' not found in data.")

        # Create output directory if not exists
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        
        # Save cleaned dataframe
        df.to_csv(OUTPUT_PATH, index=False)
        print(f"Saved cleaned data to {OUTPUT_PATH}")

    except Exception as e:
        print(f"Error during preprocessing: {e}")
        raise

if __name__ == "__main__":
    preprocess_text_columns()

# 🍽️ Zomato Content-Based Restaurant Recommendation System

This project builds a **Content-Based Recommender System** for restaurants using Zomato restaurant data. Given a restaurant name, the system recommends other restaurants with **similar reviews, cuisines, and metadata**, sorted by **aggregate rating**.

---

## 📌 Project Structure
```bash
ml_recommender/
│
├── data/
│ ├── raw/
│ │ └── processed_zomato.csv # Original input dataset
│ └── processed/
│ └── cleaned_zomato.csv # Output of text preprocessing
│
├── ingestion.py # Loads raw data into processed format
├── preprocessing.py # Cleans and lemmatizes text columns (e.g., cuisines, city)
├── vectorization.py # TF-IDF + normalized numeric features
├── similarity.py # Computes cosine similarity matrix
├── recommendation.py # Gets top-N similar restaurants
├── evaluation.py # Placeholder for evaluation logic
└── README.md

```

---

## 🔍 Features Used

The recommender is built using the following features:

- **Textual**: `Cuisines`, `Restaurant Name`, `City`, `Locality`
- **Numerical**: `Average Cost for two`, `Aggregate rating`, `Votes`
- **Boolean**: `Has Online delivery`, `Has Table booking`, `Is delivering now`

These features are cleaned, encoded, and combined into a unified vector space using **TF-IDF** and **MinMax Scaling**, followed by **cosine similarity**.

---

## ⚙️ How to Run

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/zomato-recommender.git
cd zomato-recommender/ml_recommender
2. Install dependencies
Make sure to activate your Python environment:
```
`` bash

pip install -r requirements.txt
```
---
Dependencies include:

pandas

scikit-learn

nltk

numpy

scipy

Note: You may need to download NLTK corpora manually (done in code).
---

# 3. Run the pipeline
``` bash

python ingestion.py            # Load and save raw data
python preprocessing.py        # Clean text features
python vectorization.py        # Generate feature matrix
python similarity.py           # Compute similarity matrix
python recommendation.py       # Recommend restaurants
```
# Example of  Output
---
Restaurant: "Domino's Pizza"

Top 5 Similar Restaurants:
- Pizza Hut – Italian, Fast Food – Rating: 4.1
- Oven Story – Pizza, Delivery – Rating: 4.2
- Mojo Pizza – Thin Crust, Affordable – Rating: 4.0
📁 Sample Input Data (processed_zomato.csv)
Restaurant Name	Cuisines	City	Avg Cost	Rating	Votes
Domino's Pizza	Fast Food	Delhi	400	4.1	2500
Pizza Hut	Italian	Delhi	450	4.2	3200
---
---
📌 Future Improvements
Add user reviews for better textual matching

Implement hybrid recommendation (collaborative + content)
---
# Deploy via  Flask

🧑‍💻 Author
Ankul maurya – @ankulmaurya88



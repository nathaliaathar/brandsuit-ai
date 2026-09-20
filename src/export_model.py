# Retrain the category classifier and write joblib files for Streamlit.
# The app must only transform + predict. Run this when mapping or hyperparameters change.

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "models"

CATEGORY_MAPPING = {
    "Autos & Vehicles": "Lifestyle & Interests",
    "Comedy": "Entertainment & Culture",
    "Education": "Education, Science & Technology",
    "Entertainment": "Entertainment & Culture",
    "Film & Animation": "Entertainment & Culture",
    "Gaming": "Gaming",
    "Howto & Style": "Lifestyle & Interests",
    "Music": "Entertainment & Culture",
    "News & Politics": "News, Politics & Society",
    "Nonprofits & Activism": "News, Politics & Society",
    "People & Blogs": "Lifestyle & Interests",
    "Pets & Animals": "Education, Science & Technology",
    "Science & Technology": "Education, Science & Technology",
    "Shows": "Entertainment & Culture",
    "Sports": "Sports",
    "Travel & Events": "Lifestyle & Interests",
}


def main():
    payload = json.loads((RAW / "US_category_id.json").read_text(encoding="utf-8"))
    id_to_name = {int(item["id"]): item["snippet"]["title"] for item in payload["items"]}

    df = pd.read_csv(RAW / "USvideos.csv")
    df_one = df.drop_duplicates(subset="video_id").copy()
    df_one["youtube_name"] = df_one["category_id"].map(id_to_name)
    df_one["y"] = df_one["youtube_name"].map(CATEGORY_MAPPING).fillna("Other")
    df_one["text"] = (
        df_one["title"].fillna("") + " " + df_one["description"].fillna("")
    ).str.strip()

    X = df_one["text"]
    y = df_one["y"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Same as notebook D4: bigrams, fit on train only
    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)
    X_train_vec = vec.fit_transform(X_train)
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train_vec, y_train)

    OUT.mkdir(exist_ok=True)
    joblib.dump(vec, OUT / "category_tfidf.joblib")
    joblib.dump(model, OUT / "category_logreg.joblib")
    print("saved", OUT / "category_tfidf.joblib")
    print("saved", OUT / "category_logreg.joblib")
    print("train rows:", len(X_train), "test rows:", len(X_test))


if __name__ == "__main__":
    main()

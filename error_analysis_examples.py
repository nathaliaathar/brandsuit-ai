"""Error analysis: News videos misclassified as Entertainment, before vs after the fix.

Reads only the local training CSV, prints a few holdout examples so the Streamlit
story tab can quote real titles. Writes nothing.
"""
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.export_model import CATEGORY_MAPPING  # noqa: E402

NEWS = "News, Politics & Society"
ENT = "Entertainment & Culture"

payload = json.loads((ROOT / "data/raw/US_category_id.json").read_text(encoding="utf-8"))
id_to_name = {int(i["id"]): i["snippet"]["title"] for i in payload["items"]}

df = pd.read_csv(ROOT / "data/raw/USvideos.csv").drop_duplicates(subset="video_id")
df["y"] = df["category_id"].map(id_to_name).map(CATEGORY_MAPPING).fillna("Other")
df["text"] = (df["title"].fillna("") + " " + df["description"].fillna("")).str.strip()

train_idx, test_idx = train_test_split(
    df.index, test_size=0.2, random_state=42, stratify=df["y"]
)
train, test = df.loc[train_idx], df.loc[test_idx]


def fit(ngram, weight):
    vec = TfidfVectorizer(max_features=5000, ngram_range=ngram, min_df=2)
    Xtr = vec.fit_transform(train["text"])
    model = LogisticRegression(max_iter=1000, class_weight=weight)
    model.fit(Xtr, train["y"])
    return pd.DataFrame(
        model.predict_proba(vec.transform(test["text"])),
        columns=model.classes_,
        index=test.index,
    )


before = fit((1, 1), None)
after = fit((1, 2), "balanced")

pred_before = before.idxmax(axis=1)
pred_after = after.idxmax(axis=1)

is_news = test["y"] == NEWS
missed = test[is_news & (pred_before == ENT)]
print("News videos called Entertainment by the FIRST model:", len(missed))
print("of", int(is_news.sum()), "News videos in the holdout\n")

for idx, row in missed.iterrows():
    title = row["title"].encode("ascii", "replace").decode("ascii")
    pa_b = before.loc[idx, ENT] + before.loc[idx, "Sports"]
    pa_a = after.loc[idx, ENT] + after.loc[idx, "Sports"]
    print(f"- {title[:78]}")
    print(f"    before: pred={pred_before[idx][:28]:28} p_allow={pa_b:.2f} -> "
          f"{'APPROVED' if pa_b >= 0.55 else 'blocked'}")
    print(f"    after : pred={pred_after[idx][:28]:28} p_allow={pa_a:.2f} -> "
          f"{'APPROVED' if pa_a >= 0.55 else 'blocked'}")
    print(f"    P(News) {before.loc[idx, NEWS]:.2f} -> {after.loc[idx, NEWS]:.2f}")

pa_before = before[ENT] + before["Sports"]
pa_after = after[ENT] + after["Sports"]
print("\nAll News videos, approved at 0.55:")
print("  before:", int((is_news & (pa_before >= 0.55)).sum()))
print("  after :", int((is_news & (pa_after >= 0.55)).sum()))
print(
    "News caught at argmax: before",
    int((is_news & (pred_before == NEWS)).sum()),
    "after",
    int((is_news & (pred_after == NEWS)).sum()),
    "of",
    int(is_news.sum()),
)

# BrandSuit AI

Should this brand advertise next to this YouTube video?

That is the only question the app answers. It is **not** a violence detector and it does not score “safe vs unsafe”. YouTube Trending has no those labels, so I did not invent them.

The model reads the **title + description**, predicts one of six content categories, and then a **brand policy** (allow-list + confidence threshold) says YES or NO.

Live demo: `streamlit run app.py`

---

## Why this exists

I started from a brief that asked for multimodal brand *safety* (text + image, including Violence / Sensitive). The public dataset I could actually use — [YouTube Trending, US](https://www.kaggle.com/datasets/datasnaek/youtube-new) — is already filtered and has **category IDs only**.

So I changed the product:

| Brief said | What shipped |
|---|---|
| Brand safety / unsafe content | Contextual **suitability** |
| Image + text | Text only (title + description) |
| Violence / Sensitive as a class | Dropped — no labels |
| One YES/NO from the model | Category model **plus** an advertiser policy |

What I chose and what I rejected is in `DECISIONS.md`.

---

## How a decision is made

```
title + description
        │
        ▼
  TF-IDF (unigrams + bigrams, 5,000 features)
  fitted on train only
        │
        ▼
  Logistic Regression (class_weight="balanced")
  → 6 probabilities that sum to 1
        │
        ▼
  p_allow = sum of P(categories this advertiser allows)
        │
        ▼
  YES if p_allow ≥ that advertiser’s threshold
```

The classifier does not know who the advertiser is. Switching the profile can flip YES to NO on the **same** scores.

Four profiles in `src/policies/brand_policies.py`:

| Profile | Allows | Threshold | Where the threshold came from |
|---|---|---|---|
| Kids & Family Brand | Sports, Entertainment | 0.55 | Holdout experiment (News approvals 6 → 0) |
| Financial Services | Sports, Ent, Lifestyle, Education | 0.60 | Starting point, not validated |
| Sports & Apparel | Sports, Gaming, Ent, Lifestyle | 0.50 | Starting point, not validated |
| Gaming Publisher | Gaming, Entertainment | 0.45 | Starting point, not validated |

Only 0.55 is a validation candidate. The other three are honest placeholders.

---

## Data

- Source: Kaggle `datasnaek/youtube-new` (US files).
- 40,949 trending-day rows → **6,351 unique `video_id`**. Same video on eight days would leak train into test if I skipped this.
- Category names come from `US_category_id.json`, then bucketed into 6 classes.

| Class | Share of unique videos |
|---|---|
| Entertainment & Culture | 51.8% |
| Lifestyle & Interests | 19.3% |
| Education, Science & Technology | 12.1% |
| News, Politics & Society | 8.2% |
| Sports | 7.1% |
| Gaming | 1.6% |

Entertainment is more than half the data. A dummy that always says Entertainment already gets **51.8% accuracy** and **0.114 macro-F1**. That is why I do not sell the model on accuracy.

Split: 80/20 stratified, `random_state=42` → 5,080 train / 1,271 development holdout.

I used that holdout to compare models, read errors, and pick 0.55. It is **not** a sealed final test. I would freeze the pipeline and score a leftover set before calling this production.

`Pets & Animals` was mapped into Education, Science & Technology (my choice). That is why pet words show up in that class’s coefficients.

---

## What I actually measured

On the development holdout, after the Pets remap:

| Setup | Accuracy | Macro-F1 | News recall | Kids & Family: News approved @ 0.55 |
|---|---|---|---|---|
| Always Entertainment | 51.8% | 0.114 | 0.0% | — |
| Unigram, no class weights | 78.0% | 0.678 | 75.0% | 6 |
| Unigram, balanced | 80.0% | 0.782 | 90.0% | 0 |
| **Bigram, balanced + policy (deployed)** | **78.8%** | **0.759** | **88.0%** | **0** |

Accuracy barely moved when I added `class_weight="balanced"`. Macro-F1 and News/Gaming recall did. That was the point.

The first model called 24 of 104 real News videos “Entertainment”. Six of those cleared `p_allow ≥ 0.55` (a retiring congressman, a submarine search, a soldier’s remains…). After bigrams + balanced weights + the 0.55 rule, those six fall below the line. Cost for Kids & Family: 258 of 658 true Entertainment videos also get withheld.

The lab notebook is `notebooks/01_inspect_youtube_trending.ipynb`. `error_analysis_examples.py` reprints those six titles.

---

## Repo layout

```
brandsuit-ai/
├── app.py                      # Streamlit: Live Decision + Project Story
├── src/export_model.py         # Retrain TF-IDF + LogReg, save joblib
├── src/policies/brand_policies.py
├── notebooks/01_inspect_youtube_trending.ipynb
├── error_analysis_examples.py  # Holdout News leaks, before vs after
├── DECISIONS.md                # What I chose and what I rejected
├── data/raw/                   # JSON mapping committed; CSV is not
└── models/                     # .gitkeep only — run export_model.py
```

`app.py` never calls `.fit`. It loads the saved vectorizer and model, runs `transform` + `predict_proba`, then `evaluate_policy`.

---

## Run it locally

Python 3.11+ (I used 3.13). From this folder:

```bash
pip install -r requirements.txt
```

1. Download `USvideos.csv` from [Kaggle: YouTube Trending](https://www.kaggle.com/datasets/datasnaek/youtube-new) and put it in `data/raw/` next to `US_category_id.json`.
2. Train and save the artifacts:

```bash
python src/export_model.py
```

3. Open the app:

```bash
streamlit run app.py
```

Try the default Infowars example under **Kids & Family Brand** (should be NO), then switch to **Financial Services** without changing the text. Same probabilities, different policy.

---

## What I would not confirm:

- That this detects hate, violence, or brand safety. It does not.
- That 0.55 is proven on a fresh test set. It is not.
- That Gaming recall is stable. The holdout has 20 Gaming videos.
- That YouTube’s Sports label is always Sports. I have seen opera and an Oprah speech tagged that way.

Next, if I continued: freeze this pipeline, score an untouched test set, get more Gaming/Sports text, clean noisy labels, and only then think about transcripts or thumbnails.

---

## License / data

Code is mine for portfolio use. The trending table belongs to the Kaggle dataset authors / YouTube. I do not redistribute `USvideos.csv`.

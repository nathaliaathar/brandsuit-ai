# BrandSuit AI

Text-only YouTube category classification, then advertiser-specific suitability policies.

**Live demo:** [https://brandsuit.streamlit.app/](https://brandsuit.streamlit.app/)  
**Run locally:** `py -m streamlit run app.py`

BrandSuit predicts one of six contextual categories from a video title and description, then applies a brand allow-list and a threshold. It is **contextual suitability**, not brand safety. The YouTube Trending scrape has no violence, hate, or unsafe-content labels.

Empty text and text with no recognized vocabulary return **Insufficient information** instead of a YES driven by the model intercept. The user-facing figure is an **allowed-category score** (sum of model outputs on that profile's allow-list), not a verified probability of safety.

## Project summary

- **Unit:** one unique `video_id` (40,949 trending-day rows → 6,351 videos).
- **Features:** `title + description`. `category_id` and channel name never go into X.
- **Labels:** official YouTube categories grouped for advertiser context. Pets & Animals stay in Lifestyle & Interests because a pet vlog is lifestyle inventory, not STEM. That choice is documented in `src/config.py` and is **not** the mapping that maximised a metric.
- **Model selected:** unigram TF-IDF + logistic regression with `class_weight="balanced"`.
- **Why not bigrams:** on the same development holdout, balanced bigrams cut Kids & Family excluded-category approvals by **one video** (14 vs 15) and retained less eligible inventory. That gap is inside a 3-video noise band, so the simpler unigram model was kept.
- **Policy finding:** Kids & Family News approvals at threshold 0.55 are **0/104**. Total approvals from **all** excluded categories are **15/523** (11 Lifestyle, 4 Gaming, 0 Education, 0 News). The older “Blocked-category approvals: 0” figure only counted News.

All reported numbers come from `reports/evaluation.json`, written by `py -m src.evaluate`. The app story tab reads that file. This README does not keep a second copy of the metrics by hand.

## Dataset

[Kaggle datasnaek/youtube-new](https://www.kaggle.com/datasets/datasnaek/youtube-new), US files only. License and download notes: `data/raw/SOURCE.txt`.

Place `USvideos.csv` at `data/raw/USvideos.csv` (gitignored, ~60 MB). `US_category_id.json` is already in the repo.

## Reproduce

From this repository root:

```text
py -m pip install -r requirements.txt
py -m src.evaluate
py -m pytest tests
py -m streamlit run app.py
```

`py -m src.evaluate` fits four specs on the same 80/20 stratified split (`random_state=42`), writes:

- `reports/evaluation.json` — configs, dataset fingerprint, metrics, policy counts, calibration, robustness, error examples
- `reports/split_ids.json` — train and holdout `video_id` lists
- `models/category_tfidf.joblib`, `models/category_logreg.joblib`, `models/manifest.json` — the **selected** pipeline, fit on train only

`py -m src.export_model` rebuilds those joblib files from the selected key in `evaluation.json`. Use it if you already have results and only need the app artifacts.

The analytical notebook is `notebooks/01_youtube_category_suitability.ipynb`. It imports the shared mapping and loads `evaluation.json`. It does not re-fit the four models.

## Model comparison (development holdout n=1,271)

Same video membership for every row. Majority class on this split: Entertainment & Culture (658/1,271).

| Model | Accuracy | Macro-F1 | News recall | Kids News approved | Kids excluded approved | Kids allowed retained |
|---|---:|---:|---:|---:|---:|---:|
| Majority baseline | 0.518 | 0.114 | 0.000 | 104/104 | 523/523 | 748/748 (1.000) |
| Unigram LR, no weights | 0.796 | 0.673 | 0.712 | 5/104 | 77/523 | 651/748 (0.870) |
| **Unigram LR, balanced (selected)** | **0.789** | **0.718** | **0.894** | **0/104** | **15/523** | **477/748 (0.638)** |
| Unigram+bigram LR, balanced | 0.780 | 0.716 | 0.894 | 0/104 | 14/523 | 467/748 (0.624) |

Per-class precision / recall / F1 / support for the selected model:

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Entertainment & Culture | 0.893 | 0.809 | 0.848 | 658 |
| Lifestyle & Interests | 0.689 | 0.772 | 0.728 | 272 |
| Education, Science & Technology | 0.682 | 0.714 | 0.698 | 126 |
| News, Politics & Society | 0.679 | 0.894 | 0.772 | 104 |
| Sports | 0.872 | 0.756 | 0.810 | 90 |
| Gaming | 0.435 | 0.476 | 0.455 | 21 |

Confusion matrices and the full policy tables are in `reports/evaluation.json`.

## Policy outcomes (selected model)

Mapped YouTube categories are **proxies** for contextual suitability. They are not human-verified placement or safety labels. Thresholds other than Kids & Family 0.55 are illustrative starting points.

| Policy (threshold) | News approved | Incorrect / excluded | Incorrect / approved | Allowed retained | Approval rate |
|---|---:|---:|---:|---:|---:|
| Kids & Family (0.55, development candidate) | 0/104 | 15/523 (0.029) | 15/492 (0.030) | 477/748 (0.638) | 492/1,271 (0.387) |
| Financial Services (0.60, illustrative) | 14/104 | 24/125 (0.192) | 24/1,142 (0.021) | 1,118/1,146 (0.976) | 1,142/1,271 (0.899) |
| Sports & Apparel (0.50, illustrative) | 19/104 | 59/230 (0.257) | 59/1,070 (0.055) | 1,011/1,041 (0.971) | 1,070/1,271 (0.842) |
| Gaming Publisher (0.45, illustrative) | 0/104 | 34/592 (0.057) | 34/537 (0.063) | 503/679 (0.741) | 537/1,271 (0.423) |

Kids & Family excluded-category breakdown at 0.55: Lifestyle 11, Gaming 4, Education 0, News 0.

Kids & Family threshold curve (incorrect approvals vs inventory retained):

| Threshold | Incorrect | News | Allowed retained | Approved |
|---:|---:|---:|---:|---:|
| 0.40 | 50 | 6 | 0.810 | 656 |
| 0.45 | 31 | 0 | 0.753 | 594 |
| 0.50 | 17 | 0 | 0.701 | 541 |
| 0.55 | 15 | 0 | 0.638 | 492 |
| 0.60 | 11 | 0 | 0.561 | 431 |
| 0.65 | 7 | 0 | 0.497 | 379 |
| 0.70 | 7 | 0 | 0.430 | 329 |

## Validation status

The 1,271-video slice is a **development holdout**. It was used to compare models, read errors, talk about 0.55, and check calibration. It is not a sealed test set. A new random split of these same videos would not be independent confirmation.

Calibration (Kids allowed-category score vs “mapped label is allowed”): ECE 0.163, Brier 0.136. Mid-range scores are not frequencies. No calibrator was fitted. The score is a ranking input to the policy, not a proven probability.

Robustness with the **frozen selected spec** (not a confirmation set):

| Check | Holdout n | Accuracy | Macro-F1 | News recall | Kids incorrect | News approved | Rare-class warning |
|---|---:|---:|---:|---:|---:|---:|---|
| Channel-disjoint | 1,077 | 0.705 | 0.621 | 0.605 | 24 | 0 | none (<15) |
| Temporal (later publish times) | 1,270 | 0.813 | 0.748 | 0.929 | 21 | 0 | none (<15) |

**External validation plan:** freeze `src/config.py`, the selected spec, and the Kids 0.55 candidate. Score a later US scrape or another country file that was never used here. Do not reshuffle this CSV and call it a test set.

## Run locally — walkthrough for a screen recording

1. Open [https://brandsuit.streamlit.app/](https://brandsuit.streamlit.app/) or, from this folder, `py -m streamlit run app.py`.
2. **Live Decision** loads an illustrative **Sports** example (`NBA Finals Game 7 highlights`). Keep **Kids & Family Brand**. Click **Analyze placement**. Expect **YES**: Sports is on the allow-list.
3. Switch the advertiser to **Financial Services** without changing the text. Category scores stay put; only the allow-list and threshold change.
4. Load **News**, Analyze again, Kids & Family. Expect **NO**.
5. Load **Ambiguous talk-show** (the Infowars title used in error analysis). Analyze, Kids & Family. Expect **NO** even if the top class is Entertainment — the allowed-category score stays below 0.55.
6. Replace the title with `qzxv blorp zzzqq`, clear the description, Analyze. Expect **Insufficient information**, not a Financial Services YES.
7. Edit the title after a decision: the YES/NO banner hides until you analyze again. Clearing the fields removes the previous approval.
8. Open **Project Story**. Start with the snapshot (problem, selected unigram model, Kids News 0/104 vs excluded-category 15/523, holdout limitation). Expand sections for comparison, errors, and the threshold curve.

## Project layout

```text
brandsuit-ai/
├── app.py                  # Streamlit: score text, then apply a policy
├── src/config.py           # mapping, split, four model specs, selection rule
├── src/dataset.py          # load, fingerprint, primary and robustness splits
├── src/modeling.py         # fit / predict_proba for dummy and logistic specs
├── src/evaluate.py         # reproducible experiment command
├── src/export_model.py     # write the selected joblib files
├── src/decision.py         # abstention, allowed-category score, stale-result state
├── src/policies/           # allow-lists and policy counts
├── src/story.py            # Project Story tab (reads evaluation.json)
├── tests/test_pipeline.py  # overlap, mapping, policy arithmetic, saved-model match
├── tests/test_decision.py  # empty input, OOV abstention, short titles, stale state
├── notebooks/              # completed analytical narrative
├── reports/evaluation.json # source of documented numbers
├── DECISIONS.md            # historical log + current summary
└── README.md               # this file
```

## Tested versions

Python 3.13.15, pandas 3.0.5, numpy 2.5.2, scikit-learn 1.9.1, joblib 1.6.0, streamlit 1.63.0, matplotlib 3.11.1, seaborn 0.13.2, pytest 9.1.1, ipykernel 7.3.0, notebook 7.6.2. Install from `requirements.txt` so the app, tests, and `notebooks/01_youtube_category_suitability.ipynb` share the same versions.

## Limitations

- Labels are YouTube categories, sometimes wrong (UFC and Champions League highlights sitting in Lifestyle).
- Gaming support is 21 holdout videos; one error moves recall a lot.
- No unsafe-content labels, so this cannot be sold as brand safety.
- The allowed-category score is not well calibrated.
- The holdout was reused for selection. Treat production claims as unconfirmed until an external file is scored with the frozen pipeline.

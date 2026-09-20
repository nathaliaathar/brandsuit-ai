# Decisions

## D1 — MVP problem definition (Phase 1)

**Decision:** Text-only multiclass classifier for YouTube videos.
One row = one video. X = title + description. y = one of:
Sports, News/Politics, Gaming, Entertainment, Violence/Sensitive, Other.

**Labeling policy:** If sensitive/violent content is present, y = Violence/Sensitive
even when the video is also sports or gaming. News/Politics without graphic
violence stays News/Politics. Food and Travel are folded into Other for the MVP.

**Worst error:** False negative on Violence/Sensitive (ad runs next to content
that burns the brand). Lost inventory (false positive on Sensitive) is cheaper.

**Primary metric:** Recall on Violence/Sensitive.
**Guardrails:** Precision on Sensitive + macro-F1.
**Do not optimize:** Accuracy (majority baseline looks strong under imbalance).

**Alternatives considered:**
- Image or multimodal from day one — rejected; CV hides the classification basics.
- Binary safe vs sensitive — rejected; too thin for a portfolio and drops content type.
- Multilabel — rejected for MVP; closer to YouTube, but metrics and public datasets
  are harder. To revisit after a text baseline.

**Limitations:** Real videos can have several topics. Channel/YouTube category
must not go into X (leakage). War journalism vs graphic violence is subjective
(label noise). English/YouTube data will not represent other platforms or languages.

**Next experiment:** Choose a small public text dataset (Phase 2). See D2:
the public trending data has no Violence/Sensitive labels, so the MVP
shifted from safety to suitability.

## D2 — Single dataset, category + brand match (Phase 2)

**Decision:** Use only YouTube Trending (US). One row = one video.
X = title + description. The model predicts one of:
Sports, News/Politics, Gaming, Entertainment, Other.

Brand yes/no is not a second model and not a second table.
It is a policy: match if predicted category is in that brand's allow list.

Example policies:
- Nike: Sports, Gaming, Entertainment
- Bank: Sports, Entertainment, Other
- Kids cereal: Sports, Entertainment

**Not in scope for this dataset:** violence/hate/unsafe. Trending is
already filtered. Calling this "brand safety" would overclaim.
This MVP is brand *suitability* / contextual targeting.

**Worst error (cereal example):** predicting News as Entertainment
(false yes → ad next to politics). False no on Sports is lost inventory.

**Primary metric:** macro-F1 on the 5 categories.
**Do not optimize:** accuracy; do not put category_id or channel in X.

**Alternatives considered:**
- Two models (Jigsaw + YouTube) — rejected; no shared grain; the brand
  does not buy Wikipedia comments.
- Unsafe labels on trending — not available; do not invent them.
- One model per brand (Nike vs bank) — rejected; no Nike labels in the CSV.

**Limitations:** category_id is the training label, never a feature (leakage).
The same video can trend on several days (deduplicate by video_id).
YouTube official categories are not identical to our 5-class map.

**Next experiment:** US files are in `data/raw/` (`USvideos.csv` gitignored).
See D3 for the remapped labels, split, and first text model.

## D3 — Label groups, split, TF-IDF + logistic, class_weight

**Decision:** Collapse YouTube names into six suitability groups (not the
original five + Other dump):

- Sports
- Gaming
- Entertainment & Culture (Entertainment, Comedy, Music, Film, Shows)
- News, Politics & Society (News & Politics, Nonprofits & Activism)
- Lifestyle & Interests (People & Blogs, Howto, Pets, Travel, Autos)
- Education, Science & Technology

Deduplicate on `video_id` first (40,949 rows → 6,351 videos).
X = title + description. 80/20 stratified split, `random_state=42`.
Vectorize with TF-IDF fitted on train only. Classifier = LogisticRegression.

**Brand policies (updated names):**
- Nike: Sports, Gaming, Entertainment & Culture
- Bank: Sports, Entertainment & Culture, Lifestyle, Education
- Kids cereal: Sports, Entertainment & Culture
  (block News; Gaming is not kids inventory)

**Metrics on test (n=1271), majority class Entertainment & Culture ~52%:**

| Setup | Accuracy | Macro-F1 | News recall | Gaming recall |
|---|---|---|---|---|
| Dummy (always Entertainment) | 0.518 | ~0 | 0 | 0 |
| LogReg, no class_weight | 0.80 | 0.67 | 0.71 | 0.14 |
| LogReg, `class_weight="balanced"` | 0.79 | 0.72 | 0.89 | 0.48 |

**Choice for the cereal story:** keep `class_weight="balanced"`.
News recall matters more than a 1-point accuracy drop. Cost: News
precision 0.89 → 0.68 (more false News flags, some lost inventory).

**Do not:** fit TF-IDF on X_test (leakage). Put `category_id` in X.

**Limitations:** Gaming support is tiny (21 in test). Comedy vs comedy-film
is not in the data. This is still suitability, not violence detection.

**Next experiment:** done in D4 (error analysis → n-grams → confidence policy).

## D4 — Error analysis, n-grams, brand confidence threshold

**Decision:** Keep TF-IDF `ngram_range=(1, 2)` + LogisticRegression
`class_weight="balanced"`. Do not ship on `predict` (argmax) alone.

Cereal yes only if
`p_allow = P(Sports) + P(Entertainment & Culture) >= THRESH`.

Brand-specific THRESH (same model, different risk):
- Kids cereal: **0.55** (priority: 0 News next to the ad)
- Nike / less sensitive: **0.50** (priority: inventory)

**Evidence (test n=1271):**

| Experiment | Macro-F1 | Cereal false yes (News published as Ent) | Notes |
|---|---|---|---|
| Unigram + balanced | 0.718 | 4 | argmax leaks |
| Bigram `(1, 2)` + balanced | 0.716 | 2 | product moved; global metric did not |
| Bigram + `THRESH=0.55` | — | 0 News with `p_allow>=0.55` | 254 true Entertainment blocked; 481 yes / 1271 |

Remaining argmax leaks after n-grams:
- *Milo Takes Calls From Infowars Listeners* — talk-show wording; **model error**. `p_allow ≈ 0.40`
- *Rose McGowan / Time's Up* — celebrity interview; **label is dubious**. `p_allow ≈ 0.47`

Both sit below 0.50, so 0.50 already blocks these two. 0.55 is extra buffer for kids.

**Alternatives considered:**
- `(1, 3)` n-grams — rejected; same trick, more noise.
- More News class weight — already used `balanced`; would not fix talk-show format.
- Relabel 20 videos — useful for Rose, does not catch Infowars, and relabeling Rose to Entertainment would make cereal say yes.

**Do not:** treat probabilities as certainty. Fit TF-IDF on test.

**Limitations:** 4 → 2 on n-grams is a small count (could be luck). Threshold 0.55
throws away a lot of Entertainment (~39% of true Ent). Time's Up as Entertainment
would still be a strange kids-cereal placement (allow-list limit). No image model yet.

**Next experiment:** text-only Streamlit demo — paste title+description, show
top classes + `p_allow` + cereal/Nike yes-no. No image upload until there is
an image model. Do not overclaim brand *safety*.

## D5 — Product name: BrandSuit AI (not BrandSafe)

**Decision:** User-facing name is **BrandSuit AI**. The sklearn model is a
**category classifier** (`category_tfidf.joblib` + `category_logreg.joblib`).
Brand YES/NO is policy, not a risk head.

**Why:** "Safe" sounds like violence/unsafe scoring. This MVP never had that
label. Suitability = does the predicted category match the brand allow-list.

The public GitHub repo is `brandsuit-ai`. The older folder name `Brandsafe-ai`
was only a local path.

**Next:** Streamlit demo with Kids & Family at 0.55 (Infowars → NO). Do not
overclaim brand safety.

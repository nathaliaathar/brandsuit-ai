# Decisions

This file is a **historical record** of how the project evolved, plus a
short current freeze. Interview answers should start from the current
summary and `reports/evaluation.json`, not from older metric tables below.

---

## Current decision summary (portfolio freeze)

**Problem.** Text-only multiclass category prediction on unique YouTube
Trending videos, then an advertiser allow-list. Suitability, not safety.

**Mapping.** Six advertiser-context groups in `src/config.py`. Pets &
Animals stay in Lifestyle & Interests. A previous export mapped Pets into
Education after looking at coefficients; that is rejected. The grouping
is about placement context, not about raising macro-F1.

**Split.** 80/20 stratified on the mapped label, `random_state=42`, unit
`video_id`. This slice is a **development holdout** (comparison, errors,
threshold talk, calibration). It is not a sealed test set.

**Candidates compared on that same membership.** Majority baseline;
unigram LR without weights; unigram LR with balanced weights;
unigram-plus-bigram LR with balanced weights.

**Selected model.** Unigram TF-IDF + logistic regression,
`class_weight="balanced"`. Rule: minimize Kids & Family excluded-category
approvals at 0.55; treat gaps of 3 or fewer videos as a tie; then maximize
retention of allowed labels; then prefer the simpler spec.

**Evidence (holdout n=1,271, from `reports/evaluation.json`).**
Selected: accuracy 0.789, macro-F1 0.718, News recall 0.894.
Kids & Family: News 0/104, all excluded 15/523 (Lifestyle 11, Gaming 4),
retention 477/748. Balanced bigrams: 14/523 excluded and 467/748 retained.
The one-video leak gap is inside the tie band; bigrams are not kept for
sounding more advanced.

**Policies.** Count every excluded category, not News alone. Kids 0.55 is a
development-holdout candidate. Other advertiser thresholds are illustrative.
Labels are YouTube proxies, not human suitability audits.

**Calibration.** Kids `p_allow` ECE 0.163 on the development holdout. No
calibrator fitted. `p_allow` is a score.

**Robustness.** Frozen selected spec on a channel-disjoint split and a
later-publish-time split. Channel-disjoint macro-F1 drops (0.621). These
are stress tests, not a new random confirmation set.

**App numbers.** Streamlit reads `reports/evaluation.json`. Empty or
out-of-vocabulary text abstains (`Insufficient information`) instead of
approving from the intercept. The user-facing figure is an allowed-category
score, not a verified safety probability.

**Next.** Freeze this pipeline and score an external file (later US scrape
or another country). Do not reshuffle this CSV and call it a test set.

---

## Historical record

The sections below are kept so the learning path stays visible. Metrics in
D3/D4 used an earlier mapping and should not be quoted as current results.

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

**Do not:** rename the folder `Brandsafe-ai` yet (paths/git). Keep
`project_instructions.md` as the original brief.

**Next experiment:** finish Streamlit (`THRESH_CEREAL` from D4) and demo
Infowars → cereal NO.

## D6 — Shared pipeline, honest policy counts, unigram selected (historical close)

**Decision:** One config module; Pets remain Lifestyle; evaluate four specs
on one development holdout; count every excluded category; select unigram
balanced LR; call 0.55 a development candidate; document calibration and
channel/temporal robustness.

**Why this supersedes D4's "keep bigrams":** D4 optimized a News-only leak
count and treated the holdout like a test set. Recomputed policy metrics
show News 0/104 for both balanced models and 15 vs 14 excluded-category
approvals. That is not a reason to keep n-grams.

**Evidence:** `reports/evaluation.json` from `py -m src.evaluate`.

**Do not:** quote D3/D4 tables as current results; remap Pets to Education
to clean coefficients; describe a reshuffle of this CSV as a confirmation
set.

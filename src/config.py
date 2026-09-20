"""Single source of truth for mapping, text, splits, and model specs.

The notebook, evaluator, exporter, tests, and app all import from here
so a mapping or hyperparameter cannot drift in one file and not the others.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "raw" / "USvideos.csv"
CATEGORY_JSON = ROOT / "data" / "raw" / "US_category_id.json"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
EVALUATION_PATH = REPORTS_DIR / "evaluation.json"
SPLIT_IDS_PATH = REPORTS_DIR / "split_ids.json"
VEC_PATH = MODELS_DIR / "category_tfidf.joblib"
MODEL_PATH = MODELS_DIR / "category_logreg.joblib"
MANIFEST_PATH = MODELS_DIR / "manifest.json"

RANDOM_STATE = 42
HOLDOUT_SIZE = 0.20
MAX_ITER = 1000
MAX_FEATURES = 5000
MIN_DF = 2

# Stable report order (roughly most common first). Not used as a feature.
TARGET_CLASSES = (
    "Entertainment & Culture",
    "Lifestyle & Interests",
    "Education, Science & Technology",
    "News, Politics & Society",
    "Sports",
    "Gaming",
)

NEWS_CLASS = "News, Politics & Society"

# Advertiser-context grouping of official YouTube category names.
# Chosen from how a brand would treat the *placement context*, not from
# whichever assignment raised macro-F1.
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
    "Pets & Animals": "Lifestyle & Interests",
    "Science & Technology": "Education, Science & Technology",
    "Shows": "Entertainment & Culture",
    "Sports": "Sports",
    "Travel & Events": "Lifestyle & Interests",
}

MAPPING_VERSION = "advertiser-context-v1"

CATEGORY_MAPPING_RATIONALE = (
    "YouTube's 16 official names are too fine-grained for four advertiser "
    "allow-lists, so they are grouped by placement context. "
    "Entertainment, Comedy, Music, Film & Animation, and Shows are one "
    "culture/entertainment bucket. News & Politics and Nonprofits & Activism "
    "are one public-affairs bucket. Education stays with Science & Technology "
    "as instructional/STEM inventory. Sports and Gaming stay separate because "
    "brands treat them differently (a kids brand may allow sports and still "
    "exclude games). Lifestyle & Interests covers People & Blogs, Howto & Style, "
    "Travel & Events, Autos & Vehicles, and Pets & Animals: hobby, vlog, and "
    "consumer-interest inventory. Pets stay here on purpose. A pet vlog is not "
    "STEM instruction; putting Pets into Education would mix cat/dog vocabulary "
    "into an education allow-list and was previously done in export_model.py "
    "after seeing coefficients. That mapping is rejected: we do not move a "
    "class to improve a metric. Unmapped YouTube names, if any appear, become Other."
)

# Primary split: used for fitting vs comparing models and choosing a threshold.
# It is a development holdout, not a sealed confirmation set.
PRIMARY_SPLIT = {
    "name": "development_holdout",
    "method": "sklearn.model_selection.train_test_split",
    "test_size": HOLDOUT_SIZE,
    "random_state": RANDOM_STATE,
    "stratify": "mapped_category",
    "unit": "video_id",
    "role": (
        "Train is used only to fit TF-IDF and classifiers. The 20% slice is "
        "reused for model comparison, error analysis, threshold discussion, "
        "and calibration checks. It is therefore a development holdout, not "
        "an untouched test set. A later random split of the same videos would "
        "not be independent confirmation."
    ),
}

# Complexity rank is the last tie-breaker in model selection (lower = simpler).
MODEL_SPECS = {
    "majority_baseline": {
        "kind": "dummy",
        "strategy": "most_frequent",
        "label": "Majority-class baseline",
        "complexity": 0,
    },
    "unigram_unweighted": {
        "kind": "logreg",
        "ngram_range": (1, 1),
        "class_weight": None,
        "max_features": MAX_FEATURES,
        "min_df": MIN_DF,
        "max_iter": MAX_ITER,
        "label": "Unigram logistic regression, no class weights",
        "complexity": 1,
    },
    "unigram_balanced": {
        "kind": "logreg",
        "ngram_range": (1, 1),
        "class_weight": "balanced",
        "max_features": MAX_FEATURES,
        "min_df": MIN_DF,
        "max_iter": MAX_ITER,
        "label": "Unigram logistic regression, balanced weights",
        "complexity": 2,
    },
    "bigram_balanced": {
        "kind": "logreg",
        "ngram_range": (1, 2),
        "class_weight": "balanced",
        "max_features": MAX_FEATURES,
        "min_df": MIN_DF,
        "max_iter": MAX_ITER,
        "label": "Unigram-plus-bigram logistic regression, balanced weights",
        "complexity": 3,
    },
}

# Kids & Family is the strictest product story: incorrect approvals are costly.
SELECTION_POLICY = "Kids & Family Brand"
# 14 vs 15 incorrect approvals is one video on a reused holdout; treat small
# absolute gaps as ties rather than a reason to keep extra n-grams.
INCORRECT_APPROVAL_TIE_DELTA = 3
SELECTION_RULE = (
    "On the development holdout, at each model's Kids & Family stated "
    "threshold, start from incorrect approvals across every excluded category "
    "(not News alone). Specs within INCORRECT_APPROVAL_TIE_DELTA videos of the "
    "lowest count are treated as tied, because a 1-2 video gap on this holdout "
    "is small-count noise. Among that tie group, maximize retention of videos "
    "whose mapped label is allowed. If still tied, prefer the simpler spec "
    "(baseline, then unigram, then balanced unigram, then bigrams). Macro-F1 is "
    "reported and is not the selection criterion. Bigrams are not kept because "
    "they sound more advanced."
)

THRESHOLD_SWEEP = (0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70)
COMPARABLE_THRESHOLDS = (0.45, 0.50, 0.55, 0.60)

TEMPORAL_QUANTILE = 0.80
CHANNEL_HOLDOUT_FRACTION = 0.20
CALIBRATION_BINS = 10

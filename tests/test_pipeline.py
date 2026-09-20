"""Focused checks: mapping, splits, policy arithmetic, saved-model match."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import pytest

from src.config import (
    CATEGORY_MAPPING,
    EVALUATION_PATH,
    MODEL_PATH,
    RAW_CSV,
    SPLIT_IDS_PATH,
    TARGET_CLASSES,
    VEC_PATH,
)
from src.dataset import make_primary_split, prepare_dataset
from src.modeling import FittedBundle, predict_proba_frame
from src.policies.brand_policies import (
    BRAND_POLICIES,
    evaluate_policy,
    excluded_categories,
    policy_outcome_metrics,
    safe_div,
)

ROOT = Path(__file__).resolve().parents[1]


def test_pets_mapping_is_lifestyle_not_education():
    assert CATEGORY_MAPPING["Pets & Animals"] == "Lifestyle & Interests"
    assert CATEGORY_MAPPING["Education"] == "Education, Science & Technology"
    assert CATEGORY_MAPPING["Science & Technology"] == "Education, Science & Technology"


def test_single_mapping_source():
    export_src = (ROOT / "src" / "export_model.py").read_text(encoding="utf-8")
    assert "CATEGORY_MAPPING =" not in export_src
    assert "from src.config import" in export_src or "src.config" in export_src


def test_policy_sum_and_threshold_boundary():
    kids = BRAND_POLICIES["Kids & Family Brand"]
    probs = {
        "Sports": 0.20,
        "Entertainment & Culture": 0.35,
        "News, Politics & Society": 0.45,
    }
    p_allow, approved = evaluate_policy(probs, kids, threshold=0.55)
    assert math.isclose(p_allow, 0.55)
    assert approved is True
    _, just_below = evaluate_policy(probs, kids, threshold=0.5500001)
    assert just_below is False
    missing = {"News, Politics & Society": 1.0}
    p_missing, approved_missing = evaluate_policy(missing, kids, threshold=0.01)
    assert p_missing == 0.0
    assert approved_missing is False


def test_excluded_categories_include_more_than_news_for_kids():
    kids = BRAND_POLICIES["Kids & Family Brand"]
    excluded = excluded_categories(kids)
    assert "News, Politics & Society" in excluded
    assert "Gaming" in excluded
    assert "Lifestyle & Interests" in excluded
    assert "Education, Science & Technology" in excluded
    assert "Sports" not in excluded
    assert "Entertainment & Culture" not in excluded


def test_policy_counts_every_excluded_category_and_zero_denominators():
    y = pd.Series(
        [
            "News, Politics & Society",
            "Gaming",
            "Lifestyle & Interests",
            "Entertainment & Culture",
            "Sports",
        ]
    )
    proba = pd.DataFrame(
        [
            {
                "Sports": 0.10,
                "Entertainment & Culture": 0.60,
                "News, Politics & Society": 0.30,
                "Gaming": 0.0,
                "Lifestyle & Interests": 0.0,
                "Education, Science & Technology": 0.0,
            },
            {
                "Sports": 0.05,
                "Entertainment & Culture": 0.70,
                "News, Politics & Society": 0.0,
                "Gaming": 0.25,
                "Lifestyle & Interests": 0.0,
                "Education, Science & Technology": 0.0,
            },
            {
                "Sports": 0.00,
                "Entertainment & Culture": 0.20,
                "News, Politics & Society": 0.0,
                "Gaming": 0.0,
                "Lifestyle & Interests": 0.80,
                "Education, Science & Technology": 0.0,
            },
            {
                "Sports": 0.10,
                "Entertainment & Culture": 0.80,
                "News, Politics & Society": 0.10,
                "Gaming": 0.0,
                "Lifestyle & Interests": 0.0,
                "Education, Science & Technology": 0.0,
            },
            {
                "Sports": 0.90,
                "Entertainment & Culture": 0.05,
                "News, Politics & Society": 0.05,
                "Gaming": 0.0,
                "Lifestyle & Interests": 0.0,
                "Education, Science & Technology": 0.0,
            },
        ]
    )
    kids = BRAND_POLICIES["Kids & Family Brand"]
    metrics = policy_outcome_metrics(y, proba, kids, threshold=0.55)
    assert metrics["news_approvals"] == 1
    assert metrics["incorrect_approvals"] == 2
    assert metrics["incorrect_approvals_by_category"]["News, Politics & Society"] == 1
    assert metrics["incorrect_approvals_by_category"]["Gaming"] == 1
    assert metrics["incorrect_approvals_by_category"]["Lifestyle & Interests"] == 0
    assert metrics["n_approved"] == 4
    assert metrics["n_truly_excluded"] == 3
    assert metrics["n_truly_allowed"] == 2
    assert metrics["n_retained_allowed"] == 2
    assert metrics["incorrect_over_approved"] == pytest.approx(2 / 4)
    assert metrics["incorrect_over_excluded"] == pytest.approx(2 / 3)
    assert metrics["retention_of_allowed"] == pytest.approx(1.0)
    assert metrics["overall_approval_rate"] == pytest.approx(4 / 5)

    empty = policy_outcome_metrics(y.iloc[0:0], proba.iloc[0:0], kids, threshold=0.55)
    assert empty["n_approved"] == 0
    assert empty["incorrect_over_approved"] is None
    assert empty["incorrect_over_excluded"] is None
    assert empty["retention_of_allowed"] is None
    assert empty["overall_approval_rate"] is None
    assert safe_div(1, 0) is None


@pytest.mark.skipif(not RAW_CSV.exists(), reason="USvideos.csv is not available")
def test_no_video_overlap_in_primary_split():
    df = prepare_dataset()
    train, holdout, info = make_primary_split(df)
    overlap = set(train["video_id"]).intersection(set(holdout["video_id"]))
    assert overlap == set()
    assert info["n_video_overlap"] == 0
    if SPLIT_IDS_PATH.exists():
        saved = json.loads(SPLIT_IDS_PATH.read_text(encoding="utf-8"))
        assert set(saved["train_video_ids"]).isdisjoint(saved["holdout_video_ids"])
        assert set(saved["train_video_ids"]) == set(train["video_id"].astype(str))
        assert set(saved["holdout_video_ids"]) == set(holdout["video_id"].astype(str))


@pytest.mark.skipif(
    not (RAW_CSV.exists() and VEC_PATH.exists() and MODEL_PATH.exists() and EVALUATION_PATH.exists()),
    reason="dataset, saved model, or evaluation.json missing",
)
def test_saved_model_matches_evaluated_pipeline():
    import joblib
    from sklearn.metrics import accuracy_score, f1_score

    payload = json.loads(EVALUATION_PATH.read_text(encoding="utf-8"))
    if payload.get("status") != "completed":
        pytest.skip("evaluation did not complete")
    df = prepare_dataset()
    _, holdout, _ = make_primary_split(df)
    vec = joblib.load(VEC_PATH)
    model = joblib.load(MODEL_PATH)
    bundle = FittedBundle(
        spec_key=payload["selected_model"]["key"],
        spec={},
        vectorizer=vec,
        classifier=model,
    )
    proba = predict_proba_frame(bundle, holdout["text"])
    pred = proba.idxmax(axis=1)
    acc = accuracy_score(holdout["y"], pred)
    macro = f1_score(holdout["y"], pred, average="macro", zero_division=0)
    expected = payload["selected_model"]["classification"]
    assert acc == pytest.approx(expected["accuracy"], abs=1e-9)
    assert macro == pytest.approx(expected["macro_f1"], abs=1e-9)
    manifest = json.loads((ROOT / "models" / "manifest.json").read_text(encoding="utf-8"))
    assert payload["dataset"]["mapping_sha256_16"] == manifest["mapping_sha256_16"]
    assert set(TARGET_CLASSES) == set(model.classes_)

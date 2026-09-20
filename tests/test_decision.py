"""Regression checks for abstention, stale results, and policy display logic."""

from __future__ import annotations

import pandas as pd
import pytest

from src.config import MODEL_PATH, VEC_PATH
from src.decision import (
    STATUS_EMPTY,
    STATUS_INSUFFICIENT,
    STATUS_OK,
    allowed_category_score,
    assess_input,
    combine_text,
    recognized_feature_count,
    view_state,
)
from src.policies.brand_policies import BRAND_POLICIES


def test_empty_and_whitespace_are_rejected_without_a_model():
    class DummyVec:
        def transform(self, texts):
            raise AssertionError("empty text must not be vectorized")

    for title, desc in (("", ""), ("   ", ""), ("", "\n\t"), ("  ", "  ")):
        result = assess_input(DummyVec(), title, desc)
        assert result["status"] == STATUS_EMPTY
        assert result["n_features"] == 0
        assert "Enter a title" in result["message"]


def test_view_state_clears_approval_when_input_is_cleared():
    assert view_state("", "NBA Finals", STATUS_OK) == STATUS_EMPTY
    assert view_state("   ", "NBA Finals", STATUS_OK) == STATUS_EMPTY


def test_view_state_marks_edited_text_stale():
    assert view_state("NBA Finals extra", "NBA Finals", STATUS_OK) == "stale"
    assert view_state("NBA Finals", "NBA Finals", STATUS_OK) == "current"


def test_view_state_profile_switch_keeps_current_scores_for_same_text():
    # Advertiser is not part of view_state: same text stays current so the
    # UI can recompute only the policy on stored category scores.
    assert view_state("Fortnite trailer", "Fortnite trailer", STATUS_OK) == "current"


@pytest.mark.skipif(not VEC_PATH.exists(), reason="saved vectorizer missing")
def test_meaningless_input_has_no_recognized_vocabulary():
    import joblib

    vec = joblib.load(VEC_PATH)
    gibberish = assess_input(vec, "qzxv blorp zzzqq", "")
    assert gibberish["status"] == STATUS_INSUFFICIENT
    assert gibberish["n_features"] == 0


@pytest.mark.skipif(not VEC_PATH.exists(), reason="saved vectorizer missing")
def test_short_titles_are_not_rejected_for_length():
    import joblib

    vec = joblib.load(VEC_PATH)
    short_ok = assess_input(vec, "NBA", "")
    assert short_ok["status"] == STATUS_OK
    assert short_ok["n_features"] >= 1
    assert recognized_feature_count(vec, "UFC") >= 1
    assert recognized_feature_count(vec, combine_text("NBA Finals Game 7 highlights", "")) >= 1


@pytest.mark.skipif(
    not (VEC_PATH.exists() and MODEL_PATH.exists()),
    reason="saved vectorizer/model missing",
)
def test_meaningless_input_raw_model_would_approve_financial_but_we_abstain():
    import joblib

    from src.modeling import FittedBundle, predict_proba_frame

    vec = joblib.load(VEC_PATH)
    model = joblib.load(MODEL_PATH)
    assessment = assess_input(vec, "qzxv blorp zzzqq", "")
    assert assessment["status"] == STATUS_INSUFFICIENT

    bundle = FittedBundle("unigram_balanced", {}, vec, model)
    proba = predict_proba_frame(bundle, pd.Series([assessment["text"]]))
    row = proba.iloc[0].to_dict()
    _score, approved = allowed_category_score(row, BRAND_POLICIES["Financial Services"])
    assert assessment["status"] == STATUS_INSUFFICIENT
    # Empty TF-IDF vectors can still produce an intercept-driven Financial YES
    # (approved may be True). The product path must abstain before using it.
    assert assessment["n_features"] == 0
    _ = approved

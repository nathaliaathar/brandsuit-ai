"""Advertiser policies: allow-list + threshold.

This is a business rule on top of the category classifier, not a second model.
Mapped YouTube categories are proxies for contextual suitability, not
human-verified placement or safety labels.
"""

from __future__ import annotations

from src.config import NEWS_CLASS, TARGET_CLASSES

DEVELOPMENT_CANDIDATE_SOURCE = "Development-holdout candidate"
ILLUSTRATIVE_SOURCE = "Illustrative — not selected from holdout counts"

# Keep the old name so the app highlight logic can still import one constant.
VALIDATED_SOURCE = DEVELOPMENT_CANDIDATE_SOURCE

BRAND_POLICIES = {
    "Kids & Family Brand": {
        "risk_profile": "Strict",
        "allowed_categories": [
            "Sports",
            "Entertainment & Culture",
        ],
        "threshold": 0.55,
        "threshold_source": DEVELOPMENT_CANDIDATE_SOURCE,
        "threshold_note": (
            "0.55 was discussed on the development holdout while inspecting "
            "News leaks. It is labeled a candidate, not a production threshold."
        ),
    },
    "Financial Services": {
        "risk_profile": "Conservative",
        "allowed_categories": [
            "Sports",
            "Entertainment & Culture",
            "Lifestyle & Interests",
            "Education, Science & Technology",
        ],
        "threshold": 0.60,
        "threshold_source": ILLUSTRATIVE_SOURCE,
        "threshold_note": "Starting point for a conservative allow-list; not tuned on holdout counts.",
    },
    "Sports & Apparel": {
        "risk_profile": "Balanced",
        "allowed_categories": [
            "Sports",
            "Gaming",
            "Entertainment & Culture",
            "Lifestyle & Interests",
        ],
        "threshold": 0.50,
        "threshold_source": ILLUSTRATIVE_SOURCE,
        "threshold_note": "Starting point; not tuned on holdout counts.",
    },
    "Gaming Publisher": {
        "risk_profile": "Reach-focused",
        "allowed_categories": [
            "Gaming",
            "Entertainment & Culture",
        ],
        "threshold": 0.45,
        "threshold_source": ILLUSTRATIVE_SOURCE,
        "threshold_note": "Starting point for reach; not tuned on holdout counts.",
    },
}


def excluded_categories(policy, all_classes=TARGET_CLASSES):
    allowed = set(policy["allowed_categories"])
    return [cls for cls in all_classes if cls not in allowed]


def evaluate_policy(probabilities, policy, threshold=None):
    """Sum the probabilities this advertiser allows and compare to a threshold.

    probabilities: {category_name: probability} built from model.classes_
    Categories missing from the model are treated as 0.0 instead of raising.
    Approval is p_allow >= threshold (the boundary counts as approved).
    """
    cut = policy["threshold"] if threshold is None else threshold
    p_allow = sum(
        float(probabilities.get(category, 0.0))
        for category in policy["allowed_categories"]
    )
    return p_allow, p_allow >= cut


def safe_div(numerator, denominator):
    """Return None when the denominator is zero instead of raising or faking 0."""
    if denominator == 0:
        return None
    return numerator / denominator


def policy_outcome_metrics(y_true, proba_frame, policy, threshold=None):
    """Count approvals against mapped labels, with explicit denominators.

    Mapped labels are proxies for contextual suitability. They are not
    human-verified brand-safety or placement labels.
    """
    import pandas as pd

    cut = policy["threshold"] if threshold is None else threshold
    allowed = list(policy["allowed_categories"])
    excluded = excluded_categories(policy)
    p_allow = proba_frame.reindex(columns=allowed, fill_value=0.0).sum(axis=1)
    approved = p_allow >= cut
    y = pd.Series(y_true).reset_index(drop=True)
    approved = pd.Series(approved.to_numpy(), index=y.index)
    p_allow = pd.Series(p_allow.to_numpy(), index=y.index)

    truly_allowed = y.isin(allowed)
    truly_excluded = ~truly_allowed
    incorrect = approved & truly_excluded
    news_approvals = approved & (y == NEWS_CLASS)

    incorrect_by_category = {
        cls: int((incorrect & (y == cls)).sum()) for cls in excluded
    }

    n_total = int(len(y))
    n_approved = int(approved.sum())
    n_excluded = int(truly_excluded.sum())
    n_allowed = int(truly_allowed.sum())
    n_incorrect = int(incorrect.sum())
    n_retained = int((approved & truly_allowed).sum())
    n_news = int((y == NEWS_CLASS).sum())
    n_news_approved = int(news_approvals.sum())

    return {
        "threshold": float(cut),
        "threshold_source": policy.get("threshold_source"),
        "allowed_categories": allowed,
        "excluded_categories": excluded,
        "n_total": n_total,
        "news_approvals": n_news_approved,
        "news_in_split": n_news,
        "incorrect_approvals": n_incorrect,
        "incorrect_approvals_by_category": incorrect_by_category,
        "n_approved": n_approved,
        "n_truly_excluded": n_excluded,
        "n_truly_allowed": n_allowed,
        "n_retained_allowed": n_retained,
        "incorrect_over_approved": safe_div(n_incorrect, n_approved),
        "incorrect_over_excluded": safe_div(n_incorrect, n_excluded),
        "retention_of_allowed": safe_div(n_retained, n_allowed),
        "overall_approval_rate": safe_div(n_approved, n_total),
        "denominators": {
            "incorrect_over_approved": "incorrect_approvals / n_approved (undefined if n_approved=0)",
            "incorrect_over_excluded": "incorrect_approvals / n_truly_excluded (undefined if n_truly_excluded=0)",
            "retention_of_allowed": "n_retained_allowed / n_truly_allowed (undefined if n_truly_allowed=0)",
            "overall_approval_rate": "n_approved / n_total (undefined if n_total=0)",
        },
        "label_caveat": (
            "Numerators and denominators use mapped YouTube categories as "
            "proxies for contextual suitability, not human-verified placement "
            "suitability or safety labels."
        ),
    }

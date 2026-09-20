# Advertiser policies: allow-list + threshold.
# This is a business rule, not a second model.

BRAND_POLICIES = {
    "Kids & Family Brand": {
        "risk_profile": "Strict",
        "allowed_categories": [
            "Sports",
            "Entertainment & Culture",
        ],
        "threshold": 0.55,
        "threshold_source": "Validation candidate",
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
        "threshold_source": "Illustrative — requires validation",
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
        "threshold_source": "Illustrative — requires validation",
    },
    "Gaming Publisher": {
        "risk_profile": "Reach-focused",
        "allowed_categories": [
            "Gaming",
            "Entertainment & Culture",
        ],
        "threshold": 0.45,
        "threshold_source": "Illustrative — requires validation",
    },
}

# Only the Kids & Family threshold came out of an experiment.
VALIDATED_SOURCE = "Validation candidate"


def evaluate_policy(probabilities, policy):
    """Sum the probabilities this advertiser allows and compare to its threshold.

    probabilities: {category_name: probability} built from model.classes_
    Categories missing from the model are treated as 0.0 instead of raising.
    """
    p_allow = sum(
        float(probabilities.get(category, 0.0))
        for category in policy["allowed_categories"]
    )
    return p_allow, p_allow >= policy["threshold"]

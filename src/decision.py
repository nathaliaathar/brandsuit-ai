"""Placement decision helpers used by the app and by tests.

The classifier can still emit probabilities when the TF-IDF vector is empty
(the intercept pushes mass toward Entertainment). That is not a suitability
decision. Abstention is based only on empty text or zero recognized features.
"""

from __future__ import annotations

EMPTY_MESSAGE = (
    "Enter a title or a description before analyzing the placement."
)
NO_FEATURES_MESSAGE = (
    "Insufficient information: none of these words appear in the model's "
    "vocabulary. Add a descriptive title or description (for example a sport, "
    "show, game, or topic) and analyze again."
)

STATUS_EMPTY = "empty"
STATUS_INSUFFICIENT = "insufficient"
STATUS_OK = "ok"


def combine_text(title: str | None, description: str | None) -> str:
    return f"{title or ''} {description or ''}".strip()


def recognized_feature_count(vectorizer, text: str) -> int:
    """How many vocabulary features fired. Zero means the vector is empty."""
    if not text.strip():
        return 0
    return int(vectorizer.transform([text]).nnz)


def assess_input(vectorizer, title: str | None, description: str | None) -> dict:
    """Return status before any YES/NO policy decision.

    Weak-signal investigation: short titles that hit the vocabulary (NBA, UFC,
    Fortnite, news) keep status=ok. Gibberish with nnz=0 is insufficient.
    No character-length cutoff is applied.
    """
    text = combine_text(title, description)
    if not text:
        return {
            "status": STATUS_EMPTY,
            "text": "",
            "n_features": 0,
            "message": EMPTY_MESSAGE,
        }
    n_features = recognized_feature_count(vectorizer, text)
    if n_features == 0:
        return {
            "status": STATUS_INSUFFICIENT,
            "text": text,
            "n_features": 0,
            "message": NO_FEATURES_MESSAGE,
        }
    return {
        "status": STATUS_OK,
        "text": text,
        "n_features": n_features,
        "message": None,
    }


def view_state(current_text: str, scored_text: str | None, analysis_status: str | None) -> str:
    """What the result pane should show.

    empty: no current text — never show a previous approval
    idle: text present, nothing analyzed yet
    stale: text changed since the last analysis
    insufficient: last analysis abstained, text unchanged
    current: last analysis produced category scores, text unchanged
    """
    if not (current_text or "").strip():
        return STATUS_EMPTY
    if analysis_status is None or scored_text is None:
        return "idle"
    if current_text.strip() != scored_text.strip():
        return "stale"
    if analysis_status == STATUS_INSUFFICIENT:
        return STATUS_INSUFFICIENT
    if analysis_status == STATUS_EMPTY:
        return STATUS_EMPTY
    return "current"


def allowed_category_score(probabilities: dict, policy: dict, threshold=None) -> tuple[float, bool]:
    """Sum of model outputs on the selected policy's allow-list.

    This is a ranking score against that profile's threshold, not a verified
    probability of safety or of human placement suitability.
    """
    from src.policies.brand_policies import evaluate_policy

    return evaluate_policy(probabilities, policy, threshold=threshold)

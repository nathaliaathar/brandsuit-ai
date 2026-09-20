# Streamlit app for BrandSuit AI: Live Decision + Project Story.
# Never fits. Category probabilities come from the saved model; YES/NO
# comes from src/policies/brand_policies.py.

import html
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.policies.brand_policies import (
    BRAND_POLICIES,
    VALIDATED_SOURCE,
    evaluate_policy,
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
REPO_URL = "https://github.com/nathaliaathar/brandsuit-ai"

EXAMPLE_TITLE = "Milo Takes Calls From Infowars Listeners"
EXAMPLE_DESCRIPTION = (
    "Political talk-show host takes live calls and discusses current events "
    "with Infowars listeners."
)

st.set_page_config(
    page_title="BrandSuit AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def esc(value):
    """Everything injected into markdown HTML goes through this."""
    return html.escape(str(value))


@st.cache_resource
def load_artifacts():
    """Load once per session. Returns (vec, model) or (None, None)."""
    tfidf_path = MODEL_DIR / "category_tfidf.joblib"
    model_path = MODEL_DIR / "category_logreg.joblib"
    if not tfidf_path.exists() or not model_path.exists():
        return None, None
    try:
        return joblib.load(tfidf_path), joblib.load(model_path)
    except Exception:  # never show a stack trace to the user
        return None, None


def predict_probabilities(vec, model, text):
    """transform only — the vocabulary was learned during training."""
    scores = model.predict_proba(vec.transform([text]))[0]
    # Class order comes from the estimator, never from a hardcoded list.
    return dict(zip(model.classes_, (float(s) for s in scores)))


@st.cache_data
def top_words_by_category(_vec, _model, top_n=6):
    """Words with the largest positive logistic coefficient per class.

    Same idea as petal_width for iris: these are the features the model
    leans on when it picks a category. Coefficients come from the fitted
    LogReg; feature names come from the fitted TF-IDF vocabulary.
    """
    feature_names = _vec.get_feature_names_out()
    result = {}
    for class_idx, class_name in enumerate(_model.classes_):
        coefs = _model.coef_[class_idx]
        top_idx = coefs.argsort()[::-1][:top_n]
        result[class_name] = [
            (str(feature_names[i]), float(coefs[i])) for i in top_idx
        ]
    return result


# Soft palette per class — informational blues/greens, never decorative red.
CATEGORY_COLORS = {
    "Entertainment & Culture": "#0369a1",
    "Lifestyle & Interests": "#0e7490",
    "Education, Science & Technology": "#15803d",
    "News, Politics & Society": "#1d4ed8",
    "Sports": "#0f766e",
    "Gaming": "#4338ca",
}

st.markdown(
    """
    <style>
    /* Global typography and colours */
    .stApp {color: #0f172a; background: #ffffff;}
    [data-testid="stHeader"] {display: none;}
    .block-container {padding: 1rem 2rem 2rem; max-width: 1600px;}
    p, li, label p {color: #334155;}

    /* Header */
    .main-header {
        background-color: #0f172a;
        color: #ffffff;
        padding: 1.15rem 1.75rem;
        border-radius: 10px;
        margin-bottom: 1.1rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 1.25rem;
        flex-wrap: wrap;
    }
    .header-title {font-size: 1.5rem; font-weight: 700; margin: 0; line-height: 1.25;}
    .header-subtitle {font-size: 0.9rem; color: #cbd5e1; margin-top: 4px; line-height: 1.35;}
    .header-links {margin-top: 6px;}
    .header-links a {color: #7dd3fc; font-size: 0.78rem; text-decoration: none;}
    .badge-pill {
        background: #1e293b;
        color: #38bdf8;
        border: 1px solid #38bdf8;
        padding: 6px 14px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        white-space: nowrap;
    }

    /* Tabs */
    button[data-baseweb="tab"] {font-size: 0.92rem; font-weight: 600;}

    /* Metric cards */
    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        height: 100%;
    }
    .metric-card h4 {
        color: #475569;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin: 0 0 0.5rem 0;
        line-height: 1.35;
    }
    .metric-card .value {font-size: 1.75rem; font-weight: 800; color: #0f172a; line-height: 1.2;}
    .metric-card .caption {font-size: 0.8rem; color: #475569; margin-top: 0.4rem; line-height: 1.4;}

    /* Verdict banners */
    .decision-banner {
        border-radius: 6px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 1.1rem;
        color: #ffffff;
    }
    .decision-banner-yes {background: #143e26; border: 1px solid #166534;}
    .decision-banner-no {background: #4a131a; border: 1px solid #7f1d1d;}
    .decision-banner .status-row {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        gap: 1rem;
        flex-wrap: wrap;
        border-bottom: 1px solid rgba(255, 255, 255, 0.15);
        padding-bottom: 0.6rem;
        margin-bottom: 0.75rem;
    }
    .decision-banner h3 {font-size: 1.25rem; font-weight: 700; margin: 0; line-height: 1.25;}
    .decision-banner-yes h3 {color: #bbf7d0;}
    .decision-banner-no h3 {color: #fecaca;}
    .decision-banner .compare {
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
        font-weight: 600;
        font-size: 0.95rem;
        color: #ffffff;
    }
    .decision-banner p {color: #ffffff; margin: 0; font-size: 0.85rem; line-height: 1.45;}

    /* Pipeline */
    .pipeline-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 8px;
        margin-bottom: 1.15rem;
    }
    .pipeline-cell {
        background: #ffffff;
        border: 1px solid #cbd5e1;
        border-radius: 4px;
        padding: 0.6rem 0.5rem;
        text-align: center;
    }
    .pipeline-cell .label {
        font-size: 0.72rem;
        font-weight: 700;
        color: #475569;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .pipeline-cell .val {
        font-size: 0.78rem;
        color: #0f172a;
        font-weight: 500;
        margin-top: 2px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    /* Insight callout */
    .insight-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-left: 3px solid #0f172a;
        padding: 0.85rem 1rem;
        font-size: 0.82rem;
        color: #334155;
        line-height: 1.45;
        border-radius: 0 4px 4px 0;
        margin-top: 1rem;
    }

    /* Primary button */
    div.stButton > button,
    div.stButton > button p,
    div.stButton > button span,
    button[kind="primary"],
    [data-testid="stBaseButton-primary"] {
        background-color: #0f172a !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 4px;
        font-weight: 700 !important;
        letter-spacing: 0.01em;
        padding: 0.55rem 1rem;
    }
    div.stButton > button:hover,
    button[kind="primary"]:hover,
    [data-testid="stBaseButton-primary"]:hover {
        background-color: #1e293b !important;
        color: #ffffff !important;
    }

    /* Allowed-category tags */
    .cat-tag {
        display: inline-block;
        background: #f1f5f9;
        border: 1px solid #cbd5e1;
        color: #1e293b;
        padding: 3px 9px;
        border-radius: 4px;
        font-size: 0.76rem;
        font-weight: 600;
        margin: 0 4px 4px 0;
    }
    .policy-line {font-size: 0.85rem; color: #334155; margin-top: 0.5rem; line-height: 1.5;}
    .source-validated {color: #0369a1; font-weight: 700;}
    .source-illustrative {color: #475569; font-weight: 600;}
    .policy-depends {
        background: #eff6ff;
        border-left: 4px solid #0369a1;
        border-radius: 0 8px 8px 0;
        padding: 0.7rem 1rem;
        font-size: 0.85rem;
        color: #1e3a5f;
        line-height: 1.5;
        margin: 0.7rem 0 0.4rem;
    }
    .profile-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 0.6rem;
        margin: 0.6rem 0 0.8rem;
    }
    @media (max-width: 1280px) {
        .profile-grid {grid-template-columns: repeat(2, 1fr);}
    }
    .profile-card {
        background: #f8fafc;
        border: 1px solid #cbd5e1;
        border-radius: 8px;
        padding: 0.75rem 0.8rem;
    }
    .profile-card h4 {margin: 0 0 0.2rem 0; font-size: 0.88rem; color: #0f172a; line-height: 1.3;}
    .profile-card .meta {font-size: 0.75rem; color: #334155; margin-bottom: 0.4rem; line-height: 1.4;}
    .profile-card .cats {font-size: 0.72rem; color: #475569; line-height: 1.4;}
    .profile-card.active {border-color: #0369a1; background: #eff6ff;}

    /* Probability rows */
    .prob-name {font-size: 0.86rem; color: #0f172a; font-weight: 600; line-height: 1.9;}
    .prob-name-muted {font-size: 0.86rem; color: #64748b; line-height: 1.9;}
    .prob-value {font-size: 0.86rem; color: #334155; line-height: 1.9; text-align: right;}

    /* Story cards */
    .story-card {
        background: #f8fafc;
        border-left: 4px solid #0284c7;
        padding: 1rem 1.25rem;
        margin-bottom: 1rem;
        border-radius: 0 8px 8px 0;
    }
    .story-card h4 {margin: 0 0 0.4rem 0; color: #0f172a; font-size: 1rem; line-height: 1.35;}
    .story-card p {margin: 0; color: #334155; font-size: 0.9rem; line-height: 1.5;}
    .note-box {
        background: #f1f5f9;
        padding: 0.9rem 1.2rem;
        border-radius: 8px;
        font-size: 0.88rem;
        color: #334155;
        line-height: 1.55;
        margin-top: 0.5rem;
    }

    /* Data storytelling */
    .story-step {
        display: flex;
        align-items: baseline;
        gap: 0.6rem;
        margin: 1.1rem 0 0.45rem;
    }
    .story-step .num {
        background: #0f172a;
        color: #ffffff;
        font-size: 0.75rem;
        font-weight: 700;
        border-radius: 999px;
        padding: 3px 10px;
    }
    .story-step .txt {font-size: 0.98rem; font-weight: 700; color: #0f172a;}
    .story-step .sub {font-size: 0.85rem; color: #475569;}
    .funnel {
        display: grid;
        grid-template-columns: 1fr auto 1fr auto 1fr auto 1fr;
        gap: 0.4rem;
        align-items: stretch;
    }
    .funnel-box {
        background: #f8fafc;
        border: 1px solid #cbd5e1;
        border-radius: 8px;
        padding: 0.7rem 0.6rem;
        text-align: center;
    }
    .funnel-box .n {font-size: 1.45rem; font-weight: 800; color: #0f172a; line-height: 1.15;}
    .funnel-box .l {font-size: 0.78rem; color: #334155; margin-top: 0.2rem; line-height: 1.35;}
    .funnel-box .w {font-size: 0.72rem; color: #64748b; margin-top: 0.2rem; line-height: 1.3;}
    .funnel-arrow {display: flex; align-items: center; color: #94a3b8; font-weight: 700;}

    .dist-row {display: flex; align-items: center; gap: 0.7rem; margin-bottom: 0.75rem;}
    .dist-row .nm {flex: 0 0 210px; font-size: 0.85rem; color: #0f172a;}
    .dist-row .tr {
        flex: 1;
        height: 20px;
        background: #f1f5f9;
        border-radius: 5px;
        position: relative;
        margin-top: 16px;
        overflow: visible;
    }
    .dist-row .fl {height: 20px; border-radius: 5px; position: relative; min-width: 8px;}
    .dist-row .fl .bar-top-label {
        position: absolute;
        bottom: 100%;
        right: 0;
        transform: translateY(-3px);
        font-size: 0.72rem;
        font-weight: 700;
        color: #0f172a;
        line-height: 1;
        white-space: nowrap;
    }
    .dist-row .pc {flex: 0 0 150px; font-size: 0.85rem; color: #334155; font-weight: 600;}

    .metric-journey {display: flex; align-items: stretch; gap: 0.5rem; margin-top: 0.6rem; overflow: visible;}
    .journey-col {flex: 1; display: flex; flex-direction: column; overflow: visible;}
    .journey-bars {
        display: flex;
        align-items: stretch;
        gap: 8px;
        height: 170px;
        padding-top: 22px;
        box-sizing: border-box;
        border-bottom: 1px solid #cbd5e1;
        overflow: visible;
    }
    .journey-bar {
        flex: 1;
        height: 100%;
        display: flex;
        align-items: flex-end;
        position: relative;
    }
    .bar-fill {
        width: 100%;
        min-height: 3px;
        border-radius: 4px 4px 0 0;
        position: relative;
    }
    .bar-top-label {
        position: absolute;
        bottom: 100%;
        left: 50%;
        transform: translateX(-50%);
        font-size: 0.72rem;
        font-weight: 700;
        color: #0f172a;
        line-height: 1;
        margin-bottom: 4px;
        white-space: nowrap;
    }
    .journey-label {font-size: 0.72rem; color: #334155; text-align: center; margin-top: 0.45rem; line-height: 1.3;}
    .legend-row {display: flex; gap: 1rem; font-size: 0.78rem; color: #334155; margin-top: 0.5rem;}
    .legend-row span b {display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 5px;}

    .case-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.82rem;
        margin: 0.6rem 0 0.3rem;
    }
    .case-table th {
        text-align: left;
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #475569;
        border-bottom: 1px solid #cbd5e1;
        padding: 0.4rem 0.6rem;
        font-weight: 700;
    }
    .case-table td {
        padding: 0.5rem 0.6rem;
        border-bottom: 1px solid #e2e8f0;
        color: #0f172a;
        vertical-align: middle;
    }
    .case-table td.num {text-align: center; font-variant-numeric: tabular-nums; font-weight: 700;}
    .pill {
        display: inline-block;
        padding: 2px 9px;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 700;
        white-space: nowrap;
    }
    .pill-bad {background: #fee2e2; color: #991b1b;}
    .pill-good {background: #dcfce7; color: #166534;}
    .change-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 0.7rem;
        margin: 0.7rem 0 0.4rem;
    }
    .change-card {
        background: #ffffff;
        border: 1px solid #cbd5e1;
        border-top: 3px solid #0369a1;
        border-radius: 8px;
        padding: 0.85rem 1rem;
    }
    .change-card .kicker {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #0369a1;
        font-weight: 700;
    }
    .change-card h4 {margin: 0.25rem 0 0.35rem 0; font-size: 0.92rem; color: #0f172a; line-height: 1.3;}
    .change-card p {margin: 0 0 0.35rem 0; font-size: 0.82rem; color: #334155; line-height: 1.5;}
    .change-card .plain {
        font-size: 0.8rem;
        color: #1e3a5f;
        background: #eff6ff;
        border-radius: 6px;
        padding: 0.45rem 0.6rem;
        line-height: 1.45;
    }

    .word-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 0.75rem;
        margin: 0.7rem 0 0.4rem;
    }
    @media (max-width: 1280px) {
        .word-grid {grid-template-columns: repeat(2, 1fr);}
    }
    .word-panel {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.75rem 0.9rem 0.85rem;
    }
    .word-panel h4 {
        margin: 0 0 0.55rem 0;
        font-size: 0.82rem;
        color: #0f172a;
        line-height: 1.3;
        border-bottom: 2px solid #cbd5e1;
        padding-bottom: 0.35rem;
    }
    .word-row {
        display: grid;
        grid-template-columns: 88px 1fr 42px;
        gap: 0.45rem;
        align-items: center;
        margin-bottom: 0.28rem;
    }
    .word-row .w {font-size: 0.78rem; color: #0f172a; font-weight: 600; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;}
    .word-row .track {height: 10px; background: #f1f5f9; border-radius: 4px;}
    .word-row .fill {height: 10px; border-radius: 4px; min-width: 3px;}
    .word-row .wt {font-size: 0.72rem; color: #475569; text-align: right; font-variant-numeric: tabular-nums;}

    .takeaway {
        background: #ecfdf5;
        border-left: 4px solid #15803d;
        border-radius: 0 8px 8px 0;
        padding: 0.85rem 1.1rem;
        font-size: 0.88rem;
        color: #14532d;
        line-height: 1.55;
    }
    .caveat {
        background: #fef2f2;
        border-left: 4px solid #b91c1c;
        border-radius: 0 8px 8px 0;
        padding: 0.85rem 1.1rem;
        font-size: 0.88rem;
        color: #7f1d1d;
        line-height: 1.55;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="main-header">
        <div>
            <div class="header-title">BrandSuit AI</div>
            <div class="header-subtitle">Contextual suitability for YouTube advertising</div>
            <div class="header-links"><a href="{REPO_URL}" target="_blank">View repository ↗</a></div>
        </div>
        <div class="badge-pill">Suitability, not safety detection</div>
    </div>
    """,
    unsafe_allow_html=True,
)

vec, model = load_artifacts()
if vec is None or model is None:
    st.error(
        "The trained classifier is not available. From this folder run "
        "`py src/export_model.py` to rebuild models/category_tfidf.joblib "
        "and models/category_logreg.joblib."
    )
    st.stop()

tab_live, tab_story = st.tabs(
    ["Live Decision", "Project Story & Architecture"]
)

# ==========================================
# TAB 1 — LIVE DECISION
# ==========================================
with tab_live:
    st.subheader("Should this brand advertise next to this video?")
    st.caption(
        "Text-only category prediction, then a brand-specific allow-list policy. "
        "The model scores stay the same; switching the advertiser profile can flip YES to NO."
    )

    col_input, col_result = st.columns([1, 1.2], gap="large")

    with col_input:
        st.markdown("### 1. Video text and advertiser policy")

        video_title = st.text_input(
            "Video title",
            value=EXAMPLE_TITLE,
            placeholder="Paste the YouTube video title...",
        )
        video_desc = st.text_area(
            "Video description",
            value=EXAMPLE_DESCRIPTION,
            height=110,
            placeholder="Paste the video description...",
        )
        advertiser = st.selectbox(
            "Advertiser profile — this chooses the policy",
            list(BRAND_POLICIES),
            index=0,
            help=(
                "The classifier does not decide YES or NO. Each advertiser profile "
                "has its own allowed categories and confidence threshold."
            ),
        )

        policy = BRAND_POLICIES[advertiser]
        allowed_cats = policy["allowed_categories"]

        st.markdown(
            f"**Allowed for {esc(advertiser)}** "
            f"(this profile's allow-list, not the model's):"
        )
        st.markdown(
            "".join(f'<span class="cat-tag">{esc(cat)}</span>' for cat in allowed_cats),
            unsafe_allow_html=True,
        )

        # Only one threshold came from an experiment; the rest are starting points.
        is_validated = policy["threshold_source"] == VALIDATED_SOURCE
        source_class = "source-validated" if is_validated else "source-illustrative"
        st.markdown(
            f'<div class="policy-line">'
            f'Policy for <b>{esc(advertiser)}</b>: '
            f'<b>{esc(policy["risk_profile"])}</b> · '
            f'threshold <b>{policy["threshold"]:.2f}</b> · '
            f'<span class="{source_class}">{esc(policy["threshold_source"])}</span>'
            f"</div>",
            unsafe_allow_html=True,
        )
        st.caption(
            "Change the advertiser profile to load a different allow-list and threshold. "
            "The video text is not re-scored until you click Analyze placement."
        )

        analyze_btn = st.button(
            "Analyze placement", type="primary", use_container_width=True
        )

    # Training text was title + " " + description, then stripped.
    text = f"{video_title} {video_desc}".strip()

    # Predict only on click (or first load). Switching advertiser re-uses the
    # stored probabilities, so the policy updates without a new inference.
    if analyze_btn or "probabilities" not in st.session_state:
        if not text:
            st.session_state.pop("probabilities", None)
            st.session_state["input_error"] = (
                "Enter a title or a description before analyzing the placement."
            )
        else:
            try:
                st.session_state["probabilities"] = predict_probabilities(vec, model, text)
                st.session_state["scored_text"] = text
                st.session_state.pop("input_error", None)
            except Exception:
                st.session_state.pop("probabilities", None)
                st.session_state["input_error"] = (
                    "This text could not be scored. Try a different title or description."
                )

    probabilities = st.session_state.get("probabilities")
    input_error = st.session_state.get("input_error")

    with col_result:
        st.markdown("### 2. Placement decision")

        st.markdown(
            f"""
            <div class="pipeline-grid">
                <div class="pipeline-cell">
                    <div class="label">1. Input</div>
                    <div class="val">Title + description</div>
                </div>
                <div class="pipeline-cell">
                    <div class="label">2. Features</div>
                    <div class="val">TF-IDF n-grams</div>
                </div>
                <div class="pipeline-cell">
                    <div class="label">3. Model</div>
                    <div class="val">Balanced LogReg</div>
                </div>
                <div class="pipeline-cell">
                    <div class="label">4. Profile</div>
                    <div class="val">{esc(advertiser)} · {policy['threshold']:.2f}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if input_error:
            st.error(input_error)

        if probabilities:
            p_allow, is_suitable = evaluate_policy(probabilities, policy)
            ranked = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
            predicted = ranked[0][0]
            threshold = policy["threshold"]
            allowed_set = set(allowed_cats)

            if is_suitable:
                banner, word = "decision-banner-yes", "YES · Suitable for this profile"
                symbol, explanation = "&ge;", (
                    f"The combined probability of categories allowed by "
                    f"<b>{esc(advertiser)}</b> clears this profile's threshold. "
                    "The placement is approved."
                )
            else:
                banner, word = "decision-banner-no", "NO · Not suitable for this profile"
                symbol, explanation = "&lt;", (
                    f"The combined probability of categories allowed by "
                    f"<b>{esc(advertiser)}</b> is below this profile's threshold. "
                    "The placement is withheld."
                )

            st.markdown(
                f"""
                <div class="decision-banner {banner}">
                    <div class="status-row">
                        <h3>{word}</h3>
                        <span class="compare">
                            p_allow: {p_allow * 100:.1f}% {symbol} threshold: {threshold * 100:.1f}%
                        </span>
                    </div>
                    <p><b>Predicted category:</b> {esc(predicted)}
                       &nbsp;&bull;&nbsp; <b>Advertiser profile:</b> {esc(advertiser)}<br>
                       {explanation}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if st.session_state.get("scored_text") != text:
                st.info("The text changed. Click **Analyze placement** to score it again.")

            st.markdown("**Category probabilities (model output — same for every profile)**")
            for category, score in ranked:
                style = "prob-name" if category in allowed_set else "prob-name-muted"
                col_label, col_bar, col_val = st.columns([2.5, 5, 1])
                col_label.markdown(
                    f'<div class="{style}">{esc(category)}</div>', unsafe_allow_html=True
                )
                col_bar.progress(min(score, 1.0))
                col_val.markdown(
                    f'<div class="prob-value">{score * 100:.1f}%</div>',
                    unsafe_allow_html=True,
                )

        st.markdown(
            "<div class='insight-card'><b>Model output &ne; business decision:</b> "
            "the classifier predicts context once. YES / NO comes from the selected "
            f"advertiser profile (<b>{esc(advertiser)}</b>): its allow-list and its "
            "threshold. Switch the profile to evaluate a different risk tolerance on "
            "the same probabilities.</div>",
            unsafe_allow_html=True,
        )

# ==========================================
# TAB 2 — PROJECT STORY (data storytelling)
# ==========================================

# Measured on the development holdout (see DECISIONS.md D3/D4), recomputed
# after Pets & Animals was remapped to Education, Science & Technology.
CLASS_DISTRIBUTION = [
    ("Entertainment & Culture", 3287, 658),
    ("Lifestyle & Interests", 1223, 245),
    ("Education, Science & Technology", 768, 154),
    ("News, Politics & Society", 519, 104),
    ("Sports", 451, 90),
    ("Gaming", 103, 20),
]
TOTAL_VIDEOS = 6351

EXPERIMENT_JOURNEY = [
    ("Majority baseline", "Always guesses Entertainment", 0.518, 0.114, 0.00, None),
    ("Unigram, no weights", "First real model", 0.780, 0.678, 0.75, 6),
    ("Unigram, balanced", "class_weight=balanced", 0.800, 0.782, 0.90, 0),
    ("Bigram, balanced", "Deployed + policy 0.55", 0.788, 0.759, 0.88, 0),
]

# The six real News videos that the first model approved for a Kids & Family ad,
# with p_allow before (unigram, no weights) and after (deployed model).
# Produced by error_analysis_examples.py on the development holdout.
LEAKED_NEWS_CASES = [
    ("Veteran Congressman John Conyers Announces He Is Retiring | The View", 0.67, 0.39),
    ("Rose McGowan Shares Her Thoughts On 'Time's Up' Movement | The View", 0.67, 0.43),
    ("Search continues for missing Argentine submarine with 44 crew members", 0.63, 0.40),
    ("Additional Remains Of Miami Gardens Soldier Recovered", 0.56, 0.34),
    ("Pepsi Uses Aborted Babies to Flavor Test Soda - Alex Jones", 0.56, 0.31),
    ("Jerry Van Dyke, star of 'Coach', dead at 86", 0.55, 0.32),
]


def step_header(number, title, subtitle):
    st.markdown(
        f'<div class="story-step"><span class="num">{number}</span>'
        f'<span class="txt">{title}</span>'
        f'<span class="sub">{subtitle}</span></div>',
        unsafe_allow_html=True,
    )


with tab_story:
    st.subheader("How this dataset became an advertising decision")
    st.caption(
        "Follow the data: 40,949 raw rows → one model → a business rule. "
        "Every number below was measured on the development holdout."
    )

    # ---------- 1. From raw rows to labelled videos ----------
    step_header(1, "From raw rows to labelled videos", "one row per video, not per trending day")
    st.markdown(
        """
        <div class="funnel">
          <div class="funnel-box">
            <div class="n">40,949</div><div class="l">raw CSV rows</div>
            <div class="w">a video repeats on every trending day</div>
          </div>
          <div class="funnel-arrow">→</div>
          <div class="funnel-box">
            <div class="n">6,351</div><div class="l">unique video_id</div>
            <div class="w">deduplicated before any split</div>
          </div>
          <div class="funnel-arrow">→</div>
          <div class="funnel-box">
            <div class="n">16</div><div class="l">YouTube categories</div>
            <div class="w">joined from US_category_id.json</div>
          </div>
          <div class="funnel-arrow">→</div>
          <div class="funnel-box">
            <div class="n">6</div><div class="l">suitability classes</div>
            <div class="w">bucketed for advertiser policies</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        "Deduplicating first matters: the same video trending 8 days would otherwise leak "
        "between train and test and inflate every score."
    )

    # ---------- 2. Class distribution ----------
    step_header(2, "The classes are heavily imbalanced", "share of the 6,351 unique videos")
    rows = ""
    for name, total, _ in CLASS_DISTRIBUTION:
        share = total / TOTAL_VIDEOS * 100
        colour = "#0369a1" if share >= 50 else ("#b91c1c" if share < 2 else "#38bdf8")
        rows += (
            f'<div class="dist-row"><div class="nm">{esc(name)}</div>'
            f'<div class="tr"><div class="fl" style="width:{share:.1f}%;background:{colour};">'
            f'<span class="bar-top-label">{share:.1f}%</span></div></div>'
            f'<div class="pc">{total:,} videos</div></div>'
        )
    st.markdown(rows, unsafe_allow_html=True)
    st.markdown(
        '<div class="caveat">One class owns half the dataset and Gaming has only '
        "103 videos (1.6%). Any metric that averages over <i>videos</i> instead of "
        "<i>classes</i> will be dominated by Entertainment.</div>",
        unsafe_allow_html=True,
    )

    # ---------- 3. Baseline ----------
    step_header(3, "The dumb baseline sets the bar", "always answer with the most common class")
    base_a, base_b, base_c = st.columns(3)
    base_a.markdown(
        '<div class="metric-card"><h4>Baseline accuracy</h4>'
        '<div class="value">51.8%</div>'
        '<div class="caption">always Entertainment &amp; Culture</div></div>',
        unsafe_allow_html=True,
    )
    base_b.markdown(
        '<div class="metric-card"><h4>Baseline macro-F1</h4>'
        '<div class="value" style="color:#b91c1c;">0.114</div>'
        '<div class="caption">five of six classes score zero</div></div>',
        unsafe_allow_html=True,
    )
    base_c.markdown(
        '<div class="metric-card"><h4>Baseline News recall</h4>'
        '<div class="value" style="color:#b91c1c;">0.0%</div>'
        '<div class="caption">it never flags politics</div></div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "This is why accuracy alone is a trap: a model that understands nothing "
        "already scores 51.8%. Macro-F1 exposes it instantly."
    )

    # ---------- 4. Split ----------
    step_header(4, "Split before touching the vocabulary", "80 / 20 stratified, random_state=42")
    st.markdown(
        """
        <div class="funnel">
          <div class="funnel-box">
            <div class="n">5,080</div><div class="l">training videos</div>
            <div class="w">TF-IDF is fitted here only</div>
          </div>
          <div class="funnel-arrow">→</div>
          <div class="funnel-box">
            <div class="n">1,271</div><div class="l">holdout videos</div>
            <div class="w">transform only, never fit</div>
          </div>
          <div class="funnel-arrow">→</div>
          <div class="funnel-box">
            <div class="n">5,000</div><div class="l">TF-IDF features</div>
            <div class="w">most frequent uni + bigrams</div>
          </div>
          <div class="funnel-arrow">→</div>
          <div class="funnel-box">
            <div class="n">6</div><div class="l">predicted classes</div>
            <div class="w">probabilities summing to 1</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---------- 5. Words the model learned (iris petal analogy) ----------
    step_header(
        5,
        "What the model studied — the words that separate the classes",
        "like petal_width for iris flowers, these n-grams pull a video toward a category",
    )
    st.markdown(
        "In the iris notebook, `petal_length` and `petal_width` separate the flower "
        "species better than sepal measures. Here the features are **words and word "
        "pairs** from title + description. After training, each logistic coefficient "
        "says: *if this word shows up, lean toward this category*."
    )
    st.markdown(
        '<div class="policy-depends"><b>How to read the chart.</b> Each panel is one '
        "category. The bars are the six words with the <b>largest positive "
        "coefficients</b> for that class in the deployed model "
        "(TF-IDF + balanced LogReg). Longer bar = stronger pull. "
        "This is the text version of a pairplot: you can see which signals the "
        "model trusts.</div>",
        unsafe_allow_html=True,
    )

    top_words = top_words_by_category(vec, model, top_n=6)
    global_max = max(w for pairs in top_words.values() for _, w in pairs) or 1.0
    panels = ""
    for class_name, pairs in top_words.items():
        colour = CATEGORY_COLORS.get(class_name, "#0369a1")
        rows_html = ""
        for word, weight in pairs:
            width_pct = max(weight / global_max * 100, 2.0)
            rows_html += (
                f'<div class="word-row"><div class="w" title="{esc(word)}">{esc(word)}</div>'
                f'<div class="track"><div class="fill" style="width:{width_pct:.1f}%;'
                f'background:{colour};"></div></div>'
                f'<div class="wt">{weight:.2f}</div></div>'
            )
        panels += (
            f'<div class="word-panel"><h4 style="border-color:{colour};">{esc(class_name)}</h4>'
            f"{rows_html}</div>"
        )
    st.markdown(f'<div class="word-grid">{panels}</div>', unsafe_allow_html=True)
    st.caption(
        "Weights are logistic regression coefficients on TF-IDF features "
        "(fit on the 5,080 training videos only). Same scale across panels."
    )
    st.markdown(
        '<div class="takeaway"><b>What a 5-year-old version sounds like:</b> '
        "the model keeps a cheat sheet. If it sees <i>nba</i> or <i>nfl</i>, it thinks "
        "Sports. If it sees <i>nintendo</i> or <i>fortnite</i>, it thinks Gaming. "
        "If it sees <i>trump</i> or <i>washingtonpost</i>, it thinks News. "
        "Entertainment leans on <i>music</i>, <i>movie</i>, <i>netflix</i>.</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="caveat"><b>Also an honest quirk:</b> Education, Science &amp; '
        "Technology lists <i>cat</i>, <i>dog</i>, <i>animals</i> near the top because "
        "Pets &amp; Animals videos were remapped into that bucket. The model is not "
        "wrong about the words — the label grouping pulled pet vocabulary into "
        "Education. That is the kind of thing an interviewer should ask about.</div>",
        unsafe_allow_html=True,
    )

    # ---------- 6. The failure we found ----------
    step_header(
        6,
        "Then we read the mistakes, one by one",
        "the first model confused politics with entertainment",
    )
    st.markdown(
        "The first trained model looked fine on paper: 78.0% accuracy. "
        "So we listed every video it got wrong. A pattern jumped out. "
        "**Of the 104 real News videos in the holdout, 24 were labelled "
        "Entertainment & Culture** — a retiring congressman, a submarine search, "
        "a soldier's remains being recovered."
    )
    st.markdown(
        '<div class="caveat"><b>Why that is a product bug, not just a metric.</b> '
        "Kids & Family Brand allows Sports and Entertainment. When a news video is "
        "scored as entertainment, its <code>p_allow</code> climbs above 0.55 and the "
        "policy says YES. Six real videos passed that gate — a cereal ad would have "
        "run next to them.</div>",
        unsafe_allow_html=True,
    )

    before_rows = "".join(
        f"<tr><td>{esc(title)}</td>"
        f'<td class="num">{p_before:.2f}</td>'
        f'<td><span class="pill pill-bad">APPROVED for kids ad</span></td></tr>'
        for title, p_before, _ in LEAKED_NEWS_CASES
    )
    st.markdown(
        "<table class='case-table'><thead><tr>"
        "<th>Real News video in the holdout</th>"
        "<th style='text-align:center;'>p_allow (first model)</th>"
        "<th>Kids &amp; Family decision at 0.55</th>"
        f"</tr></thead><tbody>{before_rows}</tbody></table>",
        unsafe_allow_html=True,
    )

    # ---------- 7. What we changed ----------
    step_header(7, "So we changed three things", "each fix answers one thing we saw in the errors")
    st.markdown(
        """
        <div class="change-grid">
          <div class="change-card">
            <div class="kicker">Fix 1 · Class weights</div>
            <h4>class_weight="balanced"</h4>
            <p>Entertainment is 51.8% of the data, so guessing it was almost always
               a safe bet. Weighting makes one mistake on a rare class cost as much
               as many mistakes on the common one.</p>
            <div class="plain">Like a teacher who stops giving credit for the
               answer everybody already knows.</div>
          </div>
          <div class="change-card">
            <div class="kicker">Fix 2 · Bigrams</div>
            <h4>ngram_range=(1, 2)</h4>
            <p>Talk-show wording fooled the model: single words like <i>calls</i>,
               <i>show</i> and <i>host</i> look like entertainment. Word pairs let it
               read <i>white house</i> or <i>press conference</i> as one signal.</p>
            <div class="plain">"Hot" and "dog" mean something different from
               "hot dog".</div>
          </div>
          <div class="change-card">
            <div class="kicker">Fix 3 · Confidence policy</div>
            <h4>p_allow ≥ threshold</h4>
            <p>The old rule trusted the single top guess. The new rule adds up the
               probability of the categories a brand allows and requires that sum to
               clear the brand's threshold.</p>
            <div class="plain">Not "what do you think?" but "how sure are you?"</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---------- 8. Metric journey ----------
    step_header(8, "What those fixes did to the metrics", "measured on the same 1,271 videos")
    journey = ""
    for name, note, _acc, macro, news_recall, approved in EXPERIMENT_JOURNEY:
        approved_txt = "—" if approved is None else f"Kids & Family profile: {approved} News approved"
        journey += (
            '<div class="journey-col"><div class="journey-bars">'
            f'<div class="journey-bar"><div class="bar-fill" style="height:{max(macro * 100, 1.5):.1f}%;'
            f'background:#0369a1;"><span class="bar-top-label">{macro * 100:.1f}%</span></div></div>'
            f'<div class="journey-bar"><div class="bar-fill" style="height:{max(news_recall * 100, 1.5):.1f}%;'
            f'background:#15803d;"><span class="bar-top-label">{news_recall * 100:.1f}%</span></div></div>'
            "</div>"
            f'<div class="journey-label"><b>{esc(name)}</b><br>{esc(note)}<br>{approved_txt}</div></div>'
        )
    st.markdown(f'<div class="metric-journey">{journey}</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="legend-row">'
        '<span><b style="background:#0369a1;"></b>Macro-F1 — average F1 of all 6 classes (shown as %)</span>'
        '<span><b style="background:#15803d;"></b>News recall — share of real News videos the model catches</span>'
        "</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        "The green bars are a model metric, not a brand policy. "
        "Kids & Family Brand is only one of four advertiser profiles (step 10); "
        "News recall matters to it because that is the class it blocks."
    )
    st.markdown(
        '<div class="takeaway">Accuracy barely moved (78.0% → 80.0%), and that is the '
        "interesting part. Macro-F1 jumped from 67.8% to 78.2% and News recall from 75.0% "
        "to 90.0%, so the gain came entirely from the rare classes the first model was "
        "ignoring. Gaming recall moved the most: 20.0% → 90.0%. An accuracy-only report "
        "would have shown almost nothing and we would have shipped the broken model.</div>",
        unsafe_allow_html=True,
    )

    # ---------- 9. Back to the same six videos ----------
    step_header(
        9,
        "Back to those six videos",
        "same holdout, same threshold 0.55, after the three fixes",
    )
    after_rows = "".join(
        f"<tr><td>{esc(title)}</td>"
        f'<td class="num">{p_before:.2f}</td>'
        f'<td class="num">{p_after:.2f}</td>'
        f'<td><span class="pill pill-good">Withheld</span></td></tr>'
        for title, p_before, p_after in LEAKED_NEWS_CASES
    )
    st.markdown(
        "<table class='case-table'><thead><tr>"
        "<th>Real News video in the holdout</th>"
        "<th style='text-align:center;'>p_allow before</th>"
        "<th style='text-align:center;'>p_allow after</th>"
        "<th>Kids &amp; Family decision at 0.55</th>"
        f"</tr></thead><tbody>{after_rows}</tbody></table>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="takeaway"><b>All six now fall below the line.</b> The model also '
        "recovers more of the class overall: News caught at argmax went from 78 to 92 "
        "of 104 videos, and blocked-category approvals for Kids & Family went from 6 "
        "to 0. Note what actually moved the needle — the model became less certain that "
        "politics is entertainment, and the policy turned that lower confidence into a "
        "NO.</div>",
        unsafe_allow_html=True,
    )

    # ---------- 10. Policy layer ----------
    step_header(
        10,
        "The model score is not the decision",
        "each advertiser profile has its own allow-list and threshold",
    )
    st.markdown(
        '<div class="policy-depends"><b>Policy depends on the advertiser profile.</b> '
        "The classifier always returns the same six probabilities. YES / NO is computed "
        "afterwards: <code>p_allow = sum of P(allowed categories for that profile)</code>, "
        "then compared with that profile's threshold. Switching the profile can approve "
        "a video that another profile withholds.</div>",
        unsafe_allow_html=True,
    )

    profile_cards = ""
    for name, spec in BRAND_POLICIES.items():
        cats = " · ".join(spec["allowed_categories"])
        source = spec["threshold_source"]
        active = " active" if spec["threshold_source"] == VALIDATED_SOURCE else ""
        profile_cards += (
            f'<div class="profile-card{active}"><h4>{esc(name)}</h4>'
            f'<div class="meta">{esc(spec["risk_profile"])} · threshold {spec["threshold"]:.2f}<br>'
            f"{esc(source)}</div>"
            f'<div class="cats">Allows: {esc(cats)}</div></div>'
        )
    st.markdown(f'<div class="profile-grid">{profile_cards}</div>', unsafe_allow_html=True)
    st.caption(
        "Highlighted card: only Kids & Family at 0.55 is a validation candidate. "
        "The other three thresholds are illustrative starting points."
    )

    pol_a, pol_b, pol_c = st.columns(3)
    pol_a.markdown(
        '<div class="metric-card"><h4>Kids &amp; Family · News approved before</h4>'
        '<div class="value" style="color:#b91c1c;">6</div>'
        '<div class="caption">unweighted model at this profile\'s threshold 0.55</div></div>',
        unsafe_allow_html=True,
    )
    pol_b.markdown(
        '<div class="metric-card"><h4>Kids &amp; Family · News approved after</h4>'
        '<div class="value" style="color:#15803d;">0</div>'
        '<div class="caption">balanced model + this profile\'s policy 0.55</div></div>',
        unsafe_allow_html=True,
    )
    pol_c.markdown(
        '<div class="metric-card"><h4>Cost for this profile</h4>'
        '<div class="value" style="color:#b91c1c;">258</div>'
        '<div class="caption">of 658 true Entertainment videos withheld</div></div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "These three numbers belong to Kids & Family Brand only. "
        "A Gaming Publisher (threshold 0.45, allows Gaming + Entertainment) makes the "
        "opposite trade: more inventory, more context risk."
    )

    # ---------- 11. Data scarcity ----------
    step_header(11, "Where the model runs out of data", "videos available per class")
    scarcity = ""
    largest = max(total for _, total, _ in CLASS_DISTRIBUTION)
    for name, total, holdout in CLASS_DISTRIBUTION:
        colour = "#b91c1c" if total < 600 else "#38bdf8"
        scarcity += (
            f'<div class="dist-row"><div class="nm">{esc(name)}</div>'
            f'<div class="tr"><div class="fl" style="width:{total / largest * 100:.1f}%;'
            f'background:{colour};"><span class="bar-top-label">{total:,}</span></div></div>'
            f'<div class="pc">{holdout} in holdout</div></div>'
        )
    st.markdown(scarcity, unsafe_allow_html=True)
    st.markdown(
        '<div class="caveat">Gaming is judged on 20 holdout videos, so one mistake '
        "moves its recall by five points. Rare vocabulary suffers too: phrases like "
        "<i>summer league</i> never appear often enough to enter the 5,000-feature "
        "vocabulary, so genuinely sporty titles can be scored on generic words alone.</div>",
        unsafe_allow_html=True,
    )

    # ---------- 12. Verdict ----------
    step_header(12, "Is the model actually useful?", "deployed model vs the dumb baseline")
    final_a, final_b, final_c, final_d = st.columns(4)
    final_a.markdown(
        '<div class="metric-card"><h4>Macro-F1</h4>'
        '<div class="value" style="color:#15803d;">0.759</div>'
        '<div class="caption">baseline 0.114 · 6.7× better</div></div>',
        unsafe_allow_html=True,
    )
    final_b.markdown(
        '<div class="metric-card"><h4>News recall</h4>'
        '<div class="value" style="color:#15803d;">88.0%</div>'
        '<div class="caption">baseline 0.0%</div></div>',
        unsafe_allow_html=True,
    )
    final_c.markdown(
        '<div class="metric-card"><h4>Accuracy</h4>'
        '<div class="value">78.8%</div>'
        '<div class="caption">baseline 51.8%</div></div>',
        unsafe_allow_html=True,
    )
    final_d.markdown(
        '<div class="metric-card"><h4>Blocked-category approvals</h4>'
        '<div class="value" style="color:#15803d;">0</div>'
        '<div class="caption">Kids &amp; Family at threshold 0.55</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="takeaway"><b>Verdict:</b> the model is genuinely learning language, '
        "not memorising the majority class. It multiplies macro-F1 by roughly six over the "
        "baseline and turns a class the baseline never detected (News) into one it catches "
        "almost nine times out of ten — which is the class the Kids & Family profile "
        "has to block. Other advertiser profiles would treat that same News score differently. "
        "That is enough to ship a <i>contextual suitability</i> MVP, not a safety product.</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="caveat"><b>Read this before trusting the numbers — and what to build next.</b> '
        "The 1,271 videos were reused to compare models, run error analysis and pick the 0.55 "
        "threshold, so they are a development holdout, not a sealed test set. The labels are "
        "YouTube's own categories and are sometimes wrong (opera and an Oprah speech are tagged "
        "Sports). Trending data has no unsafe-content labels, so this measures contextual "
        "suitability — never violence, hate or brand safety.<br><br>"
        "<b>How this model can be improved:</b> (1) freeze the current pipeline and evaluate "
        "once on an untouched test set; (2) collect more Gaming and Sports titles so rare "
        "phrases like <i>summer league</i> enter the vocabulary; (3) audit and relabel noisy "
        "YouTube categories before the next retrain; (4) validate the other three advertiser "
        "thresholds on holdout counts, the same way 0.55 was chosen for Kids &amp; Family; "
        "(5) only then add transcripts, thumbnails or a separate safety model — those are "
        "new products, not patches on this one.</div>",
        unsafe_allow_html=True,
    )

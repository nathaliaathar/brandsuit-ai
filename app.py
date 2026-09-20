# Streamlit app for BrandSuit AI: Live Decision + Project Story.
# Never fits. Category probabilities come from the saved model; YES/NO
# comes from src/policies/brand_policies.py.

import html
import json
from pathlib import Path

import joblib
import streamlit as st

from src.decision import (
    STATUS_INSUFFICIENT,
    STATUS_OK,
    allowed_category_score,
    assess_input,
    combine_text,
    view_state,
)
from src.policies.brand_policies import (
    BRAND_POLICIES,
    VALIDATED_SOURCE,
)
from src.story import render_story

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
LIVE_URL = "https://brandsuit.streamlit.app/"
REPO_URL = "https://github.com/nathaliaathar/brandsuit-ai"

# Illustrative only. Users can edit after loading. Default is a short sports title
# so the first decision is easy to read; the talk-show error case stays in Story.
EXAMPLES = [
    {
        "label": "Sports",
        "title": "NBA Finals Game 7 highlights",
        "description": "Watch the best plays from the championship game.",
    },
    {
        "label": "Gaming",
        "title": "Fortnite Chapter 5 Season launch trailer",
        "description": "New map, weapons, and battle pass in this gameplay trailer.",
    },
    {
        "label": "News",
        "title": "Senate votes on budget bill after overnight debate",
        "description": "Live coverage of the legislative session and political reaction.",
    },
    {
        "label": "Ambiguous talk-show",
        "title": "Milo Takes Calls From Infowars Listeners",
        "description": (
            "Political talk-show host takes live calls and discusses current events "
            "with Infowars listeners."
        ),
    },
]

st.set_page_config(
    page_title="BrandSuit AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def esc(value):
    """Everything injected into markdown HTML goes through this."""
    return html.escape(str(value))


@st.cache_data
def load_evaluation():
    path = ROOT / "reports" / "evaluation.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


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
    """Largest positive logistic coefficients per class from the fitted model.

    Feature names come from the train-only TF-IDF vocabulary. These are
    associations in text, not proof that a video is safe to place.
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
    .decision-banner-abstain {background: #334155; border: 1px solid #1e293b;}
    .decision-banner-abstain h3 {color: #e2e8f0;}
    .decision-banner-stale {background: #78350f; border: 1px solid #92400e;}
    .decision-banner-stale h3 {color: #fde68a;}
    .scored-text {
        font-size: 0.8rem;
        color: #475569;
        margin: 0.35rem 0 0.7rem;
        line-height: 1.4;
    }

    @media (max-width: 900px) {
        .block-container {padding: 1rem 1rem 2rem;}
        .funnel {grid-template-columns: 1fr;}
        .funnel-arrow {display: none;}
        .pipeline-grid {grid-template-columns: 1fr 1fr;}
        .profile-grid {grid-template-columns: 1fr;}
        .word-grid {grid-template-columns: 1fr;}
        .change-grid {grid-template-columns: 1fr;}
        .metric-journey {flex-wrap: wrap;}
        .dist-row {flex-wrap: wrap;}
        .dist-row .nm {flex: 1 1 100%;}
        .dist-row .pc {flex: 1 1 auto;}
        .main-header {padding: 1rem 1.1rem;}
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
            <div class="header-links">
                <a href="{LIVE_URL}" target="_blank">Live demo ↗</a>
                &nbsp;·&nbsp;
                <a href="{REPO_URL}" target="_blank">View repository ↗</a>
            </div>
        </div>
        <div class="badge-pill">Suitability, not safety detection</div>
    </div>
    """,
    unsafe_allow_html=True,
)

vec, model = load_artifacts()
evaluation = load_evaluation()
selected_short = "Balanced LogReg"
if evaluation and evaluation.get("status") == "completed":
    selected_short = {
        "unigram_balanced": "Unigram balanced LR",
        "bigram_balanced": "Bigram balanced LR",
        "unigram_unweighted": "Unigram LR",
    }.get(evaluation["selected_model"]["key"], evaluation["selected_model"]["label"])
if vec is None or model is None:
    st.error(
        "The trained classifier is not available. From this folder run "
        "`py -m src.evaluate` (or `py -m src.export_model` after evaluation) "
        "to rebuild models/category_tfidf.joblib and models/category_logreg.joblib."
    )
    st.stop()

tab_live, tab_story = st.tabs(
    ["Live Decision", "Project Story"]
)

if "video_title" not in st.session_state:
    st.session_state.video_title = EXAMPLES[0]["title"]
    st.session_state.video_desc = EXAMPLES[0]["description"]

# ==========================================
# TAB 1 — LIVE DECISION
# ==========================================
with tab_live:
    st.subheader("Should this brand advertise next to this video?")
    st.caption(
        "Contextual suitability from title and description — not a safety or "
        "child-appropriateness detector. Switch the advertiser to reuse the last "
        "analyzed category scores and apply a different allow-list."
    )

    col_input, col_result = st.columns([1, 1.2], gap="large")

    with col_input:
        st.markdown("**Illustrative examples**")
        st.caption("Not live inventory. Load one, edit if you want, then analyze.")
        example_cols = st.columns(len(EXAMPLES))
        for column, example in zip(example_cols, EXAMPLES):
            if column.button(example["label"], use_container_width=True):
                st.session_state.video_title = example["title"]
                st.session_state.video_desc = example["description"]
                st.rerun()

        video_title = st.text_input(
            "Video title",
            key="video_title",
            placeholder="Paste the YouTube video title...",
        )
        video_desc = st.text_area(
            "Video description",
            key="video_desc",
            height=110,
            placeholder="Paste the video description...",
        )
        advertiser = st.selectbox(
            "Advertiser profile",
            list(BRAND_POLICIES),
            index=0,
            help=(
                "The classifier does not decide YES or NO. Each profile has its "
                "own allowed categories and threshold."
            ),
        )

        policy = BRAND_POLICIES[advertiser]
        allowed_cats = policy["allowed_categories"]
        st.markdown(
            "".join(f'<span class="cat-tag">{esc(cat)}</span>' for cat in allowed_cats),
            unsafe_allow_html=True,
        )
        is_validated = policy["threshold_source"] == VALIDATED_SOURCE
        source_class = "source-validated" if is_validated else "source-illustrative"
        st.markdown(
            f'<div class="policy-line">'
            f'{esc(policy["risk_profile"])} · threshold {policy["threshold"]:.2f} · '
            f'<span class="{source_class}">{esc(policy["threshold_source"])}</span>'
            f"</div>",
            unsafe_allow_html=True,
        )

        analyze_btn = st.button(
            "Analyze placement", type="primary", use_container_width=True
        )

    text = combine_text(video_title, video_desc)

    if not text:
        st.session_state.pop("probabilities", None)
        st.session_state.pop("analysis_status", None)
        st.session_state["scored_text"] = ""
        st.session_state.pop("assessment_message", None)

    if analyze_btn:
        assessment = assess_input(vec, video_title, video_desc)
        st.session_state["analysis_status"] = assessment["status"]
        st.session_state["scored_text"] = assessment["text"]
        st.session_state["assessment_message"] = assessment["message"]
        st.session_state.pop("input_error", None)
        if assessment["status"] == STATUS_OK:
            try:
                st.session_state["probabilities"] = predict_probabilities(
                    vec, model, assessment["text"]
                )
            except Exception:
                st.session_state.pop("probabilities", None)
                st.session_state["analysis_status"] = STATUS_INSUFFICIENT
                st.session_state["assessment_message"] = (
                    "This text could not be scored. Try a different title or description."
                )
        else:
            st.session_state.pop("probabilities", None)

    pane = view_state(
        text,
        st.session_state.get("scored_text"),
        st.session_state.get("analysis_status"),
    )
    probabilities = st.session_state.get("probabilities")
    scored_text = st.session_state.get("scored_text") or ""

    with col_result:
        st.markdown("### Placement decision")

        if pane == "empty":
            st.info("Enter a title or a description, then click **Analyze placement**.")
        elif pane == "idle":
            st.info("Click **Analyze placement** to score this text for the selected profile.")
        elif pane == "stale":
            st.markdown(
                f"""
                <div class="decision-banner decision-banner-stale">
                    <div class="status-row">
                        <h3>Result is out of date</h3>
                    </div>
                    <p>The title or description changed. Click <b>Analyze placement</b>
                    to score the new text. Previous YES/NO is hidden so it cannot be
                    mistaken for the current input.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if scored_text:
                st.markdown(
                    f'<div class="scored-text">Last analyzed text: {esc(scored_text[:180])}</div>',
                    unsafe_allow_html=True,
                )
        elif pane == STATUS_INSUFFICIENT:
            st.markdown(
                f"""
                <div class="decision-banner decision-banner-abstain">
                    <div class="status-row">
                        <h3>Insufficient information</h3>
                    </div>
                    <p>{esc(st.session_state.get("assessment_message") or "")}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif pane == "current" and probabilities:
            score, is_suitable = allowed_category_score(probabilities, policy)
            ranked = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
            predicted = ranked[0][0]
            threshold = policy["threshold"]
            allowed_set = set(allowed_cats)

            if is_suitable:
                banner, word = "decision-banner-yes", "YES · Suitable for this profile"
                symbol, explanation = "&ge;", (
                    f"The allowed-category score for <b>{esc(advertiser)}</b> "
                    "clears this profile's threshold. That is a contextual match "
                    "to the allow-list, not a safety clearance."
                )
            else:
                banner, word = "decision-banner-no", "NO · Not suitable for this profile"
                symbol, explanation = "&lt;", (
                    f"The allowed-category score for <b>{esc(advertiser)}</b> "
                    "is below this profile's threshold. The placement is withheld."
                )

            st.markdown(
                f"""
                <div class="decision-banner {banner}">
                    <div class="status-row">
                        <h3>{word}</h3>
                        <span class="compare">
                            Allowed-category score: {score * 100:.1f}% {symbol}
                            threshold: {threshold * 100:.1f}%
                        </span>
                    </div>
                    <p><b>Predicted category:</b> {esc(predicted)}
                       &nbsp;&bull;&nbsp; <b>Profile:</b> {esc(advertiser)}<br>
                       {explanation}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="scored-text">Analyzed text: {esc(scored_text[:180])}</div>',
                unsafe_allow_html=True,
            )
            st.markdown("**Category scores (same for every profile)**")
            for category, cat_score in ranked:
                style = "prob-name" if category in allowed_set else "prob-name-muted"
                col_label, col_bar, col_val = st.columns([2.5, 5, 1])
                col_label.markdown(
                    f'<div class="{style}">{esc(category)}</div>', unsafe_allow_html=True
                )
                col_bar.progress(min(cat_score, 1.0))
                col_val.markdown(
                    f'<div class="prob-value">{cat_score * 100:.1f}%</div>',
                    unsafe_allow_html=True,
                )

        with st.expander("How the allowed-category score is computed"):
            st.markdown(
                f"""
The classifier turns title + description into TF-IDF features and returns six
category scores that sum to 1. **Allowed-category score** is the sum of those
outputs for the categories this profile allows
({", ".join(policy["allowed_categories"])}).

It is **not** a verified probability that the video is safe, appropriate for
children, or a good real-world placement. Thresholds other than the Kids &
Family development-holdout candidate are illustrative. Model: {selected_short}.
                """
            )

        st.markdown(
            "<div class='insight-card'><b>Model output is not the business decision.</b> "
            "YES / NO comes from the selected advertiser profile "
            f"(<b>{esc(advertiser)}</b>): its allow-list and its threshold. "
            "Switching the profile reuses the last analyzed scores.</div>",
            unsafe_allow_html=True,
        )


# ==========================================
# TAB 2 — PROJECT STORY
# ==========================================
with tab_story:
    render_story(evaluation, top_words_by_category(vec, model, top_n=6))

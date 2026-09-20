"""Project Story tab: numbers come from reports/evaluation.json."""

from __future__ import annotations

import html

import streamlit as st

from src.policies.brand_policies import BRAND_POLICIES, VALIDATED_SOURCE

CATEGORY_COLORS = {
    "Entertainment & Culture": "#0369a1",
    "Lifestyle & Interests": "#0e7490",
    "Education, Science & Technology": "#15803d",
    "News, Politics & Society": "#1d4ed8",
    "Sports": "#0f766e",
    "Gaming": "#4338ca",
}

MODEL_ORDER = (
    "majority_baseline",
    "unigram_unweighted",
    "unigram_balanced",
    "bigram_balanced",
)


def esc(value):
    return html.escape(str(value))


def pct(value, digits=1):
    if value is None:
        return "undefined"
    return f"{100 * value:.{digits}f}%"


def ratio(num, den, digits=3):
    if den == 0:
        return f"{num}/{den} (undefined)"
    return f"{num}/{den} ({num / den:.{digits}f})"


def render_story(payload, top_words):
    if payload is None or payload.get("status") != "completed":
        st.warning(
            "Saved evaluation results are not available. From this folder run "
            "`py -m src.evaluate` after placing `data/raw/USvideos.csv`. "
            "This tab does not invent metrics."
        )
        if payload:
            st.caption(payload.get("reason") or payload.get("results_note", ""))
        return

    selected_key = payload["selected_model"]["key"]
    selected = payload["models"][selected_key]
    kids = selected["policies"]["Kids & Family Brand"]["at_stated_threshold"]
    class_counts = payload["dataset"]["class_counts"]
    holdout_counts = payload["primary_split"]["holdout_class_counts"]
    n_videos = payload["dataset"]["n_unique_videos"]
    n_raw = payload["dataset"]["n_raw_rows"]
    n_train = payload["primary_split"]["n_train"]
    n_holdout = payload["primary_split"]["n_holdout"]
    cls = selected["classification"]
    majority = payload["models"]["majority_baseline"]["classification"]
    uni_b = payload["models"]["unigram_balanced"]["classification"]
    bi_b = payload["models"]["bigram_balanced"]["classification"]
    uni_kids = payload["models"]["unigram_balanced"]["policies"]["Kids & Family Brand"][
        "at_stated_threshold"
    ]
    bi_kids = payload["models"]["bigram_balanced"]["policies"]["Kids & Family Brand"][
        "at_stated_threshold"
    ]

    st.subheader("Project snapshot")
    st.caption(
        "Numbers below are from reports/evaluation.json on the development holdout. "
        "They are simulated policy outcomes on mapped YouTube categories — not measured "
        "ad-delivery or brand-safety impact."
    )

    a, b, c, d = st.columns(4)
    a.markdown(
        f'<div class="metric-card"><h4>Selected macro-F1</h4>'
        f'<div class="value" style="color:#15803d;">{cls["macro_f1"]:.3f}</div>'
        f'<div class="caption">majority {majority["macro_f1"]:.3f} · unigram balanced</div></div>',
        unsafe_allow_html=True,
    )
    b.markdown(
        f'<div class="metric-card"><h4>News recall</h4>'
        f'<div class="value" style="color:#15803d;">{pct(cls["news_recall"], 0)}</div>'
        '<div class="caption">development holdout</div></div>',
        unsafe_allow_html=True,
    )
    c.markdown(
        f'<div class="metric-card"><h4>Kids News approvals</h4>'
        f'<div class="value" style="color:#15803d;">{kids["news_approvals"]}/{kids["news_in_split"]}</div>'
        f'<div class="caption">threshold 0.55 · not all excluded classes</div></div>',
        unsafe_allow_html=True,
    )
    d.markdown(
        f'<div class="metric-card"><h4>Kids excluded-category approvals</h4>'
        f'<div class="value" style="color:#b91c1c;">{kids["incorrect_approvals"]}/{kids["n_truly_excluded"]}</div>'
        f'<div class="caption">{kids["n_approved"]} approved · {pct(kids["retention_of_allowed"])} allowed retained</div></div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
<div class="policy-depends">
<b>Problem.</b> Decide whether an advertiser should run next to a YouTube video using
only title and description. This is contextual suitability, not detection of harmful content.<br><br>
<b>Data and model.</b> {n_raw:,} trending-day rows → {n_videos:,} unique videos.
TF-IDF on title + description, logistic regression. Pets &amp; Animals stay in
Lifestyle &amp; Interests. Train {n_train:,} / development holdout {n_holdout:,},
no video overlap.<br><br>
<b>Verified result.</b> Unigram + balanced class weights is the selected model
(macro-F1 {uni_b["macro_f1"]:.3f}, News recall {uni_b["news_recall"]:.3f}).
Balanced bigrams were {bi_b["macro_f1"]:.3f} / {bi_b["news_recall"]:.3f} on the
<i>same</i> split — not better enough to keep. Kids News approvals
{uni_kids["news_approvals"]}/{uni_kids["news_in_split"]} vs bigrams
{bi_kids["news_approvals"]}/{bi_kids["news_in_split"]}. Excluded-category
approvals {uni_kids["incorrect_approvals"]} vs {bi_kids["incorrect_approvals"]}
(a one-video gap, inside the noise band).<br><br>
<b>Tradeoff.</b> At Kids threshold 0.55 the selected model keeps
{ratio(kids["n_retained_allowed"], kids["n_truly_allowed"])} of allowed mapped
labels and incorrectly approves
{ratio(kids["incorrect_approvals"], kids["n_approved"])} of its YES decisions.
Raising the threshold cuts leaks and inventory together.<br><br>
<b>Main limitation.</b> This holdout was reused for comparison, error reading, and
the 0.55 discussion. It is not an untouched test set. Category match is not proof
that a video is appropriate for children or free of harmful content.
</div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="caveat"><b>Read with the metrics.</b> Mapped YouTube labels are '
        "proxies. Other advertiser thresholds are illustrative. "
        f'Allowed-category score ECE {payload["calibration"]["p_allow_vs_truly_allowed"]["ece"]:.3f} '
        "on this holdout — treat it as a ranking score, not a calibrated probability.</div>",
        unsafe_allow_html=True,
    )

    with st.expander("Dataset, labels, and mapping"):
        st.markdown(
            f"""
            <div class="funnel">
              <div class="funnel-box"><div class="n">{n_raw:,}</div><div class="l">raw CSV rows</div></div>
              <div class="funnel-arrow">→</div>
              <div class="funnel-box"><div class="n">{n_videos:,}</div><div class="l">unique videos</div></div>
              <div class="funnel-arrow">→</div>
              <div class="funnel-box"><div class="n">16</div><div class="l">YouTube names</div></div>
              <div class="funnel-arrow">→</div>
              <div class="funnel-box"><div class="n">6</div><div class="l">suitability classes</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="policy-depends">{esc(payload["mapping_rationale"])}</div>',
            unsafe_allow_html=True,
        )
        rows = ""
        for name, total in class_counts.items():
            share = total / n_videos * 100
            colour = "#0369a1" if share >= 50 else ("#b91c1c" if share < 2 else "#38bdf8")
            hold = holdout_counts.get(name, 0)
            rows += (
                f'<div class="dist-row"><div class="nm">{esc(name)}</div>'
                f'<div class="tr"><div class="fl" style="width:{share:.1f}%;background:{colour};">'
                f'<span class="bar-top-label">{share:.1f}%</span></div></div>'
                f'<div class="pc">{total:,} · {hold} holdout</div></div>'
            )
        st.markdown(rows, unsafe_allow_html=True)

    with st.expander("Development holdout (not a sealed test)"):
        st.markdown(payload["primary_split"]["role"])
        st.caption(
            f"Train {n_train:,} · holdout {n_holdout:,} · video overlap "
            f"{payload['primary_split']['n_video_overlap']}."
        )

    with st.expander("Words the selected model associates with each class"):
        st.markdown(
            "Each bar is a TF-IDF term with a large positive logistic coefficient "
            "for that class, estimated on training videos only. These are text "
            "associations, not evidence that a video is safe to place."
        )
        if top_words:
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
            "Pets stay in Lifestyle, so terms such as cat belong there rather than in Education."
        )

    with st.expander("Model comparison and why unigrams were selected"):
        st.markdown(
            "All four specs share the same split membership and preprocessing. "
            "Bigrams are not the deployed model. Macro-F1 is a classification "
            "average, not a measure of advertising value."
        )
        table_rows = ""
        for key in MODEL_ORDER:
            result = payload["models"][key]
            c = result["classification"]
            k = result["policies"]["Kids & Family Brand"]["at_stated_threshold"]
            sel = "Selected" if key == selected_key else ""
            table_rows += (
                f"<tr><td>{esc(result['label'])}</td>"
                f'<td class="num">{c["accuracy"]:.3f}</td>'
                f'<td class="num">{c["macro_f1"]:.3f}</td>'
                f'<td class="num">{c["news_recall"]:.3f}</td>'
                f'<td class="num">{k["news_approvals"]}/{k["news_in_split"]}</td>'
                f'<td class="num">{k["incorrect_approvals"]}/{k["n_truly_excluded"]}</td>'
                f'<td class="num">{pct(k["retention_of_allowed"])}</td>'
                f"<td>{sel}</td></tr>"
            )
        st.markdown(
            "<table class='case-table'><thead><tr>"
            "<th>Model</th><th>Accuracy</th><th>Macro-F1</th><th>News recall</th>"
            "<th>News approvals</th><th>Excluded-category approvals</th>"
            "<th>Allowed retained</th><th></th>"
            f"</tr></thead><tbody>{table_rows}</tbody></table>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="takeaway">{esc(payload["selected_model"]["rationale"])}</div>',
            unsafe_allow_html=True,
        )

    with st.expander("Error examples (News leaks and remaining excluded approvals)"):
        leaks = payload.get("news_leaks_unweighted_vs_selected", [])
        if leaks:
            leak_rows = "".join(
                "<tr>"
                f"<td>{esc(item['title'])}</td>"
                f'<td class="num">{item["p_allow_before"]:.2f}</td>'
                f'<td class="num">{item["p_allow_after"]:.2f}</td>'
                f'<td><span class="pill {"pill-bad" if item["approved_after"] else "pill-good"}">'
                f'{"Approved" if item["approved_after"] else "Withheld"}</span></td></tr>'
                for item in leaks
            )
            st.markdown(
                "<table class='case-table'><thead><tr>"
                "<th>News video the unweighted model approved for Kids &amp; Family</th>"
                "<th>Score before</th><th>Score after</th><th>Kids after</th>"
                f"</tr></thead><tbody>{leak_rows}</tbody></table>",
                unsafe_allow_html=True,
            )
        by_cat = kids["incorrect_approvals_by_category"]
        cat_txt = ", ".join(f"{name}: {count}" for name, count in by_cat.items())
        st.markdown(
            f'<div class="caveat"><b>News approvals are not the full leak count.</b> '
            f'Kids &amp; Family News approvals: {kids["news_approvals"]}/{kids["news_in_split"]}. '
            f'All excluded-category approvals: {kids["incorrect_approvals"]}/{kids["n_truly_excluded"]} '
            f"({cat_txt}). Gaming, Lifestyle, and Education are also excluded.</div>",
            unsafe_allow_html=True,
        )
        remaining = [
            row for row in payload.get("error_examples", []) if row["kind"] == "incorrect_approval"
        ]
        if remaining:
            rem_rows = "".join(
                "<tr>"
                f"<td>{esc(item['title'])}</td>"
                f"<td>{esc(item['mapped_label'])}</td>"
                f'<td class="num">{item["p_allow"]:.2f}</td></tr>'
                for item in remaining
            )
            st.markdown(
                "<table class='case-table'><thead><tr>"
                "<th>Still approved for kids (mapped label is excluded)</th>"
                "<th>Mapped label</th><th>Allowed-category score</th>"
                f"</tr></thead><tbody>{rem_rows}</tbody></table>",
                unsafe_allow_html=True,
            )
            st.caption(
                "Some Lifestyle labels are sports or music videos (YouTube category noise). "
                "Gaming rows are genuine kids-policy leaks under this proxy."
            )

    with st.expander("Policy outcomes and the threshold tradeoff"):
        st.caption(
            "Simulated on mapped labels. Not observed advertising impact. "
            "Zero denominators would be shown as undefined."
        )
        profile_cards = ""
        for name, spec in BRAND_POLICIES.items():
            cats = " · ".join(spec["allowed_categories"])
            source = spec["threshold_source"]
            active = " active" if spec["threshold_source"] == VALIDATED_SOURCE else ""
            profile_cards += (
                f'<div class="profile-card{active}"><h4>{esc(name)}</h4>'
                f'<div class="meta">{esc(spec["risk_profile"])} · {spec["threshold"]:.2f}<br>'
                f"{esc(source)}</div>"
                f'<div class="cats">Allows: {esc(cats)}</div></div>'
            )
        st.markdown(f'<div class="profile-grid">{profile_cards}</div>', unsafe_allow_html=True)
        policy_rows = ""
        for name, spec in BRAND_POLICIES.items():
            t = selected["policies"][name]["at_stated_threshold"]
            policy_rows += (
                "<tr>"
                f"<td>{esc(name)} ({t['threshold']:.2f})</td>"
                f'<td class="num">{t["news_approvals"]}/{t["news_in_split"]}</td>'
                f'<td class="num">{ratio(t["incorrect_approvals"], t["n_truly_excluded"])}</td>'
                f'<td class="num">{ratio(t["incorrect_approvals"], t["n_approved"])}</td>'
                f'<td class="num">{ratio(t["n_retained_allowed"], t["n_truly_allowed"])}</td>'
                f'<td class="num">{ratio(t["n_approved"], t["n_total"])}</td></tr>'
            )
        st.markdown(
            "<table class='case-table'><thead><tr>"
            "<th>Policy</th><th>News approvals</th><th>Incorrect / excluded</th>"
            "<th>Incorrect / approved</th><th>Allowed retained</th><th>Approval rate</th>"
            f"</tr></thead><tbody>{policy_rows}</tbody></table>",
            unsafe_allow_html=True,
        )
        sweep = selected["policies"]["Kids & Family Brand"]["threshold_sweep"]
        sweep_rows = "".join(
            "<tr>"
            f'<td class="num">{cut}</td>'
            f'<td class="num">{row["incorrect_approvals"]}</td>'
            f'<td class="num">{row["news_approvals"]}</td>'
            f'<td class="num">{pct(row["retention_of_allowed"])}</td>'
            f'<td class="num">{row["n_approved"]}/{row["n_total"]}</td></tr>'
            for cut, row in sweep.items()
        )
        st.markdown("**Kids & Family threshold curve (development holdout)**")
        st.markdown(
            "<table class='case-table'><thead><tr>"
            "<th>Threshold</th><th>Excluded-category approvals</th><th>News approvals</th>"
            "<th>Allowed retained</th><th>Approved / holdout</th>"
            f"</tr></thead><tbody>{sweep_rows}</tbody></table>",
            unsafe_allow_html=True,
        )

    with st.expander("Calibration and robustness"):
        cal = payload["calibration"]["p_allow_vs_truly_allowed"]
        st.markdown(
            f"Allowed-category score vs “mapped label is allowed”: ECE {cal['ece']:.3f}, "
            f"Brier {cal['brier']:.3f}. No calibrator was fitted. "
            "Any future calibrator must use development data only."
        )
        rob_rows = ""
        for name, block in payload["robustness"].items():
            c = block["classification"]
            k = block["kids_family_policy"]
            warn = ", ".join(
                cls for cls, flag in block["split"].get("rare_class_warning", {}).items() if flag
            ) or "none"
            rob_rows += (
                "<tr>"
                f"<td>{esc(name)}</td>"
                f'<td class="num">{block["split"]["n_holdout"]}</td>'
                f'<td class="num">{c["accuracy"]:.3f}</td>'
                f'<td class="num">{c["macro_f1"]:.3f}</td>'
                f'<td class="num">{c["news_recall"]:.3f}</td>'
                f'<td class="num">{k["incorrect_approvals"]}</td>'
                f'<td class="num">{k["news_approvals"]}</td>'
                f"<td>{esc(warn)}</td></tr>"
            )
        st.markdown(
            "<table class='case-table'><thead><tr>"
            "<th>Check</th><th>n</th><th>Accuracy</th><th>Macro-F1</th>"
            "<th>News recall</th><th>Kids excluded approvals</th><th>News approvals</th><th>Rare-class warning</th>"
            f"</tr></thead><tbody>{rob_rows}</tbody></table>",
            unsafe_allow_html=True,
        )
        st.caption(
            "Channel-disjoint and temporal checks retrain the frozen spec under a different "
            "split rule. They are stress tests, not a reshuffle sold as confirmation."
        )

    with st.expander("Limitations and external validation plan"):
        limits = "".join(f"<li>{esc(item)}</li>" for item in payload.get("limitations", []))
        st.markdown(
            f"<ul>{limits}</ul>"
            "Freeze this mapping, spec, and Kids 0.55 candidate, then score a later US "
            "scrape or another country file never used here."
        )

"""Reproducible evaluation: fit four specs on one split, score policies, save JSON.

Run from the project folder:

    py -m src.evaluate

Writes reports/evaluation.json, reports/split_ids.json, and the selected
joblib artifacts. Does not invent metrics when the CSV is missing.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
)

from src.config import (
    CALIBRATION_BINS,
    CATEGORY_MAPPING,
    CATEGORY_MAPPING_RATIONALE,
    COMPARABLE_THRESHOLDS,
    EVALUATION_PATH,
    INCORRECT_APPROVAL_TIE_DELTA,
    MODEL_SPECS,
    NEWS_CLASS,
    PRIMARY_SPLIT,
    REPORTS_DIR,
    SELECTION_POLICY,
    SELECTION_RULE,
    SPLIT_IDS_PATH,
    TARGET_CLASSES,
    THRESHOLD_SWEEP,
)
from src.dataset import (
    DatasetUnavailable,
    dataset_fingerprint,
    load_raw_videos,
    make_channel_disjoint_split,
    make_primary_split,
    make_temporal_split,
    mapping_fingerprint,
    prepare_dataset,
)
from src.export_model import save_fitted_artifacts
from src.modeling import fit_spec, predict_proba_frame, top_coefficients
from src.policies.brand_policies import BRAND_POLICIES, policy_outcome_metrics, safe_div


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Not JSON serializable: {type(value)}")


def package_versions() -> dict[str, str]:
    names = [
        "pandas",
        "numpy",
        "scikit-learn",
        "joblib",
        "streamlit",
        "matplotlib",
        "seaborn",
        "pytest",
        "jupyter",
    ]
    versions = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def classification_block(y_true: pd.Series, y_pred: pd.Series) -> dict:
    labels = list(TARGET_CLASSES)
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    per_class = {}
    for cls in labels:
        row = report.get(cls, {})
        per_class[cls] = {
            "precision": float(row.get("precision", 0.0)),
            "recall": float(row.get("recall", 0.0)),
            "f1": float(row.get("f1-score", 0.0)),
            "support": int(row.get("support", 0)),
        }
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "per_class": per_class,
        "confusion_matrix": {
            "labels": labels,
            "matrix": matrix.astype(int).tolist(),
        },
        "news_recall": per_class[NEWS_CLASS]["recall"],
        "news_precision": per_class[NEWS_CLASS]["precision"],
        "news_support": per_class[NEWS_CLASS]["support"],
    }


def expected_calibration_error(y_binary: np.ndarray, proba: np.ndarray, n_bins: int) -> dict:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bin_rows = []
    n = len(y_binary)
    for i in range(n_bins):
        left, right = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (proba >= left) & (proba <= right)
        else:
            mask = (proba >= left) & (proba < right)
        count = int(mask.sum())
        if count == 0:
            bin_rows.append(
                {
                    "bin": [float(left), float(right)],
                    "count": 0,
                    "mean_predicted": None,
                    "fraction_positive": None,
                }
            )
            continue
        mean_pred = float(proba[mask].mean())
        frac = float(y_binary[mask].mean())
        ece += (count / n) * abs(frac - mean_pred)
        bin_rows.append(
            {
                "bin": [float(left), float(right)],
                "count": count,
                "mean_predicted": mean_pred,
                "fraction_positive": frac,
            }
        )
    return {
        "n_bins": n_bins,
        "ece": float(ece) if n else None,
        "brier": float(brier_score_loss(y_binary, proba)) if n else None,
        "bins": bin_rows,
        "note": (
            "Computed on the development holdout only. No isotonic/Platt "
            "calibrator was fitted. p_allow is a summed model score, not a "
            "guaranteed frequency of 'truly allowed'."
        ),
    }


def policy_block(y_true: pd.Series, proba: pd.DataFrame) -> dict:
    named = {}
    for name, policy in BRAND_POLICIES.items():
        named[name] = {
            "at_stated_threshold": policy_outcome_metrics(
                y_true, proba, policy
            ),
            "comparable_thresholds": {
                f"{cut:.2f}": policy_outcome_metrics(y_true, proba, policy, threshold=cut)
                for cut in COMPARABLE_THRESHOLDS
            },
            "threshold_sweep": {
                f"{cut:.2f}": policy_outcome_metrics(y_true, proba, policy, threshold=cut)
                for cut in THRESHOLD_SWEEP
            },
        }
    return named


def select_model(model_results: dict) -> tuple[str, str]:
    rows = []
    for key, result in model_results.items():
        kids = result["policies"][SELECTION_POLICY]["at_stated_threshold"]
        rows.append((key, kids, MODEL_SPECS[key]["complexity"]))
    min_incorrect = min(kids["incorrect_approvals"] for _, kids, _ in rows)
    tied = [
        (key, kids, complexity)
        for key, kids, complexity in rows
        if kids["incorrect_approvals"] <= min_incorrect + INCORRECT_APPROVAL_TIE_DELTA
    ]

    def sort_key(item):
        key, kids, complexity = item
        retention = kids["retention_of_allowed"]
        retention_rank = -retention if retention is not None else 0.0
        return (retention_rank, complexity, key)

    tied.sort(key=sort_key)
    winner_key = tied[0][0]
    lines = [
        SELECTION_RULE,
        f"Lowest incorrect-approval count: {min_incorrect}. "
        f"Tie band: {INCORRECT_APPROVAL_TIE_DELTA} videos.",
        "Kids & Family at its stated threshold:",
    ]
    for key, kids, complexity in sorted(rows, key=lambda item: item[2]):
        ret = kids["retention_of_allowed"]
        ret_txt = "undefined" if ret is None else f"{ret:.3f}"
        in_tie = kids["incorrect_approvals"] <= min_incorrect + INCORRECT_APPROVAL_TIE_DELTA
        mark = " [SELECTED]" if key == winner_key else (" [tie group]" if in_tie else "")
        lines.append(
            f"- {MODEL_SPECS[key]['label']}: incorrect_approvals="
            f"{kids['incorrect_approvals']}/{kids['n_truly_excluded']} excluded, "
            f"retention={ret_txt} of {kids['n_truly_allowed']} allowed, "
            f"news_approvals={kids['news_approvals']}/{kids['news_in_split']}, "
            f"complexity={complexity}{mark}"
        )
    return winner_key, "\n".join(lines)


def error_examples(holdout: pd.DataFrame, proba: pd.DataFrame, policy_name: str, n: int = 8) -> list[dict]:
    policy = BRAND_POLICIES[policy_name]
    allowed = policy["allowed_categories"]
    p_allow = proba.reindex(columns=allowed, fill_value=0.0).sum(axis=1)
    approved = p_allow >= policy["threshold"]
    truly_excluded = ~holdout["y"].isin(allowed)
    incorrect = holdout[approved.to_numpy() & truly_excluded.to_numpy()].copy()
    incorrect = incorrect.assign(
        p_allow=p_allow.loc[incorrect.index].to_numpy(),
        p_news=proba.loc[incorrect.index, NEWS_CLASS].to_numpy(),
        pred=proba.loc[incorrect.index].idxmax(axis=1).to_numpy(),
    )
    rows = []
    for _, row in incorrect.sort_values("p_allow", ascending=False).head(n).iterrows():
        rows.append(
            {
                "kind": "incorrect_approval",
                "video_id": str(row["video_id"]),
                "title": str(row["title"]),
                "mapped_label": str(row["y"]),
                "predicted": str(row["pred"]),
                "p_allow": float(row["p_allow"]),
                "p_news": float(row["p_news"]),
            }
        )
    # Remaining News videos the model still scores as Entertainment (argmax),
    # whether or not the policy approved them — useful for the app story.
    pred = proba.idxmax(axis=1)
    news_as_ent = holdout[
        (holdout["y"] == NEWS_CLASS) & (pred == "Entertainment & Culture")
    ].copy()
    news_as_ent = news_as_ent.assign(
        p_allow=p_allow.loc[news_as_ent.index].to_numpy(),
        p_news=proba.loc[news_as_ent.index, NEWS_CLASS].to_numpy(),
        pred=pred.loc[news_as_ent.index].to_numpy(),
    )
    for _, row in news_as_ent.sort_values("p_allow", ascending=False).head(n).iterrows():
        rows.append(
            {
                "kind": "news_argmax_entertainment",
                "video_id": str(row["video_id"]),
                "title": str(row["title"]),
                "mapped_label": str(row["y"]),
                "predicted": str(row["pred"]),
                "p_allow": float(row["p_allow"]),
                "p_news": float(row["p_news"]),
                "policy_would_approve": bool(row["p_allow"] >= policy["threshold"]),
            }
        )
    return rows


def news_leaks_before_after(
    holdout: pd.DataFrame,
    before_proba: pd.DataFrame,
    after_proba: pd.DataFrame,
    policy_name: str,
) -> list[dict]:
    """News videos the first real model approved for Kids & Family, with later scores."""
    policy = BRAND_POLICIES[policy_name]
    allowed = policy["allowed_categories"]
    cut = policy["threshold"]
    p_before = before_proba.reindex(columns=allowed, fill_value=0.0).sum(axis=1)
    p_after = after_proba.reindex(columns=allowed, fill_value=0.0).sum(axis=1)
    is_news = holdout["y"] == NEWS_CLASS
    leaked = holdout[is_news.to_numpy() & (p_before >= cut).to_numpy()].copy()
    rows = []
    for idx, row in leaked.iterrows():
        rows.append(
            {
                "video_id": str(row["video_id"]),
                "title": str(row["title"]),
                "p_allow_before": float(p_before.loc[idx]),
                "p_allow_after": float(p_after.loc[idx]),
                "approved_before": True,
                "approved_after": bool(p_after.loc[idx] >= cut),
            }
        )
    rows.sort(key=lambda item: item["p_allow_before"], reverse=True)
    return rows


def robustness_eval(df: pd.DataFrame, spec_key: str) -> dict:
    spec = MODEL_SPECS[spec_key]
    out = {}
    for name, splitter in (
        ("channel_disjoint", make_channel_disjoint_split),
        ("temporal", make_temporal_split),
    ):
        train, holdout, info = splitter(df)
        bundle = fit_spec(spec_key, spec, train["text"], train["y"])
        proba = predict_proba_frame(bundle, holdout["text"])
        pred = proba.idxmax(axis=1)
        out[name] = {
            "split": info,
            "classification": classification_block(holdout["y"], pred),
            "kids_family_policy": policy_outcome_metrics(
                holdout["y"], proba, BRAND_POLICIES[SELECTION_POLICY]
            ),
        }
    return out


def pending_payload(reason: str) -> dict:
    return {
        "status": "pending_data",
        "reason": reason,
        "mapping_version": "advertiser-context-v1",
        "mapping_sha256_16": mapping_fingerprint(),
        "primary_split": PRIMARY_SPLIT,
        "model_specs": {
            key: {k: v for k, v in spec.items() if k != "kind"} | {"kind": spec["kind"]}
            for key, spec in MODEL_SPECS.items()
        },
        "selection_rule": SELECTION_RULE,
        "results_note": "No metrics were invented. Run py -m src.evaluate after placing USvideos.csv.",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": sys.version,
            "packages": package_versions(),
        },
    }


def run_evaluation() -> dict:
    raw = load_raw_videos()
    df = prepare_dataset()
    train, holdout, split_info = make_primary_split(df)
    if split_info["n_video_overlap"] != 0:
        raise RuntimeError("Train and development holdout share video_id values.")

    model_results = {}
    fitted = {}
    holdout_probas = {}
    for key, spec in MODEL_SPECS.items():
        print(f"fitting {key} ...")
        bundle = fit_spec(key, spec, train["text"], train["y"])
        fitted[key] = bundle
        proba = predict_proba_frame(bundle, holdout["text"])
        holdout_probas[key] = proba
        pred = proba.idxmax(axis=1)
        model_results[key] = {
            "label": spec["label"],
            "spec": {
                "kind": spec["kind"],
                "ngram_range": list(spec.get("ngram_range", [])),
                "class_weight": spec.get("class_weight"),
                "max_features": spec.get("max_features"),
                "min_df": spec.get("min_df"),
                "strategy": spec.get("strategy"),
            },
            "classification": classification_block(holdout["y"], pred),
            "policies": policy_block(holdout["y"], proba),
        }

    selected_key, rationale = select_model(model_results)
    print(f"selected model: {selected_key}")
    selected_proba = holdout_probas[selected_key]
    selected_policy = BRAND_POLICIES[SELECTION_POLICY]
    allowed = selected_policy["allowed_categories"]
    p_allow = selected_proba.reindex(columns=allowed, fill_value=0.0).sum(axis=1)
    y_allowed = holdout["y"].isin(allowed).astype(int).to_numpy()
    calibration = {
        "p_allow_vs_truly_allowed": expected_calibration_error(
            y_allowed, p_allow.to_numpy(), CALIBRATION_BINS
        ),
        "max_class_prob_vs_correct": expected_calibration_error(
            (selected_proba.idxmax(axis=1).to_numpy() == holdout["y"].to_numpy()).astype(int),
            selected_proba.max(axis=1).to_numpy(),
            CALIBRATION_BINS,
        ),
    }

    print("robustness checks ...")
    robustness = robustness_eval(df, selected_key)

    kids_stated = model_results[selected_key]["policies"][SELECTION_POLICY][
        "at_stated_threshold"
    ]
    selected_bundle = fitted[selected_key]
    manifest = save_fitted_artifacts(selected_bundle)

    payload = {
        "status": "completed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "n_raw_rows": int(len(raw)),
            "source": "Kaggle datasnaek/youtube-new US files; see data/raw/SOURCE.txt",
            **dataset_fingerprint(df),
            "pets_mapping": CATEGORY_MAPPING["Pets & Animals"],
        },
        "mapping_rationale": CATEGORY_MAPPING_RATIONALE,
        "primary_split": split_info,
        "environment": {
            "python": sys.version,
            "packages": package_versions(),
        },
        "models": model_results,
        "selected_model": {
            "key": selected_key,
            "label": MODEL_SPECS[selected_key]["label"],
            "rationale": rationale,
            "classification": model_results[selected_key]["classification"],
            "kids_family_at_stated_threshold": kids_stated,
            "top_coefficients": top_coefficients(selected_bundle),
        },
        "selection_rule": SELECTION_RULE,
        "calibration": calibration,
        "robustness": robustness,
        "error_examples": error_examples(holdout, selected_proba, SELECTION_POLICY),
        "news_leaks_unweighted_vs_selected": news_leaks_before_after(
            holdout,
            holdout_probas["unigram_unweighted"],
            selected_proba,
            SELECTION_POLICY,
        ),
        "leak_comparison": _leak_table(model_results),
        "integrity_checks": {
            "video_overlap_train_holdout": split_info["n_video_overlap"],
            "shared_mapping_source": "src.config.CATEGORY_MAPPING",
            "mapping_sha256_16": mapping_fingerprint(),
            "policy_boundary": "approved iff p_allow >= threshold",
            "excluded_categories_counted": (
                "incorrect_approvals sums every mapped class not on the allow-list, "
                "not News alone"
            ),
            "saved_model_spec_key": selected_key,
        },
        "limitations": [
            "The 20% slice is a development holdout reused for comparison, errors, and threshold talk. It is not a sealed test set.",
            "A new random split of these same videos would not be independent confirmation.",
            "Mapped YouTube categories are proxies for context, not human-verified suitability or safety.",
            "No violence/hate labels exist in this scrape; the product is suitability, not brand safety.",
            "Gaming support is small; per-class metrics and robustness slices can move a lot from one error.",
            "p_allow was not recalibrated; treat it as a score to rank and threshold, not as a proven probability.",
            "External validation would freeze this pipeline and score a later time window or another country file never used here.",
        ],
        "manifest": manifest,
    }
    return payload, train, holdout


def _leak_table(model_results: dict) -> dict:
    table = {}
    for key, result in model_results.items():
        kids = result["policies"][SELECTION_POLICY]["at_stated_threshold"]
        table[key] = {
            "news_approvals": kids["news_approvals"],
            "incorrect_approvals": kids["incorrect_approvals"],
            "incorrect_by_category": kids["incorrect_approvals_by_category"],
            "retention_of_allowed": kids["retention_of_allowed"],
            "n_approved": kids["n_approved"],
        }
    return table


def write_outputs(payload: dict, train: pd.DataFrame | None, holdout: pd.DataFrame | None) -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    EVALUATION_PATH.write_text(
        json.dumps(payload, indent=2, default=_json_default),
        encoding="utf-8",
    )
    if train is not None and holdout is not None:
        SPLIT_IDS_PATH.write_text(
            json.dumps(
                {
                    "primary_split_name": "development_holdout",
                    "random_state": PRIMARY_SPLIT["random_state"],
                    "train_video_ids": train["video_id"].astype(str).tolist(),
                    "holdout_video_ids": holdout["video_id"].astype(str).tolist(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    print("wrote", EVALUATION_PATH)
    if train is not None:
        print("wrote", SPLIT_IDS_PATH)


def main() -> int:
    try:
        payload, train, holdout = run_evaluation()
        write_outputs(payload, train, holdout)
        selected = payload["selected_model"]
        cls = selected["classification"]
        kids = selected["kids_family_at_stated_threshold"]
        print(
            f"holdout accuracy={cls['accuracy']:.3f} macro-F1={cls['macro_f1']:.3f} "
            f"News recall={cls['news_recall']:.3f}"
        )
        print(
            f"Kids & Family incorrect approvals={kids['incorrect_approvals']} "
            f"(News {kids['news_approvals']}) / {kids['n_approved']} approved; "
            f"retention={kids['retention_of_allowed']}"
        )
        return 0
    except DatasetUnavailable as exc:
        print(exc)
        write_outputs(pending_payload(str(exc)), None, None)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

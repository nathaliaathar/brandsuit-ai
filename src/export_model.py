"""Write the selected classifier artifacts for Streamlit.

The app only transforms and predicts. This file fits the spec named in
reports/evaluation.json (or, if that file is missing, tells you to run
the evaluator first). Training uses the same primary split as evaluation
so saved inference matches the reported pipeline.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import joblib

from src.config import (
    EVALUATION_PATH,
    MANIFEST_PATH,
    MODEL_PATH,
    MODEL_SPECS,
    MODELS_DIR,
    VEC_PATH,
)
from src.dataset import DatasetUnavailable, make_primary_split, mapping_fingerprint, prepare_dataset
from src.modeling import FittedBundle, fit_spec


def save_fitted_artifacts(bundle: FittedBundle) -> dict:
    MODELS_DIR.mkdir(exist_ok=True)
    if bundle.vectorizer is None:
        raise ValueError("The majority baseline has no vectorizer to export for the app.")
    joblib.dump(bundle.vectorizer, VEC_PATH)
    joblib.dump(bundle.classifier, MODEL_PATH)
    manifest = {
        "spec_key": bundle.spec_key,
        "label": bundle.spec.get("label"),
        "ngram_range": list(bundle.spec.get("ngram_range", [])),
        "class_weight": bundle.spec.get("class_weight"),
        "max_features": bundle.spec.get("max_features"),
        "min_df": bundle.spec.get("min_df"),
        "mapping_sha256_16": mapping_fingerprint(),
        "classes": list(bundle.classifier.classes_),
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "vectorizer_path": str(VEC_PATH.name),
        "model_path": str(MODEL_PATH.name),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("saved", VEC_PATH)
    print("saved", MODEL_PATH)
    print("saved", MANIFEST_PATH)
    return manifest


def selected_key_from_evaluation() -> str:
    if not EVALUATION_PATH.exists():
        raise FileNotFoundError(
            f"{EVALUATION_PATH} is missing. Run `py -m src.evaluate` first "
            "so the exported model is the one selected by the stated rule."
        )
    payload = json.loads(EVALUATION_PATH.read_text(encoding="utf-8"))
    if payload.get("status") != "completed":
        raise FileNotFoundError(
            "Evaluation did not complete. Place data/raw/USvideos.csv and run "
            "`py -m src.evaluate` before exporting."
        )
    return payload["selected_model"]["key"]


def main() -> None:
    key = selected_key_from_evaluation()
    if key not in MODEL_SPECS or MODEL_SPECS[key]["kind"] == "dummy":
        raise ValueError(f"Cannot export spec {key!r} as the Streamlit classifier.")
    df = prepare_dataset()
    train, holdout, info = make_primary_split(df)
    print("re-fitting", key, "on", info["n_train"], "train videos")
    bundle = fit_spec(key, MODEL_SPECS[key], train["text"], train["y"])
    save_fitted_artifacts(bundle)
    print("holdout videos (not used for fitting):", info["n_holdout"])


if __name__ == "__main__":
    try:
        main()
    except DatasetUnavailable as exc:
        print(exc)
        raise SystemExit(2) from exc

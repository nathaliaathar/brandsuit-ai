"""Fit and score the four candidate specs with a shared interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.config import TARGET_CLASSES


@dataclass
class FittedBundle:
    spec_key: str
    spec: dict[str, Any]
    vectorizer: TfidfVectorizer | None
    classifier: Any


def fit_spec(spec_key: str, spec: dict[str, Any], texts: pd.Series, y: pd.Series) -> FittedBundle:
    if spec["kind"] == "dummy":
        clf = DummyClassifier(strategy=spec["strategy"])
        clf.fit(np.zeros((len(y), 1)), y)
        return FittedBundle(spec_key, spec, None, clf)

    ngram = tuple(spec["ngram_range"])
    vectorizer = TfidfVectorizer(
        max_features=spec["max_features"],
        ngram_range=ngram,
        min_df=spec["min_df"],
    )
    X_train = vectorizer.fit_transform(texts)
    clf = LogisticRegression(
        max_iter=spec["max_iter"],
        class_weight=spec["class_weight"],
    )
    clf.fit(X_train, y)
    return FittedBundle(spec_key, spec, vectorizer, clf)


def predict_proba_frame(bundle: FittedBundle, texts: pd.Series) -> pd.DataFrame:
    if bundle.vectorizer is None:
        X = np.zeros((len(texts), 1))
    else:
        X = bundle.vectorizer.transform(texts)
    proba = bundle.classifier.predict_proba(X)
    frame = pd.DataFrame(
        proba,
        columns=list(bundle.classifier.classes_),
        index=texts.index,
    )
    return frame.reindex(columns=list(TARGET_CLASSES), fill_value=0.0)


def top_coefficients(bundle: FittedBundle, top_n: int = 6) -> dict[str, list[list]]:
    if bundle.vectorizer is None:
        return {}
    names = bundle.vectorizer.get_feature_names_out()
    result = {}
    for class_idx, class_name in enumerate(bundle.classifier.classes_):
        coefs = bundle.classifier.coef_[class_idx]
        top_idx = coefs.argsort()[::-1][:top_n]
        result[str(class_name)] = [
            [str(names[i]), float(coefs[i])] for i in top_idx
        ]
    return result

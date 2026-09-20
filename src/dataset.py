"""Load, map, fingerprint, and split the YouTube Trending videos.

Category mapping, text construction, and split parameters come from
src.config so the notebook and the evaluator cannot disagree.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    CATEGORY_JSON,
    CATEGORY_MAPPING,
    CHANNEL_HOLDOUT_FRACTION,
    MAPPING_VERSION,
    PRIMARY_SPLIT,
    RANDOM_STATE,
    RAW_CSV,
    TARGET_CLASSES,
    TEMPORAL_QUANTILE,
)


class DatasetUnavailable(FileNotFoundError):
    """Raised when USvideos.csv is not on disk."""


def load_category_names(path: Path | None = None) -> dict[int, str]:
    payload = json.loads((path or CATEGORY_JSON).read_text(encoding="utf-8"))
    return {int(item["id"]): item["snippet"]["title"] for item in payload["items"]}


def load_raw_videos(csv_path: Path | None = None) -> pd.DataFrame:
    path = csv_path or RAW_CSV
    if not path.exists():
        raise DatasetUnavailable(
            f"Required dataset not found: {path}\n"
            "Place the Kaggle datasnaek/youtube-new US file as data/raw/USvideos.csv "
            "(see data/raw/SOURCE.txt). Do not invent rows."
        )
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1")


def build_text(title, description) -> str:
    return f"{'' if pd.isna(title) else title} {'' if pd.isna(description) else description}".strip()


def prepare_dataset(csv_path: Path | None = None) -> pd.DataFrame:
    """One row per video_id, with mapped label y and model text."""
    raw = load_raw_videos(csv_path)
    id_to_name = load_category_names()
    df = raw.drop_duplicates(subset="video_id", keep="first").copy()
    df["youtube_name"] = df["category_id"].map(id_to_name)
    df["y"] = df["youtube_name"].map(CATEGORY_MAPPING).fillna("Other")
    unexpected = sorted(set(df["y"].unique()) - set(TARGET_CLASSES))
    if unexpected:
        raise ValueError(
            f"Mapped labels outside TARGET_CLASSES: {unexpected}. "
            "Update CATEGORY_MAPPING or TARGET_CLASSES together."
        )
    df["text"] = [
        build_text(title, desc) for title, desc in zip(df["title"], df["description"])
    ]
    df["publish_dt"] = pd.to_datetime(df["publish_time"], utc=True, errors="coerce")
    return df.reset_index(drop=True)


def mapping_fingerprint() -> str:
    payload = json.dumps(CATEGORY_MAPPING, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def dataset_fingerprint(df: pd.DataFrame) -> dict:
    video_ids = sorted(df["video_id"].astype(str))
    joined = "\n".join(video_ids).encode("utf-8")
    return {
        "n_unique_videos": int(len(df)),
        "n_raw_source_note": "see evaluation.json dataset.n_raw_rows if recorded",
        "sha256_sorted_video_ids": hashlib.sha256(joined).hexdigest(),
        "mapping_version": MAPPING_VERSION,
        "mapping_sha256_16": mapping_fingerprint(),
        "class_counts": {cls: int((df["y"] == cls).sum()) for cls in TARGET_CLASSES},
        "n_channels": int(df["channel_title"].nunique(dropna=False)),
    }


def make_primary_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Stratified 80/20 split on mapped category. Same membership for every model."""
    train_idx, holdout_idx = train_test_split(
        df.index,
        test_size=PRIMARY_SPLIT["test_size"],
        random_state=PRIMARY_SPLIT["random_state"],
        stratify=df["y"],
    )
    train = df.loc[train_idx].copy()
    holdout = df.loc[holdout_idx].copy()
    overlap = set(train["video_id"]).intersection(set(holdout["video_id"]))
    info = {
        **PRIMARY_SPLIT,
        "n_train": int(len(train)),
        "n_holdout": int(len(holdout)),
        "n_video_overlap": int(len(overlap)),
        "train_class_counts": {cls: int((train["y"] == cls).sum()) for cls in TARGET_CLASSES},
        "holdout_class_counts": {
            cls: int((holdout["y"] == cls).sum()) for cls in TARGET_CLASSES
        },
    }
    return train, holdout, info


def make_channel_disjoint_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """All videos from a channel go to train or to holdout.

    This is a robustness check, not the primary comparison split. It asks
    whether the model still works on channels it has never seen. Rare-class
    support can be thin because channels are the sampling unit, not videos.
    """
    channels = (
        df["channel_title"].fillna("__missing_channel__").unique().tolist()
    )
    train_ch, holdout_ch = train_test_split(
        channels,
        test_size=CHANNEL_HOLDOUT_FRACTION,
        random_state=RANDOM_STATE,
    )
    holdout_set = set(holdout_ch)
    key = df["channel_title"].fillna("__missing_channel__")
    holdout = df[key.isin(holdout_set)].copy()
    train = df[~key.isin(holdout_set)].copy()
    overlap_videos = set(train["video_id"]).intersection(set(holdout["video_id"]))
    overlap_channels = set(train["channel_title"].fillna("__missing_channel__")).intersection(
        set(holdout["channel_title"].fillna("__missing_channel__"))
    )
    info = {
        "name": "channel_disjoint_robustness",
        "role": (
            "Retrain the frozen selected spec so no channel appears on both "
            "sides. Tests channel-specific wording leakage, not a new random "
            "confirmation set of the same videos."
        ),
        "n_train": int(len(train)),
        "n_holdout": int(len(holdout)),
        "n_train_channels": int(len(train_ch)),
        "n_holdout_channels": int(len(holdout_ch)),
        "n_video_overlap": int(len(overlap_videos)),
        "n_channel_overlap": int(len(overlap_channels)),
        "holdout_class_counts": {
            cls: int((holdout["y"] == cls).sum()) for cls in TARGET_CLASSES
        },
        "rare_class_warning": {
            cls: int((holdout["y"] == cls).sum()) < 15
            for cls in ("Gaming", "Sports", "News, Politics & Society")
        },
    }
    return train, holdout, info


def make_temporal_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Earlier publish times train; later times evaluate.

    Robustness to time, not an untouched test set. Videos without a parsed
    timestamp are dropped from this check only.
    """
    dated = df.dropna(subset=["publish_dt"]).copy()
    cutoff = dated["publish_dt"].quantile(TEMPORAL_QUANTILE)
    train = dated[dated["publish_dt"] <= cutoff].copy()
    holdout = dated[dated["publish_dt"] > cutoff].copy()
    overlap = set(train["video_id"]).intersection(set(holdout["video_id"]))
    info = {
        "name": "temporal_robustness",
        "role": (
            "Frozen selected spec, trained on earlier published videos and "
            "scored on later ones. Tests whether language that predicted a "
            "category in 2017 still holds later in the scrape window. Not a "
            "sealed production test."
        ),
        "cutoff_utc": cutoff.isoformat() if pd.notna(cutoff) else None,
        "quantile": TEMPORAL_QUANTILE,
        "n_train": int(len(train)),
        "n_holdout": int(len(holdout)),
        "n_dropped_missing_time": int(df["publish_dt"].isna().sum()),
        "n_video_overlap": int(len(overlap)),
        "holdout_class_counts": {
            cls: int((holdout["y"] == cls).sum()) for cls in TARGET_CLASSES
        },
        "rare_class_warning": {
            cls: int((holdout["y"] == cls).sum()) < 15
            for cls in ("Gaming", "Sports", "News, Politics & Society")
        },
    }
    return train, holdout, info

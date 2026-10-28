"""Read the original UCI data using its documented 36-feature schema."""

import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
TARGET = "outcome"
LABELS = {"nowin": 0, "won": 1}


def feature_names():
    text = (ROOT / "data/raw/kr-vs-kp.names").read_text()
    match = re.search(r"\[bkblk,([^\]]+)\]", text)
    if match is None:
        raise ValueError("The original UCI feature list is missing.")
    names = ["bkblk"] + re.sub(r"\s+", "", match.group(1)).split(",")
    if len(names) != 36 or len(set(names)) != 36:
        raise ValueError("Expected 36 distinct feature names.")
    return names


def load_dataset():
    # header=None prevents the first position being consumed as a header.
    frame = pd.read_csv(ROOT / "data/raw/kr-vs-kp.data", header=None, dtype=str)
    names = feature_names()
    if frame.shape != (3196, 37):
        raise ValueError(f"Unexpected original dataset shape: {frame.shape}")
    frame.columns = names + [TARGET]
    if frame.isna().any().any() or set(frame[TARGET]) != set(LABELS):
        raise ValueError("Unexpected missing values or target labels.")
    if frame.duplicated(subset=names).any():
        raise ValueError("Resolve duplicate feature vectors before splitting.")
    return frame


def category_domains():
    """Input validation vocabulary, separate from the fitted encoder.

    These domains describe the audited data format; they do not supply fitted
    encoder categories or use target labels. The rare spcop=t value can be absent
    from a training fold, so validation must not use that fold's vocabulary.
    """
    domains = {name: ["f", "t"] for name in feature_names()}
    domains.update({"dwipd": ["g", "l"], "katri": ["b", "n", "w"], "wtoeg": ["n", "t"]})
    return domains

"""Audit the original KRKPA7 files using only the Python standard library.

Run from the repository root: python3 scripts/audit_dataset.py
The source files are preserved unchanged; blank lines are ignored when parsing.
"""

import csv
import hashlib
import re
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
SOURCE = "https://archive.ics.uci.edu/dataset/22/chess+king+rook+vs+king+pawn"
BASE = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "chess/king-rook-vs-king-pawn/"
)


def main():
    documentation = (RAW / "kr-vs-kp.names").read_text()
    # The original documentation supplies the ordered names in square brackets.
    match = re.search(r"\[bkblk,([^\]]+)\]", documentation)
    if match is None:
        raise ValueError("Cannot locate the original ordered feature names.")
    features = ["bkblk"] + re.sub(r"\s+", "", match.group(1)).split(",")

    with (RAW / "kr-vs-kp.data").open(newline="") as source:
        parsed = list(csv.reader(source))
    rows = [row for row in parsed if row]
    if len(features) != 36 or any(len(row) != 37 for row in rows):
        raise ValueError("Expected 36 input features and a final target column.")

    labels = Counter(row[-1] for row in rows)
    if set(labels) != {"won", "nowin"}:
        raise ValueError(f"Unexpected targets: {dict(labels)}")
    missing = sum(
        value.strip().lower() in {"", "?", "na", "nan", "null"}
        for row in rows
        for value in row
    )
    unique_rows = set(map(tuple, rows))
    feature_groups = defaultdict(set)
    for row in rows:
        feature_groups[tuple(row[:-1])].add(row[-1])
    conflicts = sum(len(values) > 1 for values in feature_groups.values())
    columns = list(zip(*rows))[:-1]
    constants = [name for name, column in zip(features, columns) if len(set(column)) == 1]

    report = [
        "# KRKPA7 dataset audit",
        "",
        "Generated with `python3 scripts/audit_dataset.py`.",
        "",
        f"Source: [UCI Chess (King-Rook vs. King-Pawn)]({SOURCE}).",
        f"Original files: [data]({BASE}kr-vs-kp.data), "
        f"[documentation]({BASE}kr-vs-kp.names).",
        "Citation: Shapiro, A. (1983). Chess (King-Rook vs. King-Pawn). "
        "UCI Machine Learning Repository. https://doi.org/10.24432/C5DK5C.",
        "UCI lists the dataset under CC BY 4.0; retain attribution when redistributing.",
        "",
        "## Verified file contents",
        "",
        "| Check | Result |",
        "| --- | --- |",
        f"| Data rows | {len(rows)} |",
        f"| Input features | {len(features)} |",
        "| Target | Final (37th) column: won / nowin |",
        f"| Blank CSV records ignored | {len(parsed) - len(rows)} |",
        f"| Missing-value tokens | {missing} |",
        f"| Exact duplicate rows beyond the first | {len(rows) - len(unique_rows)} |",
        f"| Feature vectors with conflicting labels | {conflicts} |",
        f"| Constant input features | {', '.join(constants) or 'None'} |",
        "",
        "## Target distribution",
        "",
        "| Target | Count | Percentage |",
        "| --- | --- | --- |",
    ]
    for label, count in sorted(labels.items()):
        report.append(f"| {label} | {count} | {count / len(rows):.2%} |")
    report += [
        "",
        "## Input schema in original column order",
        "",
        "| Column | Feature | Observed categories |",
        "| --- | --- | --- |",
    ]
    for index, (name, column) in enumerate(zip(features, columns), start=1):
        report.append(f"| {index} | {name} | {', '.join(sorted(set(column)))} |")
    report += [
        "",
        "## Modeling decisions to carry forward",
        "",
        "- Use the original schema above: 36 predictors and the final won/nowin target. "
        "The current UCI landing page has inconsistent feature metadata.",
        "- Define won = 1 (White can win) and nowin = 0 (White cannot win). "
        "These are endgame labels, not observed outcomes of human games.",
        "- Preserve the downloaded files unchanged and fit categorical encoding "
        "inside training pipelines.",
        "- Inspect duplicates before making a stratified training/test split; "
        "identical feature vectors must not occur in both partitions.",
        "- Evaluate all five classifiers on the same held-out rows. "
        "Perform any tuning using only the training partition.",
        "- Feature names are published abbreviations; detailed meanings require "
        "the cited source material. Do not invent expansions.",
        "",
        "## SHA-256 checksums of original files",
        "",
    ]
    for filename in ("kr-vs-kp.data", "kr-vs-kp.names"):
        digest = hashlib.sha256((RAW / filename).read_bytes()).hexdigest()
        report.append(f"- `{filename}`: `{digest}`")
    output = ROOT / "data" / "DATASET_AUDIT.md"
    output.write_text("\n".join(report) + "\n")
    print("\n".join(report[:report.index("## Input schema in original column order")]))
    print(f"Full report: {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

# KRKPA7 dataset audit

Generated with `python3 scripts/audit_dataset.py`.

Source: [UCI Chess (King-Rook vs. King-Pawn)](https://archive.ics.uci.edu/dataset/22/chess+king+rook+vs+king+pawn).
Original files: [data](https://archive.ics.uci.edu/ml/machine-learning-databases/chess/king-rook-vs-king-pawn/kr-vs-kp.data), [documentation](https://archive.ics.uci.edu/ml/machine-learning-databases/chess/king-rook-vs-king-pawn/kr-vs-kp.names).
Citation: Shapiro, A. (1983). Chess (King-Rook vs. King-Pawn). UCI Machine Learning Repository. https://doi.org/10.24432/C5DK5C.
UCI lists the dataset under CC BY 4.0; retain attribution when redistributing.

## Verified file contents

| Check | Result |
| --- | --- |
| Data rows | 3196 |
| Input features | 36 |
| Target | Final (37th) column: won / nowin |
| Blank CSV records ignored | 2 |
| Missing-value tokens | 0 |
| Exact duplicate rows beyond the first | 0 |
| Feature vectors with conflicting labels | 0 |
| Constant input features | None |

## Target distribution

| Target | Count | Percentage |
| --- | --- | --- |
| nowin | 1527 | 47.78% |
| won | 1669 | 52.22% |

## Input schema in original column order

| Column | Feature | Observed categories |
| --- | --- | --- |
| 1 | bkblk | f, t |
| 2 | bknwy | f, t |
| 3 | bkon8 | f, t |
| 4 | bkona | f, t |
| 5 | bkspr | f, t |
| 6 | bkxbq | f, t |
| 7 | bkxcr | f, t |
| 8 | bkxwp | f, t |
| 9 | blxwp | f, t |
| 10 | bxqsq | f, t |
| 11 | cntxt | f, t |
| 12 | dsopp | f, t |
| 13 | dwipd | g, l |
| 14 | hdchk | f, t |
| 15 | katri | b, n, w |
| 16 | mulch | f, t |
| 17 | qxmsq | f, t |
| 18 | r2ar8 | f, t |
| 19 | reskd | f, t |
| 20 | reskr | f, t |
| 21 | rimmx | f, t |
| 22 | rkxwp | f, t |
| 23 | rxmsq | f, t |
| 24 | simpl | f, t |
| 25 | skach | f, t |
| 26 | skewr | f, t |
| 27 | skrxp | f, t |
| 28 | spcop | f, t |
| 29 | stlmt | f, t |
| 30 | thrsk | f, t |
| 31 | wkcti | f, t |
| 32 | wkna8 | f, t |
| 33 | wknck | f, t |
| 34 | wkovl | f, t |
| 35 | wkpos | f, t |
| 36 | wtoeg | n, t |

## Modeling decisions to carry forward

- Use the original schema above: 36 predictors and the final won/nowin target. The current UCI landing page has inconsistent feature metadata.
- Define won = 1 (White can win) and nowin = 0 (White cannot win). These are endgame labels, not observed outcomes of human games.
- Preserve the downloaded files unchanged and fit categorical encoding inside training pipelines.
- Inspect duplicates before making a stratified training/test split; identical feature vectors must not occur in both partitions.
- Evaluate all five classifiers on the same held-out rows. Perform any tuning using only the training partition.
- Feature names are published abbreviations; detailed meanings require the cited source material. Do not invent expansions.

## SHA-256 checksums of original files

- `kr-vs-kp.data`: `70a5540902b4063d3977d3e820ac5b9fe5624eb5d93709eee5d50d7a94fca205`
- `kr-vs-kp.names`: `491da0e0a8756fb8553c62d7eb3bd95333db1ccd5c62819f0e43f92aad614d72`

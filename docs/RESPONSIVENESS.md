# Responsiveness verification

Local verification on 28 October 2026, using the pinned Streamlit environment
and the bundled 640-position test CSV. These checks concern the local app;
Community Cloud and BITS Virtual Lab execution remain separate milestones.

## Interaction performance

Previously every widget rerun parsed the CSV and evaluated all five classifiers.
The app now retains one evaluation in the visitor's session, keyed by CSV content
and saved-model revision. Switching classifiers or filtering prediction errors
reuses that result. Different CSV content or changed models invalidates it.
Invalid uploads and clearing the uploader discard the previous result.

Results are not placed in a shared data cache. The cache contains no other
visitor's uploads, and it never fits or tunes a model. Download buttons use
`on_click="ignore"` to avoid unnecessary reruns.

An AppTest measurement initialized the app, then selected Logistic Regression,
Decision Tree, KNN, Naive Bayes and SVM in that order. Each duration includes
the corresponding script rerun, rather than only prediction time.

| Measurement | Before | After |
| --- | ---: | ---: |
| Median of five model-switch reruns | 0.446 s | 0.091 s |
| All-model evaluations across initialization and five switches | 6 | 1 |

The measured median fell by about 80% on this local machine. This is a small
local benchmark; browser/network latency and cloud hardware can change timings.
Regression tests assert inference call counts and correct invalidation, avoiding
fragile timing thresholds.

## Layout changes

- Metric cards use the available content width, accounting for the sidebar:
  six columns above 900 px, three at intermediate widths, and two at 500 px
  or below. Container queries avoid relying only on the device viewport.
- Narrow layouts stack the confusion matrix and classification report.
- Short tab labels fit phone screens: Evaluation, Compare models, Positions.
- The comparison chart uses short model labels and retains full names in
  tooltips and the comparison table.
- Heading size, top spacing and metric typography are adjusted for small screens.
- A sidebar hint makes the collapsed controls easier to find on phones.

The scoped layout CSS uses Streamlit container keys and test identifiers. If
Streamlit is upgraded, recheck these selectors; the dependency version is pinned.

## Browser verification

Headless Chrome was opened with a separate temporary profile against the local
server. Screenshots and DOM measurements were taken at these viewport sizes:

| Viewport | Expected metric columns |
| --- | ---: |
| 1440 × 1000, sidebar expanded | 6 |
| 1200 × 900, sidebar expanded | 3 |
| 1024 × 900, sidebar expanded | 3 |
| 768 × 1024, sidebar collapsed | 3 |
| 390 × 844, sidebar collapsed | 2 |
| 360 × 800, sidebar collapsed | 2 |

Checks verify that the six cards form the expected rows, all three tab labels
fit within the viewport, each tab can be selected, no app exception is displayed,
and the page has no horizontal overflow. Screenshots were inspected for readable
metric values and spacing. Wide data tables may scroll inside their own view;
this does not require scrolling the entire page sideways.

To repeat the manual review, open browser developer tools, select responsive
device mode, and use the sizes above. Open all tabs, expand/collapse the sidebar,
switch models, toggle error filtering, and upload the example CSV. The evaluation
metrics should match the recorded results for that CSV.

## Automated regression checks

```bash
.venv/bin/python -m unittest discover -s tests -v
```

The suite includes 16 checks, with explicit coverage for session isolation,
data/model invalidation, clearing invalid results, and reusing inference across
model selection and filtering. Existing saved-model, metric, upload and
no-retraining checks continue to pass.

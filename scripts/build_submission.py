"""Build the submission from the current README, manifest and redacted evidence.

Document-only prerequisites: Pandoc and Google Chrome or Chromium.
Run: python scripts/build_submission.py
"""

import argparse
import base64
from datetime import datetime
import hashlib
import html
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
REPO = "https://github.com/tusharviradiya/aiml-ml-classification-assignment-2"
APP = "https://chess-endgame-lab-tushar.streamlit.app/"


def submission_html():
    manifest = json.loads((ROOT / "results/manifest.json").read_text())
    evidence = json.loads((ROOT / "docs/lab_evidence.json").read_text())
    metrics_hash = hashlib.sha256((ROOT / "results/metrics.csv").read_bytes()).hexdigest()
    evidence_matches = (
        evidence["training_source_sha256"] == manifest["source_sha256"]["train.py"]
        and evidence["metrics_sha256"] == metrics_hash
    )
    readme = subprocess.check_output(
        ["pandoc", "--from=gfm", "--to=html5", str(ROOT / "README.md")], text=True,
    )

    def expand_link(match):
        target = match.group(1)
        if target.startswith(("https://", "http://", "mailto:", "#")):
            return match.group(0)
        return f'href="{REPO}/blob/main/{target}"'

    readme = re.sub(r'href="([^"]+)"', expand_link, readme)
    css = (ROOT / "docs/submission.css").read_text()
    date = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%d %B %Y").lstrip("0")
    environment = manifest["execution_environment"]
    source_hash = manifest["source_sha256"]["train.py"][:12]
    evidence_status = (
        "Lab evidence matches the recorded training source and metrics."
        if evidence_matches else
        "The retained lab screenshots document an earlier run. Revised lab verification is pending."
    )
    cover = f'''<section class="cover">
<p class="eyebrow">BITS PILANI · M.TECH (AIML) · MACHINE LEARNING</p>
<h1>Assignment 2</h1><p class="subtitle">Chess Endgame Outcome Classification</p>
<p class="dataset">Dataset: UCI Chess (King-Rook vs. King-Pawn), KRKPA7</p>
<div class="identity"><p><strong>Name:</strong> Viradiya Tushar Rajeshbhai</p>
<p><strong>BITS ID:</strong> 2026aj05073</p><p><strong>Updated:</strong> {date}</p>
<p><strong>Recorded model run:</strong> {html.escape(environment)} · Python {manifest['python']}</p>
<p><strong>Training source SHA-256:</strong> {source_hash}…</p></div>
<div class="link-card"><h2>1. GitHub repository</h2><a href="{REPO}">{REPO}</a>
<p>Source code, dependencies, test data, five saved pipelines and evaluation results.</p></div>
<div class="link-card"><h2>2. Live Streamlit application</h2><a href="{APP}">{APP}</a>
<p>Interactive CSV evaluation, model selection, six metrics and model comparisons.</p></div>
<p class="run-note">{evidence_status} The current model results and validation status appear in the README.</p>
</section>'''

    def figure(item):
        image = base64.b64encode((ROOT / "docs" / item["file"]).read_bytes()).decode()
        return (
            f'<figure><h3>{html.escape(item["title"])}</h3>'
            f'<img src="data:image/png;base64,{image}">'
            f'<figcaption>{html.escape(item["caption"])}</figcaption></figure>'
        )

    label = "Lab run" if evidence_matches else "Earlier lab run"
    evidence1 = (
        '<section class="evidence"><h2>3. BITS Virtual Lab execution evidence</h2>'
        f'<p class="intro">{label}: {html.escape(evidence["captured_date"])} '
        f'· revision {html.escape(evidence["artifact_revision"])}</p>'
        + "".join(figure(item) for item in evidence["figures"][:2]) + "</section>"
    )
    evidence2 = ""
    if evidence["figures"][2:]:
        evidence2 = (
            '<section class="evidence"><h2>3. Supporting lab-session views</h2>'
            + "".join(figure(item) for item in evidence["figures"][2:])
            + '<p class="note">Outer lab browser address bars are redacted. '
            'Execution output is preserved.</p></section>'
        )
    app_preview = (
        '<section class="evidence"><h2>Streamlit application preview</h2>'
        f'<p class="intro">Live application: <a href="{APP}">{APP}</a></p>'
        + figure({
            "file": "streamlit_knn_verification.png",
            "title": "Streamlit app opened from BITS Virtual Lab",
            "caption": "1 November 2026, 13:14. The public Streamlit app is open inside "
            "the BITS Virtual Lab browser. The lab title, public lab hostname and app URL "
            "are preserved; session details are redacted. All six KNN metrics match the "
            "final lab results, including F1 0.9632 and AUC 0.9932. Training execution "
            "is documented on the preceding evidence page.",
        }) + "</section>"
    )
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        f'<title>ML Assignment 2 Submission</title><style>{css}</style></head><body>'
        + cover + evidence1 + evidence2 + app_preview
        + '<main><p class="readme-label">4. Project README</p>' + readme + "</main></body></html>"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--html-only", type=Path, help="Write HTML without launching a browser.")
    parser.add_argument("--output", type=Path, default=ROOT / "ML_Assignment_2_Submission.pdf")
    args = parser.parse_args()
    document = submission_html()
    if args.html_only:
        args.html_only.write_text(document)
        print(args.html_only.resolve())
        return
    browser = next((shutil.which(name) for name in [
        "google-chrome", "google-chrome-stable", "chromium", "chromium-browser",
    ] if shutil.which(name)), None)
    if browser is None:
        raise SystemExit("Install Google Chrome or Chromium to render the PDF.")
    with tempfile.TemporaryDirectory(prefix="ml-submission-") as temporary:
        directory = Path(temporary)
        source = directory / "submission.html"
        source.write_text(document)
        rendered = directory / "submission.pdf"
        subprocess.run([
            browser, "--headless", "--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage",
            "--no-first-run", "--no-default-browser-check", "--no-pdf-header-footer",
            f"--user-data-dir={directory / 'browser'}", f"--print-to-pdf={rendered}", source.as_uri(),
        ], check=True)
        if not rendered.exists():
            raise SystemExit("The browser did not create a PDF; the previous submission was preserved.")
        shutil.copyfile(rendered, args.output)
    print(args.output.resolve())


if __name__ == "__main__":
    main()

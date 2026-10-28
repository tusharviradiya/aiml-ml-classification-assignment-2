# Git workflow for this assignment

Keep each commit focused on one reviewable development step. The initial commits
organize the work already completed locally; they do not claim that these stages
were committed on different days. Commit dates and authorship remain genuine.

## Initial modeling milestones

| Order | Commit message | Contents |
| --- | --- | --- |
| 1 | `chore: configure Python dependencies and Git exclusions` | `.gitignore`, `requirements.txt` |
| 2 | `data: add original KRKPA7 dataset and reproducible audit` | Original files, audit script, report and file-format rules |
| 3 | `feat: implement reproducible classifier training and evaluation` | Data loading, inference validation, five-model training |
| 4 | `feat: add trained pipelines and held-out evaluation artifacts` | Five saved pipelines, split CSVs, metrics, CV results and manifest |
| 5 | `test: verify saved models and held-out evaluation integrity` | Integration checks |
| 6 | `docs: explain dataset, model results and development workflow` | README and this guide |

Saved models and test data are deliberately tracked because the assignment
requires them. The virtual environment, caches, local credentials, temporary
uploads, and supplied assignment handout stay local. `.gitignore` does not
remove files that are already tracked and cannot detect every possible secret;
review staged changes before committing.

`.gitattributes` preserves the original UCI downloads byte-for-byte, including
their existing trailing blank lines. It also keeps code, Markdown and CSV line
endings consistent and marks saved model files as binary.

## Step-by-step routine for each new piece of work

1. Start with a clean working tree and synchronize `main`:

   ```bash
   git status
   git switch main
   git pull --ff-only origin main
   ```

   If files are modified, finish and commit the current step before pulling.
   If a fast-forward fails, inspect the divergence before proceeding.

2. Create a short feature branch for the next substantial task:

   ```bash
   git switch -c feat/streamlit-evaluation
   ```

3. Implement one coherent change and run the checks relevant to it. For the
   modeling workflow, activate the environment and run:

   ```bash
   source .venv/bin/activate
   python -m unittest discover -s tests -v
   python -m pip check
   ```

4. Inspect changes and stage explicit paths. For example, once `app.py` exists:

   ```bash
   git status --short
   git diff
   git add app.py requirements.txt
   git diff --cached --stat
   git diff --cached
   git diff --cached --check
   ```

   Normal `git diff` does not display untracked file contents; inspect those
   after staging with `git diff --cached`. To unstage an unintended file after
   the first commit, use `git restore --staged path/to/file`; it stays on disk.

5. Commit with a message describing the behavior added or fixed:

   ```bash
   git commit -m "feat: evaluate uploaded chess positions in Streamlit"
   ```

   Useful prefixes: `feat` for functionality, `fix` for defects, `data` for
   dataset changes, `test` for checks, `docs` for explanation, and `chore` for
   configuration. A body can explain an important decision, such as why the
   encoder handles a rare category. Avoid messages such as `update` or `final`.

6. Push the branch and open a GitHub pull request into `main`:

   ```bash
   git push -u origin feat/streamlit-evaluation
   ```

   Review the changed files and validation evidence. To retain the individual
   development commits, use GitHub's **Create a merge commit** option when it is
   available. Keep required deployment files on `main` after merging.

7. Verify the history and repository on GitHub:

   ```bash
   git log --oneline --graph --decorate -12
   git status --short --branch
   ```

   Confirm that the public repository contains source code, requirements, README,
   all five saved models, and the test CSV. Check that no credentials or caches
   were included. Never force-push shared assignment history as a routine step.

## Direct pushes for small solo changes

A feature branch is useful for a substantial feature but is optional for a small
solo documentation change. After reviewing and committing on `main`, use:

```bash
git push origin main
```

The initial local milestones are published with `git push -u origin main` once
the remote has been checked. A normal push preserves existing remote history
and rejects a conflicting update instead of overwriting it.

## Next milestones

- Streamlit CSV upload, model selection, metrics and model comparison.
- App error handling, confusion matrix, and interaction checks.
- Community Cloud deployment configuration and run instructions.
- BITS Virtual Lab reproduction and the genuine execution screenshot.
- Final README, live link, and submission PDF.

Commit these when the corresponding work is actually completed. An accurate,
understandable history matters more than a large number of commits.

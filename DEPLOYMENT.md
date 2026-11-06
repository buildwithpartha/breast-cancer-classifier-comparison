# Deployment and final submission

## Local or BITS Virtual Lab execution

Use Python 3.13 to match the local environment. On Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python train.py
.\.venv\Scripts\python verify.py
.\.venv\Scripts\python -m streamlit run app.py
```

On a Linux lab machine, use `.venv/bin/python` in place of `.venv\Scripts\python`.

Training regenerates the models, test CSV and metrics. Run it in the lab and inspect the outputs. Capture a genuine screenshot showing execution within the lab environment. Merely opening the hosted app may not satisfy the assignment's execution requirement.

## GitHub

Repository: https://github.com/buildwithpartha/breast-cancer-classifier-comparison

The repository has been created. Publish this project folder, excluding `.venv`. Include the saved `model/*.joblib` files, metadata, test CSV, results and requirements. Verify the link is accessible to the evaluator.

## Streamlit Community Cloud

1. Sign in with GitHub at https://share.streamlit.io/.
2. Create an app using your repository, the `main` branch, and entry point `app.py`.
3. Select Python 3.13, matching the trained artifacts and pinned dependencies.
4. Deploy and check the build logs.
5. Open the live app and test all model choices, download the sample CSV, and upload it again.
6. Add the real GitHub and Streamlit URLs to README.md and the submission document.

## Final PDF

Assemble one PDF in this order: clickable GitHub link, clickable live Streamlit link, actual lab screenshot, complete README content. A final PDF cannot be completed truthfully until these links and the screenshot exist.

Check the links in the exported PDF, verify all five comparison rows are legible, and submit before 8 November 2026, 23:59. Do not leave the submission as a draft.

# CellScope — Breast Cancer Classifier Comparison

Five classification algorithms compared using a common held-out dataset, with a Streamlit interface for evaluating saved models on labeled test CSV files.

## a. Problem statement

Predict whether a breast cell sample is benign or malignant from measurements derived from a digitized fine-needle aspirate image. Compare Logistic Regression, Decision Tree, K-Nearest Neighbors, Gaussian Naive Bayes and Support Vector Machine using six evaluation metrics. This is an educational experiment, not a clinical diagnostic system.

## b. Dataset description

- **Source:** [UCI Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic), [DOI: 10.24432/C5DW2B](https://doi.org/10.24432/C5DW2B). Loaded using scikit-learn's bundled version; no external dataset download is needed at runtime.
- **Size:** 569 samples and 30 numerical input features, exceeding the minimum of 500 samples and 12 features.
- **Inputs:** mean, standard error and worst measurements of radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry and fractal dimension. One row represents one sample. Identifiers are not used.
- **Classes:** 357 benign and 212 malignant samples. The bundled inputs have no missing values or duplicate rows.
- **Encoding:** `0 = benign`, `1 = malignant`. The training script reverses scikit-learn's original encoding. Uploaded CSVs must follow this project's convention.
- **Split:** stratified 80/20 split, random state 42; 455 training samples and 114 held-out samples (72 benign, 42 malignant). `test_data.csv` contains only held-out samples and labels.
- **Preprocessing:** StandardScaler inside the Logistic Regression, KNN, Gaussian NB and SVM pipelines; the tree uses unscaled measurements. Scalers learn from training data only, including within cross-validation folds. No imputation is needed; missing or nonfinite uploaded values are rejected.
- **Selection:** fixed settings followed by five-fold stratified training-only cross-validation, using mean malignant-class F1. Test results are for reporting, not parameter tuning.

## c. GitHub repository link

[GitHub source and saved artifacts](https://github.com/buildwithpartha/breast-cancer-classifier-comparison)

**Live Streamlit app:** [Open CellScope](https://breast-cancer-classifier-comparision.streamlit.app/).

## d. Models used

Settings: Logistic Regression (`max_iter=3000`); Decision Tree (`max_depth=5`, `min_samples_leaf=3`); KNN (`n_neighbors=7`); Gaussian Naive Bayes; SVM (RBF kernel, `C=1`, probability estimation enabled). Random state 42 is used where applicable; other settings use the pinned scikit-learn defaults. Settings were not optimized against the test set.

### Held-out test comparison

All models use the same 114 test samples. Precision, recall and F1 use malignant as the positive class. AUC is binary ROC AUC from malignant probabilities. Undefined precision/recall/F1 use zero; AUC is unavailable for single-class uploads. MCC uses predicted labels.

| ML model name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.9649 | 0.9960 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree | 0.9035 | 0.9471 | 0.9429 | 0.7857 | 0.8571 | 0.7926 |
| KNN | 0.9561 | 0.9825 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Naive Bayes | 0.9211 | 0.9891 | 0.9231 | 0.8571 | 0.8889 | 0.8292 |
| Support Vector Machine | 0.9737 | 0.9947 | 1.0000 | 0.9286 | 0.9630 | 0.9442 |

### Observations

| ML model name | Observation about model performance |
| --- | --- |
| Logistic Regression | Highest test AUC (0.9960), detecting 39 of 42 malignant samples with one false positive. A scaled linear boundary works well on this split. It also has the highest mean training CV F1 (0.9640). |
| Decision Tree | Lowest test F1 (0.8571) and recall (0.7857), missing 9 of 42 malignant samples. The restricted tree offers simple rules but performs less well here than the linear and kernel models. This result alone cannot distinguish underfitting from split variability. |
| KNN | Accuracy 0.9561 and precision 0.9744, detecting 38 of 42 malignant samples with one false positive. Scaling prevents large-unit features from dominating distances. It misses one more malignant sample than Logistic Regression. |
| Naive Bayes | AUC 0.9891 is high, while recall at its decision threshold is 0.8571 (36 of 42 malignant samples). Strong ranking need not yield equally strong thresholded predictions. Correlated measurements may limit the conditional-independence approximation; this experiment does not establish that cause. |
| Support Vector Machine | Best held-out accuracy, F1 and MCC: 39 true positives, 72 true negatives, no false positives and 3 false negatives. The nonlinear boundary works well on this split. Precision of 1.0 here does not imply perfect future precision. |
| Overall winner for your dataset? | **SVM wins the held-out comparison by F1 (0.9630). Logistic Regression is selected by training-only CV F1 (0.9640 versus SVM 0.9610).** Keeping these separate avoids choosing a model by repeatedly inspecting test results. A small single test split does not establish general superiority. |

Mean training CV F1: Logistic Regression 0.9640, Decision Tree 0.8963, KNN 0.9575, Naive Bayes 0.9170, SVM 0.9610. Full-precision results are in `results/`.

## Run locally or in BITS Virtual Lab

Tested with Python 3.13.5. Use Python 3.13 and pinned dependencies for consistent serialized model loading.

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python train.py
.\.venv\Scripts\python verify.py
.\.venv\Scripts\python -m streamlit run app.py
```

On Linux, use `.venv/bin/python` instead of `.venv\Scripts\python`. Committed artifacts let the app run without retraining; `train.py` reproduces them. SVM probability estimation is supported by the pinned version but emits a future-deprecation warning during training.

## App usage

The app initially evaluates the bundled test CSV. Select a classifier or upload a labeled CSV with all 30 feature names in `model/metadata.json` and the `target` column. Input column order is restored automatically; extra columns are ignored for prediction. Missing columns, missing values, nonnumeric inputs and invalid targets are rejected.

The six metrics, confusion matrix, classification report and five-model comparison update on the same rows. Download results and predictions as CSV. Single-class uploads show AUC as unavailable. Only trusted committed model artifacts are loaded; the app never trains on uploaded data.

## Repository contents

```text
app.py                       Streamlit interface
train.py                     Offline training and cross-validation
evaluation.py                CSV validation and evaluation
verify.py                    Saved-model, preprocessing and app checks
requirements.txt             Pinned direct dependencies
test_data.csv                Labeled held-out samples
model/*.joblib               Five fitted pipelines
model/metadata.json          Features, labels, versions and model mapping
results/metrics.csv          Full-precision held-out metrics
results/cross_validation.csv  Training-only CV results
prepare_submission.py        Printable submission preparation
DEPLOYMENT.md                Deployment and PDF instructions
RUBRIC_CHECKLIST.md           Assignment requirements
```

## Deployment and submission

Deploy `app.py` from `main` through Streamlit Community Cloud using Python 3.13. Follow `DEPLOYMENT.md`. Run the assignment on BITS Virtual Lab and capture a genuine execution screenshot. Local execution does not establish lab compliance.

The PDF must contain the GitHub link, live app link, actual lab screenshot and this README in that order. Once the real links and screenshot exist, run:

```text
python prepare_submission.py --repo https://github.com/buildwithpartha/breast-cancer-classifier-comparison --app https://breast-cancer-classifier-comparision.streamlit.app/ --screenshot YOUR_LAB_SCREENSHOT.png
```

Open `submission.html`, print to PDF, and check its links and table legibility. The app has been deployed; BITS Lab execution evidence and final submission remain pending.

# Understand and explain CellScope

## The complete flow

`train.py` loads the data, reverses the target encoding, creates one stratified split, compares five pipelines using training-only cross-validation, fits the pipelines on all training rows, evaluates them on held-out rows and saves artifacts.

`app.py` loads those fitted artifacts, accepts a labeled test CSV, checks its schema, predicts and calculates metrics. It never calls `fit`. `evaluation.py` keeps the metric definitions consistent between training and the app. `verify.py` checks saved scores, scaler fitting boundaries, input validation and app model switching.

## Inputs and target

The 30 inputs describe cell geometry and texture. Ten measurement types each have a mean, an error estimate and a worst-value summary. Features are the evidence the model sees. The target is the known diagnosis used for learning and evaluation.

This project uses **1 = malignant**. The scikit-learn loader originally uses 0 = malignant, so the training script deliberately reverses it. An uploaded file using the original convention would produce misleading metrics even though its numeric labels look valid. Always check label meaning, not just label values.

## Why split before preprocessing?

The test set represents unseen samples. If a scaler learned means or standard deviations from the entire dataset, test information would influence training. A pipeline fits the scaler on training inputs and applies that same transformation to test inputs. In cross-validation, each fold gets its own fitted pipeline.

Stratification approximately preserves the proportion of benign and malignant samples. The random seed makes the split reproducible. Changing the target encoding can change a stratified split for the same seed because the grouping order changes; the verification code therefore reproduces the exact encoding used in training.

## What each model learns

| Model | How it decides | Main settings and tradeoffs |
| --- | --- | --- |
| Logistic Regression | Fits a linear score and maps it to a probability | Scaled features, regularization and enough optimization iterations; a linear boundary may miss nonlinear patterns |
| Decision Tree | Recursively splits features into rules | Depth 5 and at least 3 rows per leaf restrict complexity; trees can otherwise overfit |
| KNN | Uses labels of 7 nearest training samples | Scaling makes distances more meaningful; larger k smooths decisions, smaller k reacts more to local noise |
| Gaussian Naive Bayes | Combines class priors and Gaussian feature likelihoods | Assumes conditional feature independence; correlated inputs can violate the approximation |
| SVM | Uses a large-margin decision boundary with an RBF kernel | C=1 balances margin and training errors; probability estimates are enabled for evaluation |

Scaling does not make features independent. It changes units; it does not remove correlation. Decision trees generally do not need it because their splits depend on feature orderings.

## Read the confusion matrix

Rows are true labels and columns are predictions. Benign is class 0; malignant is class 1.

| | Predicted benign | Predicted malignant |
| --- | --- | --- |
| Actually benign | True negative (TN) | False positive (FP) |
| Actually malignant | False negative (FN) | True positive (TP) |

For the held-out SVM result: TN=72, FP=0, FN=3, TP=39.

- Accuracy = (TP + TN) / all samples = 111/114 = 0.9737.
- Precision = TP / (TP + FP) = 39/39 = 1.0000.
- Recall = TP / (TP + FN) = 39/42 = 0.9286.
- F1 = 2TP / (2TP + FP + FN) = 78/81 = 0.9630.
- MCC = (TP×TN − FP×FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN)) = 0.9442.
- ROC AUC measures ranking across thresholds using probability scores. It cannot be calculated from this single confusion matrix.

A false negative misses a malignant sample. A false positive flags a benign sample. Which tradeoff is preferable depends on the use case; this educational project does not establish clinical suitability.

## Why AUC and F1 can disagree

AUC measures how well a model orders malignant samples above benign samples across thresholds. F1 uses a particular hard classification decision. Naive Bayes has a high AUC here but lower F1, demonstrating that ranking and thresholded decisions answer different questions. Any threshold adjustment must be selected using training/validation data rather than the held-out test set.

## Why two different winners are reported

Logistic Regression has the highest mean F1 in five-fold training-only cross-validation. That is the prespecified model-selection result. SVM has the highest F1 on the final held-out test set. That is the observed test comparison result. Choosing parameters or repeatedly choosing models using the test set would turn it into another validation set and make its performance estimate optimistic.

The test set has only 42 malignant samples: one additional missed malignant sample changes recall by about 0.0238. Small score differences need cautious interpretation.

## What saving a model means

Joblib stores the fitted estimator and its preprocessing pipeline. The app loads the same transformation and decision rule used during evaluation. Dependency versions matter when loading serialized Python objects, which is why requirements are pinned. Load only trusted model artifacts: joblib is not a safe format for arbitrary uploaded files.

## Before presenting or submitting

Be able to explain the target encoding, why scaling is inside pipelines, which rows are held out, what each metric measures, why SVM and Logistic Regression win different comparisons, and why the app must not train on uploaded test data. Run the commands yourself in BITS Lab and retain genuine execution evidence.


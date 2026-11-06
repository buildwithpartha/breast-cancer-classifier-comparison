# Rubric and submission checklist

## Dataset and documentation — 1 mark

- [x] Public UCI or Kaggle source cited.
- [x] At least 500 rows and 12 input features, excluding the target and identifiers.
- [x] Problem, features, target, class counts, and preprocessing explained.

## GitHub repository — 1 mark

- [x] Repository link works for the evaluator.
- [x] Complete implementation source included.
- [x] app.py or streamlit_app.py included.
- [x] Tested requirements.txt included.
- [x] README follows sections a–d required by the assignment.
- [x] Labeled test_data.csv included.
- [x] Five saved models and fitted preprocessing artifacts included under model/.
- [x] Genuine incremental commits reflect your work.

## Models and metrics — 5 marks

- [x] Logistic Regression: all six metrics.
- [x] Decision Tree: all six metrics.
- [x] KNN: all six metrics.
- [x] Gaussian or Multinomial Naive Bayes: all six metrics.
- [x] SVM: all six metrics.
- [x] Metrics are Accuracy, AUC, Precision, Recall, F1, and MCC.
- [x] All models use the same dataset and held-out test split.
- [x] Preprocessing is fitted using training data only.
- [x] AUC uses continuous scores or probabilities with the correct positive class.

## Observations — 3 marks

- [x] Explain the measured performance of each model in your own words.
- [x] Name an overall winner and justify the criterion.
- [x] Include both comparison and observation tables in the README.

## Streamlit app — 4 marks

- [x] Labeled test CSV upload works — 1 mark.
- [x] Model selection dropdown works — 1 mark.
- [x] Evaluation metrics display correctly — 1 mark.
- [x] Confusion matrix or classification report displays — 1 mark.
- [x] Results for different models are visible.
- [x] Saved models are loaded; uploaded data is never used for fitting.
- [x] Missing columns, invalid targets, and invalid numeric values produce useful messages.
- [x] One-class uploads explain why AUC is unavailable.
- [x] Live Streamlit Community Cloud link opens and works.

## BITS Virtual Lab — 1 mark

- [x] Run the assignment implementation on BITS Virtual Lab.
- [x] Capture a real screenshot proving execution there.

## Final PDF — deadline 8 November 2026, 23:59

Include in this order:

1. [x] Clickable GitHub repository link.
2. [x] Clickable live Streamlit app link.
3. [x] BITS Virtual Lab execution screenshot.
4. [x] Full required README content with actual results and observations.

- [x] Reopen the PDF and check both links.
- [ ] Submit the final PDF; do not leave a draft submission.

The rubric totals 15 marks. This checklist supports completeness; grading remains with the evaluator.

# Rubric and submission checklist

## Dataset and documentation — 1 mark

- [ ] Public UCI or Kaggle source cited.
- [ ] At least 500 rows and 12 input features, excluding the target and identifiers.
- [ ] Problem, features, target, class counts, and preprocessing explained.

## GitHub repository — 1 mark

- [ ] Repository link works for the evaluator.
- [ ] Complete implementation source included.
- [ ] app.py or streamlit_app.py included.
- [ ] Tested requirements.txt included.
- [ ] README follows sections a–d required by the assignment.
- [ ] Labeled test_data.csv included.
- [ ] Five saved models and fitted preprocessing artifacts included under model/.
- [ ] Genuine incremental commits reflect your work.

## Models and metrics — 5 marks

- [ ] Logistic Regression: all six metrics.
- [ ] Decision Tree: all six metrics.
- [ ] KNN: all six metrics.
- [ ] Gaussian or Multinomial Naive Bayes: all six metrics.
- [ ] SVM: all six metrics.
- [ ] Metrics are Accuracy, AUC, Precision, Recall, F1, and MCC.
- [ ] All models use the same dataset and held-out test split.
- [ ] Preprocessing is fitted using training data only.
- [ ] AUC uses continuous scores or probabilities with the correct positive class.

## Observations — 3 marks

- [ ] Explain the measured performance of each model in your own words.
- [ ] Name an overall winner and justify the criterion.
- [ ] Include both comparison and observation tables in the README.

## Streamlit app — 4 marks

- [ ] Labeled test CSV upload works — 1 mark.
- [ ] Model selection dropdown works — 1 mark.
- [ ] Evaluation metrics display correctly — 1 mark.
- [ ] Confusion matrix or classification report displays — 1 mark.
- [ ] Results for different models are visible.
- [ ] Saved models are loaded; uploaded data is never used for fitting.
- [ ] Missing columns, invalid targets, and invalid numeric values produce useful messages.
- [ ] One-class uploads explain why AUC is unavailable.
- [ ] Live Streamlit Community Cloud link opens and works.

## BITS Virtual Lab — 1 mark

- [ ] Run the assignment implementation on BITS Virtual Lab.
- [ ] Capture a real screenshot proving execution there.

## Final PDF — deadline 8 November 2026, 23:59

Include in this order:

1. [ ] Clickable GitHub repository link.
2. [ ] Clickable live Streamlit app link.
3. [ ] BITS Virtual Lab execution screenshot.
4. [ ] Full required README content with actual results and observations.

- [ ] Reopen the PDF and check both links.
- [ ] Submit the final PDF; do not leave a draft submission.

The rubric totals 15 marks. This checklist supports completeness; grading remains with the evaluator.

# Understanding the implementation

The project now implements the workflow below. Work through it in order to understand the code and explain each decision. Execution in BITS Lab and deployment still require completion.

1. Create a local Python environment and install the dependencies needed for data inspection. Record the Python version. Check compatibility with the lab and deployment environment.
2. Load the proposed dataset. Inspect shape, feature names, target mapping, missing values, duplicates, and class counts. Explain what one row means.
3. Choose and document the positive class. Separate inputs and target, and create a reproducible stratified train/test split. Save only held-out rows with their target to test_data.csv.
4. Implement one Logistic Regression pipeline with scaling learned from training data only. Explain why scaling matters and how a pipeline avoids leakage.
5. Calculate the six metrics on the held-out split. Explain false positives and false negatives for your encoding. Check which class each probability column represents.
6. Implement the other four classifiers. Reuse the same split; choose preprocessing appropriate to each model. Use training-only cross-validation if tuning parameters.
7. Save complete fitted pipelines under model/. Record feature order, target semantics, model names, dependency versions, and experiment settings.
8. Reload the artifacts in a fresh process and verify the predictions match those before saving.
9. Implement the Streamlit app. Begin with loading test_data.csv and one saved model; then add upload validation, model selection, metrics, confusion matrix, and model comparison.
10. Write observations from actual results, finish the README, and verify a clean installation before GitHub and Streamlit deployment.
11. Run the implementation in BITS Virtual Lab, take the required screenshot, and assemble the submission PDF.

Ask for explanation, review, or debugging at any step. Review the assignment's academic-integrity requirements and make sure you understand and can explain the implementation you submit.

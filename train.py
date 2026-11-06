"""Run offline to reproduce all five pipelines and held-out results."""
import json
from pathlib import Path
import platform
import joblib
import pandas as pd
import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from evaluation import evaluate

ROOT = Path(__file__).resolve().parent


def main():
    dataset = load_breast_cancer(as_frame=True)
    X = dataset.data
    # Reverse sklearn's encoding so recall refers to malignant samples.
    y = (dataset.target == 0).astype(int).rename('target')
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    models = {
        'Logistic Regression': ('logistic_regression', make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, random_state=42))),
        'Decision Tree': ('decision_tree', make_pipeline(DecisionTreeClassifier(max_depth=5, min_samples_leaf=3, random_state=42))),
        'KNN': ('knn', make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7))),
        'Naive Bayes': ('naive_bayes', make_pipeline(StandardScaler(), GaussianNB())),
        'Support Vector Machine': ('svm', make_pipeline(StandardScaler(), SVC(C=1.0, kernel='rbf', probability=True, random_state=42))),
    }
    (ROOT / 'model').mkdir(exist_ok=True)
    (ROOT / 'results').mkdir(exist_ok=True)
    pd.concat([X_test, y_test], axis=1).to_csv(ROOT / 'test_data.csv', index=False)
    rows, validation = [], []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for name, (filename, model) in models.items():
        cv_results = cross_validate(model, X_train, y_train, cv=cv,
                                    scoring={'f1': 'f1', 'auc': 'roc_auc'})
        validation.append({'Model': name, 'CV F1': cv_results['test_f1'].mean(),
                           'CV AUC': cv_results['test_auc'].mean()})
        model.fit(X_train, y_train)
        metrics, _ = evaluate(model, X_test, y_test)
        rows.append({'Model': name, **metrics})
        joblib.dump(model, ROOT / 'model' / (filename + '.joblib'))
    pd.DataFrame(rows).to_csv(ROOT / 'results' / 'metrics.csv', index=False)
    pd.DataFrame(validation).to_csv(ROOT / 'results' / 'cross_validation.csv', index=False)
    selected = max(validation, key=lambda item: item['CV F1'])['Model']
    metadata = {'features': X.columns.tolist(), 'target': {'0': 'benign', '1': 'malignant'},
                'models': {name: filename + '.joblib' for name, (filename, _) in models.items()},
                'seed': 42, 'train_rows': len(X_train), 'test_rows': len(X_test),
                'selected_by_training_cv_f1': selected,
                'python': platform.python_version(), 'sklearn': sklearn.__version__}
    (ROOT / 'model' / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(pd.DataFrame(rows).to_string(index=False))
    print('Selected by training-only CV F1:', selected)


if __name__ == '__main__':
    main()

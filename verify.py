"""Meaningful checks for saved artifacts, leakage boundaries and app behavior."""
import json
from pathlib import Path
import unittest
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from streamlit.testing.v1 import AppTest
from evaluation import evaluate, validate_data

ROOT = Path(__file__).resolve().parent


class ProjectChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.meta = json.loads((ROOT / 'model/metadata.json').read_text())
        cls.frame = pd.read_csv(ROOT / 'test_data.csv')
        cls.X, cls.y = validate_data(cls.frame, cls.meta['features'])

    def test_artifacts_and_recorded_scores(self):
        recorded = pd.read_csv(ROOT / 'results/metrics.csv').set_index('Model')
        for name, filename in self.meta['models'].items():
            model = joblib.load(ROOT / 'model' / filename)
            metrics, predictions = evaluate(model, self.X, self.y)
            self.assertEqual(len(predictions), len(self.X))
            for metric, value in metrics.items():
                self.assertAlmostEqual(value, recorded.loc[name, metric], places=10)

    def test_training_only_scaling(self):
        data = load_breast_cancer(as_frame=True)
        train, test, _, _ = train_test_split(data.data, (data.target == 0).astype(int),
                                            stratify=(data.target == 0).astype(int), test_size=.2, random_state=42)
        np.testing.assert_allclose(self.X, test)
        for filename in ['logistic_regression.joblib', 'knn.joblib', 'naive_bayes.joblib', 'svm.joblib']:
            model = joblib.load(ROOT / 'model' / filename)
            np.testing.assert_allclose(model[0].mean_, train.mean())

    def test_bad_inputs_and_column_order(self):
        for bad in [self.frame.drop(columns=['target']), self.frame.iloc[:0],
                    self.frame.assign(target=2), self.frame.assign(target=np.nan)]:
            with self.assertRaises(ValueError):
                validate_data(bad, self.meta['features'])
        ordered, _ = validate_data(self.frame[self.frame.columns[::-1]], self.meta['features'])
        pd.testing.assert_frame_equal(ordered, self.X)

    def test_one_class_auc(self):
        subset = self.y == 1
        for filename in self.meta['models'].values():
            model = joblib.load(ROOT / 'model' / filename)
            self.assertIsNone(evaluate(model, self.X[subset], self.y[subset])[0]['AUC'])

    def test_streamlit_all_models(self):
        app = AppTest.from_file(str(ROOT / 'app.py')).run(timeout=60)
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.metric), 6)
        for name in self.meta['models']:
            app.selectbox[0].select(name).run(timeout=60)
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(len(app.metric), 6)


if __name__ == '__main__':
    unittest.main(verbosity=2)

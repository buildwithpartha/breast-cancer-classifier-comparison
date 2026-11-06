"""Streamlit inference app. Uploaded data is never used to fit models."""
import json
import os
import tempfile
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'cellscope-matplotlib'))
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from evaluation import evaluate, validate_data

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title='CellScope | Classifier comparison', page_icon='🔬', layout='wide')
st.title('CellScope')
st.caption('Five perspectives on breast cell measurements • ML Assignment 2')
st.info('Educational classifier comparison. Target: 0 = benign, 1 = malignant. This app is not a clinical diagnostic tool.')


@st.cache_resource
def load_artifacts():
    metadata = json.loads((ROOT / 'model' / 'metadata.json').read_text(encoding='utf-8'))
    models = {name: joblib.load(ROOT / 'model' / filename)
              for name, filename in metadata['models'].items()}
    return metadata, models


try:
    metadata, models = load_artifacts()
except Exception as error:
    st.error(f'Cannot load saved models. Run python train.py first. Details: {error}')
    st.stop()

with st.sidebar:
    st.header('Experiment controls')
    selected = st.selectbox('Choose classifier', list(models))
    upload = st.file_uploader('Upload labeled test CSV', type=['csv'])
    st.download_button('Download held-out test CSV', (ROOT / 'test_data.csv').read_bytes(),
                       file_name='test_data.csv', mime='text/csv')
    st.caption('Upload only test data. Include all 30 input columns and target. No training happens here.')

try:
    frame = pd.read_csv(upload if upload is not None else ROOT / 'test_data.csv')
    X, y = validate_data(frame, metadata['features'])
except Exception as error:
    st.error(f'Please check the CSV: {error}')
    st.stop()

st.subheader('Uploaded test data' if upload is not None else 'Bundled held-out test data')
st.write(f'{len(X)} samples · {X.shape[1]} features · {(y == 1).sum()} malignant · {(y == 0).sum()} benign')
if y.nunique() < 2:
    st.warning('AUC is unavailable because this test file contains only one target class. Other metrics use malignant as the positive class; undefined precision/recall/F1 are shown as zero.')
metrics, predictions = evaluate(models[selected], X, y)
st.subheader(selected)
for column, (label, value) in zip(st.columns(6), metrics.items()):
    column.metric(label, 'N/A' if value is None else f'{value:.4f}')
left, right = st.columns(2)
with left:
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y, predictions, labels=[0, 1],
        display_labels=['Benign', 'Malignant'], cmap='Blues', colorbar=False, ax=ax)
    ax.set_title('Actual versus predicted')
    st.pyplot(fig)
    plt.close(fig)
with right:
    st.markdown('**Classification report**')
    report = classification_report(y, predictions, labels=[0, 1],
        target_names=['Benign', 'Malignant'], output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).T, width='stretch')

st.subheader('Compare all five classifiers on these same rows')
comparison = pd.DataFrame([{'Model': name, **evaluate(model, X, y)[0]}
                           for name, model in models.items()]).set_index('Model')
st.dataframe(comparison.style.format('{:.4f}', na_rep='N/A'), width='stretch')
st.download_button('Download comparison results', comparison.to_csv().encode(),
                   file_name='comparison.csv', mime='text/csv')
with st.expander('Explore inputs and predictions'):
    output = frame.copy()
    output['predicted_target'] = predictions
    st.dataframe(output, width='stretch')
    st.download_button('Download predictions', output.to_csv(index=False).encode(),
                       file_name='predictions.csv', mime='text/csv')
with st.expander('How to interpret this experiment'):
    st.write('All pipelines were fitted offline on a stratified 80% training split. '
             'Scaling is learned only from training data. The bundled CSV is the untouched 20% test split. '
             'Precision, recall and F1 refer to malignant samples (class 1); AUC uses malignant probabilities. '
             'Model selection uses five-fold training-only cross-validation F1. '
             'Repeatedly using test scores to tune models would invalidate this held-out evaluation.')
    st.write('Training CV selection:', metadata['selected_by_training_cv_f1'])

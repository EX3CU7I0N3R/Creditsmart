import os
import subprocess
from pathlib import Path

import pandas as pd
import streamlit as st

ARTIFACTS_DIR = Path("artifacts")
UPLOAD_PATH = Path("uploaded_data.csv")
PIPELINE_SCRIPT = Path("creditsmart_pipeline.py")

st.set_page_config(page_title="CreditSmart Risk Modeling", layout="wide")

st.title("CreditSmart: Credit Risk Modeling Dashboard")
st.caption("Predictive modeling, ensemble learning, and explainability for loan risk assessment")

uploaded_file = st.sidebar.file_uploader("Upload loan dataset", type=["csv"])

if uploaded_file is None:
    st.info("Upload a CSV file with a `not.fully.paid` target column to run the model pipeline.")
    st.stop()

loan_records = pd.read_csv(uploaded_file)
st.subheader("Dataset Preview")
st.dataframe(loan_records.head(), use_container_width=True)

UPLOAD_PATH.write_bytes(uploaded_file.getbuffer())

run_pipeline = st.sidebar.button("Execute Model Pipeline")

if not run_pipeline:
    st.stop()

ARTIFACTS_DIR.mkdir(exist_ok=True)

st.write("Running CreditSmart analysis... please wait.")
st.divider()

command = [
    os.sys.executable,
    str(PIPELINE_SCRIPT),
    "--data",
    str(UPLOAD_PATH),
    "--artifacts-dir",
    str(ARTIFACTS_DIR),
]

process = subprocess.Popen(
    command,
    shell=False,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
)

logs_placeholder = st.empty()
logs = ""

if process.stdout is not None:
    for line in process.stdout:
        logs += line
        logs_placeholder.text_area("Execution Log", logs, height=300)

process.wait()

if process.returncode != 0:
    st.error("Model execution failed. See the execution log for details.")
    st.stop()

st.success("Model execution completed successfully.")
st.divider()

metrics_path = ARTIFACTS_DIR / "metrics_table.csv"
if metrics_path.exists():
    st.subheader("Model Performance Summary")
    metrics_table = pd.read_csv(metrics_path)
    st.dataframe(metrics_table, use_container_width=True)

st.subheader("Model Diagnostics")
diagnostics = {
    "roc_curve.png": "ROC Curve",
    "pr_curve.png": "Precision-Recall Curve",
    "confusion_matrix.png": "Confusion Matrix",
}
diagnostic_columns = st.columns(3)

for index, (file_name, title) in enumerate(diagnostics.items()):
    image_path = ARTIFACTS_DIR / file_name
    if image_path.exists():
        diagnostic_columns[index].image(str(image_path), caption=title, use_container_width=True)

st.subheader("Feature Importance and Explainability")
explainability_columns = st.columns(2)
explainability_files = ["shap_summary.png", "shap_beeswarm.png", "pfi_importance.png"]

for index, file_name in enumerate(explainability_files):
    image_path = ARTIFACTS_DIR / file_name
    if image_path.exists():
        caption = file_name.replace("_", " ").replace(".png", "")
        explainability_columns[index % 2].image(str(image_path), caption=caption, use_container_width=True)

if metrics_path.exists():
    st.download_button(
        label="Download Results (CSV)",
        data=metrics_path.read_bytes(),
        file_name="creditsmart_results.csv",
        mime="text/csv",
    )

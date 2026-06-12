import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.append(
    str(PROJECT_ROOT)
)
import streamlit as st
import pandas as pd
import json
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go

from src.serving.inference import (
    run_inference
)

from src.ingestion.ingest import (
    run_ingestion
)

from src.validation.validate import (
    run_validation
)

from src.transformation.feature_pipeline import (
    run_feature_engineering
)

from src.transformation.preprocessing import (
    run_preprocessing
)

from src.training.train import (
    run_training
)

from src.training.random_search import (
    run_random_search
)

from src.training.threshold_tuning import (
    run_threshold_tuning
)

from src.evaluation.evaluate import (
    run_evaluation
)

from src.evaluation.shap_analysis import (
    run_shap_analysis
)

from src.evaluation.model_registry import (
    run_model_registry
)

from src.prediction.test_prediction import (
    run_prediction_test
)

from src.utils.paths import (
    TRAINING_DIR,
    TUNING_DIR,
    THRESHOLD_DIR,
    EVALUATION_DIR,
    SHAP_DIR,
    MODEL_REGISTRY_DIR,
    INFERENCE_DIR
)

st.set_page_config(
    page_title="Fraud Detection MLOps",
    page_icon="🏦",
    layout="wide"
)

st.markdown(
    """
    <style>

    .main {
        background-color: #0f172a;
    }

    .metric-card {
        background-color: #1e293b;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #334155;
    }

    .section-header {
        font-size: 28px;
        font-weight: 700;
    }

    </style>
    """,
    unsafe_allow_html=True
)

st.sidebar.title(
    "🏦 Fraud Detection"
)

st.sidebar.markdown(
    "---"
)

page = st.sidebar.radio(

    "Navigation",

    [

        "Pipeline",

        "Model Performance",

        "SHAP Analysis",

        "Fraud Prediction",

        "Model Registry"

    ]
)

def load_json(path):

    path = Path(path)

    if path.exists():

        with open(path) as f:

            return json.load(f)

    return None


def show_metrics(metrics):

    c1, c2, c3 = st.columns(3)

    c4, c5, c6 = st.columns(3)

    c1.metric(
        "Accuracy",
        f"{metrics.get('accuracy',0):.4f}"
    )

    c2.metric(
        "Precision",
        f"{metrics.get('precision',0):.4f}"
    )

    c3.metric(
        "Recall",
        f"{metrics.get('recall',0):.4f}"
    )

    c4.metric(
        "F1 Score",
        f"{metrics.get('f1',0):.4f}"
    )

    c5.metric(
        "ROC AUC",
        f"{metrics.get('roc_auc',0):.4f}"
    )

    c6.metric(
        "PR AUC",
        f"{metrics.get('pr_auc',0):.4f}"
    )

if page == "Pipeline":

    st.title(
        "🚀 Training Pipeline"
    )

    use_random_search = st.checkbox(
        "Run Hyperparameter Tuning",
        value=True
    )

    use_threshold = st.checkbox(
        "Run Threshold Tuning",
        value=True
    )

    if st.button(
        "▶ Run Pipeline",
        use_container_width=True
    ):

        stages = []

        stages.append(
            ("Ingestion", run_ingestion)
        )

        stages.append(
            ("Validation", run_validation)
        )

        stages.append(
            (
                "Feature Engineering",
                run_feature_engineering
            )
        )

        stages.append(
            (
                "Preprocessing",
                run_preprocessing
            )
        )

        stages.append(
            (
                "Training",
                run_training
            )
        )

        if use_random_search:

            stages.append(
                (
                    "Random Search",
                    run_random_search
                )
            )

        if use_threshold:

            stages.append(
                (
                    "Threshold Tuning",
                    run_threshold_tuning
                )
            )

        stages.extend([

            (
                "Evaluation",
                run_evaluation
            ),

            (
                "SHAP Analysis",
                run_shap_analysis
            ),

            (
                "Model Registry",
                run_model_registry
            ),

            (
                "Prediction Validation",
                run_prediction_test
            )

        ])

        progress_bar = st.progress(0)

        status_box = st.empty()

        log_box = st.empty()

        logs = []

        total = len(stages)

        for idx, (name, func) in enumerate(stages):

            status_box.info(
                f"Running {name}"
            )

            logs.append(
                f"▶ {name}"
            )

            log_box.code(
                "\n".join(logs)
            )

            try:

                func()

                logs.append(
                    f"✔ Completed {name}"
                )

            except Exception as e:

                logs.append(
                    f"❌ Failed {name}: {e}"
                )

                log_box.code(
                    "\n".join(logs)
                )

                st.error(str(e))

                st.stop()

            progress_bar.progress(
                int(
                    ((idx + 1) / total)
                    * 100
                )
            )

            log_box.code(
                "\n".join(logs)
            )

        status_box.success(
            "Pipeline Completed Successfully"
        )

# =====================================================
# MODEL PERFORMANCE
# =====================================================

elif page == "Model Performance":

    st.title(
        "📈 Model Performance Dashboard"
    )

    metrics = load_json(

        EVALUATION_DIR /
        "metrics.json"

    )

    if metrics is None:

        st.warning(
            "Run the pipeline first."
        )

    else:

        show_metrics(
            metrics
        )

        st.divider()

        col1, col2 = st.columns(2)

        with col1:

            cm_path = (

                EVALUATION_DIR /

                "confusion_matrix.png"

            )

            if cm_path.exists():

                st.image(
                    str(cm_path)
                )

        with col2:

            roc_path = (

                EVALUATION_DIR /

                "roc_curve.png"

            )

            if roc_path.exists():

                st.image(
                    str(roc_path)
                )

        col3, col4 = st.columns(2)

        with col3:

            pr_path = (

                EVALUATION_DIR /

                "pr_curve.png"

            )

            if pr_path.exists():

                st.image(
                    str(pr_path)
                )

        with col4:

            ks_path = (

                EVALUATION_DIR /

                "ks_curve.png"

            )

            if ks_path.exists():

                st.image(
                    str(ks_path)
                )

        st.divider()

        fig = go.Figure()

        fig.add_trace(

            go.Bar(

                x=list(metrics.keys()),

                y=list(metrics.values())

            )

        )

        fig.update_layout(

            title="Evaluation Metrics",

            height=500

        )

        st.plotly_chart(

            fig,

            use_container_width=True

        )

# =====================================================
# SHAP ANALYSIS
# =====================================================

elif page == "SHAP Analysis":

    st.title(
        "🧠 SHAP Explainability"
    )

    summary_plot = (

        SHAP_DIR /

        "shap_summary.png"

    )

    bar_plot = (

        SHAP_DIR /

        "shap_bar.png"

    )

    importance_csv = (

        SHAP_DIR /

        "feature_importance.csv"

    )

    col1, col2 = st.columns(2)

    with col1:

        if summary_plot.exists():

            st.image(
                str(summary_plot)
            )

    with col2:

        if bar_plot.exists():

            st.image(
                str(bar_plot)
            )

    st.divider()

    if importance_csv.exists():

        df = pd.read_csv(
            importance_csv
        )

        st.subheader(
            "Feature Importance"
        )

        st.dataframe(
            df,
            use_container_width=True
        )

        fig = px.bar(

            df.head(20),

            x="mean_abs_shap",

            y="feature",

            orientation="h",

            title="Top 20 Features"

        )

        st.plotly_chart(

            fig,

            use_container_width=True

        )

# =====================================================
# FRAUD PREDICTION
# =====================================================

elif page == "Fraud Prediction":

    st.title(
        "🔍 Fraud Prediction"
    )

    uploaded_file = st.file_uploader(

        "Upload CSV or JSON",

        type=[

            "csv",

            "json"

        ]

    )

    if uploaded_file is not None:

        temp_path = Path(
            uploaded_file.name
        )

        with open(
            temp_path,
            "wb"
        ) as f:

            f.write(
                uploaded_file.getbuffer()
            )

        with st.spinner(

            "Running Inference..."

        ):

            output = run_inference(
                temp_path
            )

        summary = output[
            "summary"
        ]

        predictions = output[
            "predictions"
        ]

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(

            "Transactions",

            summary[
                "total_transactions"
            ]

        )

        c2.metric(

            "Frauds",

            summary[
                "fraud_predictions"
            ]

        )

        c3.metric(

            "Fraud %",

            f"{summary['fraud_percentage']:.2f}"

        )

        c4.metric(

            "Avg Probability",

            f"{summary['average_fraud_probability']:.4f}"

        )

        st.divider()

        fig = px.histogram(

            predictions,

            x="fraud_probability",

            nbins=30,

            title="Fraud Probability Distribution"

        )

        st.plotly_chart(

            fig,

            use_container_width=True

        )

        st.subheader(
            "Top Risk Transactions"
        )

        high_risk = predictions.sort_values(

            by="fraud_probability",

            ascending=False

        ).head(25)

        st.dataframe(

            high_risk,

            use_container_width=True

        )

        csv = predictions.to_csv(
            index=False
        )

        st.download_button(

            "Download Predictions",

            csv,

            file_name=
            "predictions.csv",

            mime="text/csv"

        )

# =====================================================
# MODEL REGISTRY
# =====================================================

elif page == "Model Registry":

    st.title(
        "📦 Model Registry"
    )

    registry_file = (

        MODEL_REGISTRY_DIR /

        "model_card.json"

    )

    if registry_file.exists():

        registry = load_json(
            registry_file
        )

        st.json(
            registry
        )

        if isinstance(
            registry,
            dict
        ):

            params = registry.get(

                "hyperparameters",

                {}

            )

            if params:

                st.subheader(
                    "Hyperparameters"
                )

                st.dataframe(

                    pd.DataFrame(

                        params.items(),

                        columns=[

                            "Parameter",

                            "Value"

                        ]

                    ),

                    use_container_width=True

                )

    else:

        st.warning(
            "No model card found."
        )

st.sidebar.markdown(
    "---"
)

st.sidebar.success(
    "Fraud Detection MLOps"
)

st.sidebar.caption(
    "LightGBM + SHAP + Streamlit"
)
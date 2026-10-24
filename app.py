import joblib
import pandas as pd
import streamlit as st
from pathlib import Path

# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Fetal Health Classification",
    page_icon="🩺",
    layout="wide"
)

# ==================================================
# 2. FILE PATHS
# ==================================================

BASE_DIR = Path(__file__).parent

MODEL_PATH = BASE_DIR / "models" / "fetal_health_svm.pkl"
DATA_PATH = BASE_DIR / "data" / "raw" / "fetal_health.csv"

# ==================================================
# 3. FEATURE NAMES
# Must match the original dataset and trained model.
# ==================================================

FEATURE_NAMES = [
    "baseline value",
    "accelerations",
    "fetal_movement",
    "uterine_contractions",
    "light_decelerations",
    "severe_decelerations",
    "prolongued_decelerations",
    "abnormal_short_term_variability",
    "mean_value_of_short_term_variability",
    "percentage_of_time_with_abnormal_long_term_variability",
    "mean_value_of_long_term_variability",
    "histogram_width",
    "histogram_min",
    "histogram_max",
    "histogram_number_of_peaks",
    "histogram_number_of_zeroes",
    "histogram_mode",
    "histogram_mean",
    "histogram_median",
    "histogram_variance",
    "histogram_tendency"
]

# ==================================================
# 4. LOAD TRAINED MODEL
# ==================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


if not MODEL_PATH.exists():
    st.error(
        "Trained model not found. Please ensure that "
        "'fetal_health_svm.pkl' exists inside the models folder."
    )
    st.stop()

try:
    model = load_model()

except Exception as error:
    st.error(f"Unable to load the trained model: {error}")
    st.stop()

# ==================================================
# 5. LOAD DATASET FEATURE RANGES
# Used to validate inputs against observed data ranges.
# The target column is excluded.
# ==================================================

@st.cache_data
def load_feature_ranges():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Dataset not found. Please check data/raw/fetal_health.csv."
        )

    dataset = pd.read_csv(DATA_PATH)

    missing_features = [
        feature for feature in FEATURE_NAMES
        if feature not in dataset.columns
    ]

    if missing_features:
        raise ValueError(
            f"Dataset is missing these features: {missing_features}"
        )

    feature_data = dataset[FEATURE_NAMES]

    ranges = {}

    for feature in FEATURE_NAMES:
        ranges[feature] = {
            "min": float(feature_data[feature].min()),
            "max": float(feature_data[feature].max())
        }

    return ranges


try:
    FEATURE_RANGES = load_feature_ranges()

except Exception as error:
    st.error(f"Unable to load dataset feature ranges: {error}")
    st.stop()

# ==================================================
# 6. APP HEADER
# ==================================================

st.title("🩺 Fetal Health Classification")

st.markdown(
    """
    ### Machine Learning Prediction Dashboard

    This application uses a trained **Support Vector Machine (SVM)**
    model to classify fetal health using 21 cardiotocography (CTG)
    features.
    """
)

st.info(
    "Enter the feature values below. Each input is restricted to "
    "the minimum and maximum values observed in the project dataset."
)

# ==================================================
# 7. FEATURE INPUT FORM
# ==================================================

st.subheader("📋 Input Features")

st.caption(
    "The displayed ranges come from the original dataset. "
    "The initial values are set to the observed minimums as placeholders, "
    "not as measurements from an actual case."
)

with st.form("prediction_form"):

    input_values = {}

    col1, col2, col3 = st.columns(3)

    columns = [col1, col2, col3]

    for index, feature in enumerate(FEATURE_NAMES):

        current_col = columns[index % 3]

        minimum = FEATURE_RANGES[feature]["min"]
        maximum = FEATURE_RANGES[feature]["max"]

        # Use a suitable display label.
        label = feature.replace("_", " ").title()

        # Show the observed range beneath the input.
        with current_col:

            st.markdown(f"**{label}**")

            st.caption(
                f"Observed range: {minimum:g} to {maximum:g}"
            )

            # Integer-like features use whole-number inputs.
            integer_features = {
                "baseline value",
                "abnormal_short_term_variability",
                "histogram_width",
                "histogram_min",
                "histogram_max",
                "histogram_number_of_peaks",
                "histogram_number_of_zeroes",
                "histogram_mode",
                "histogram_mean",
                "histogram_median",
                "histogram_tendency"
            }

            if feature in integer_features:

                input_values[feature] = st.number_input(
                    label=label,
                    min_value=int(minimum),
                    max_value=int(maximum),
                    value=int(minimum),
                    step=1,
                    key=feature,
                    label_visibility="collapsed"
                )

            else:

                input_values[feature] = st.number_input(
                    label=label,
                    min_value=minimum,
                    max_value=maximum,
                    value=minimum,
                    step=0.001,
                    format="%.6f",
                    key=feature,
                    label_visibility="collapsed"
                )

    st.divider()

    submitted = st.form_submit_button(
        "🔍 Predict Fetal Health",
        use_container_width=True,
        type="primary"
    )

# ==================================================
# 8. PREDICTION
# ==================================================

if submitted:

    # Preserve the exact feature order used during training.
    input_df = pd.DataFrame(
        [[input_values[feature] for feature in FEATURE_NAMES]],
        columns=FEATURE_NAMES
    )

    # Check for missing or invalid input values.
    if input_df.isnull().values.any():

        st.error("Please provide valid values for all 21 features.")

    else:

        try:
            prediction = int(model.predict(input_df)[0])

            class_labels = {
                1: "Normal",
                2: "Suspect",
                3: "Pathological"
            }

            predicted_label = class_labels.get(
                prediction,
                f"Unknown class: {prediction}"
            )

            st.divider()
            st.subheader("📊 Prediction Result")

            if prediction == 1:

                st.success(f"Predicted Class: {predicted_label}")

            elif prediction == 2:

                st.warning(f"Predicted Class: {predicted_label}")

            elif prediction == 3:

                st.error(f"Predicted Class: {predicted_label}")

            else:

                st.info(f"Predicted Class: {predicted_label}")

            result_col1, result_col2 = st.columns(2)

            with result_col1:
                st.metric(
                    label="Predicted Class",
                    value=predicted_label
                )

            with result_col2:
                st.metric(
                    label="Encoded Class",
                    value=prediction
                )

            st.markdown("#### Class Reference")

            st.markdown(
                """
                - **1 — Normal**
                - **2 — Suspect**
                - **3 — Pathological**
                """
            )

            st.warning(
                "**Educational use only:** This prediction comes from "
                "a machine-learning model trained on a specific dataset. "
                "It is not a medical diagnosis, has not been established "
                "as clinically safe, and must not be used to make "
                "clinical decisions."
            )

        except Exception as error:
            st.error(f"Prediction failed: {error}")

# ==================================================
# 9. SIDEBAR
# ==================================================

with st.sidebar:

    st.title("🩺 Project Info")

    st.markdown(
        """
        **Project:** Fetal Health Classification

        **Algorithm:** Support Vector Machine (SVM)

        **Kernel:** RBF

        **Input Features:** 21

        **Output Classes:** 3
        """
    )

    st.divider()

    st.markdown("### Class Labels")

    st.markdown(
        """
        🟢 **1 — Normal**

        🟡 **2 — Suspect**

        🔴 **3 — Pathological**
        """
    )

    st.divider()

    st.caption(
        "Developed for machine-learning education "
        "and portfolio demonstration."
    )

# ==================================================
# 10. FOOTER
# ==================================================

st.divider()

st.caption(
    "Fetal Health Classification Using SVM | "
    "Educational ML Project | Not for clinical use"
)
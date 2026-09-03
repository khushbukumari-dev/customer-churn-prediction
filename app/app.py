import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# 2. CUSTOM DARK THEME
# ============================================================

st.markdown("""
<style>

    /* ========================================================
       MAIN APP BACKGROUND
       ======================================================== */

    .stApp {
        background-color: #0f1117;
        color: white;
    }


    /* ========================================================
       NORMAL TEXT
       ======================================================== */

    .stApp p,
    .stApp label,
    .stApp span {
        color: white !important;
    }


    /* ========================================================
       HEADINGS
       ======================================================== */

    h1,
    h2,
    h3 {
        color: white !important;
        font-weight: 700 !important;
    }


    /* ========================================================
       SELECTBOX MAIN BOX
       ======================================================== */

    div[data-baseweb="select"] > div {
        background-color: #1e222d !important;
        color: white !important;
        border: 1px solid white !important;
        border-radius: 8px !important;
    }


    /* Selected value */

    div[data-baseweb="select"] span {
        color: white !important;
    }


    /* ========================================================
       SELECTBOX ARROW
       ======================================================== */

    div[data-baseweb="select"] svg {
        color: white !important;
        fill: white !important;
    }

    div[data-baseweb="select"] path {
        fill: white !important;
        stroke: white !important;
    }


    /* ========================================================
       SELECTBOX DROPDOWN
       ======================================================== */

    div[role="listbox"] {
        background-color: #1e222d !important;
        color: white !important;
        border: 1px solid white !important;
        border-radius: 8px !important;
    }


    /* Dropdown options */

    div[role="option"] {
        background-color: #1e222d !important;
        color: white !important;
    }


    div[role="option"] span {
        color: white !important;
    }


    /* Hover option */

    div[role="option"]:hover {
        background-color: #2b3040 !important;
        color: white !important;
    }


    /* Selected option */

    div[role="option"][aria-selected="true"] {
        background-color: #2b3040 !important;
        color: white !important;
    }


    div[role="option"][aria-selected="true"] span {
        color: white !important;
    }


    /* ========================================================
       NUMBER INPUT
       ======================================================== */

    input {
        background-color: #1e222d !important;
        color: white !important;
        border-color: white !important;
    }


    /* ========================================================
       NUMBER INPUT + / - BUTTONS
       ======================================================== */

    div[data-testid="stNumberInput"] button {
        background-color: #f0f2f6 !important;
        color: #111111 !important;
    }


    div[data-testid="stNumberInput"] button svg {
        color: #111111 !important;
        fill: #111111 !important;
    }


    /* ========================================================
       PREDICT BUTTON
       ======================================================== */

    div[data-testid="stButton"] > button {
        background-color: #252a36 !important;
        color: white !important;

        border: 1px solid #ffffff !important;
        border-radius: 8px !important;

        font-weight: 700 !important;
        font-size: 16px !important;

        padding: 10px 20px !important;

        transition: all 0.2s ease-in-out;
    }


    /* Predict button hover */

    div[data-testid="stButton"] > button:hover {
        background-color: #343a4a !important;
        color: white !important;
        border-color: white !important;
    }


    /* Predict button text */

    div[data-testid="stButton"] > button p {
        color: white !important;
        font-weight: 700 !important;
    }


    /* ========================================================
       HORIZONTAL LINES
       ======================================================== */

    hr {
        border-color: #333846 !important;
    }


    /* ========================================================
       PREDICTION RESULT BOX
       ======================================================== */

    .prediction-box {
        padding: 16px 20px;

        border-radius: 8px;

        margin: 10px 0 20px 0;

        font-size: 16px;

        font-weight: 700;

        color: white !important;
    }


    /* High churn */

    .high-risk {
        background-color: #4a1717;

        border-left: 5px solid #ff4b4b;
    }


    /* Low churn */

    .low-risk {
        background-color: #123d24;

        border-left: 5px solid #21c55d;
    }


    /* ========================================================
       METRIC TEXT
       ======================================================== */

    div[data-testid="stMetricLabel"] {
        color: white !important;
        font-weight: 700 !important;
    }


    div[data-testid="stMetricValue"] {
        color: white !important;
        font-weight: 700 !important;
    }


    /* ========================================================
       SUCCESS / ERROR ALERT
       ======================================================== */

    div[data-testid="stAlert"] {
        color: white !important;
    }

    div[data-testid="stAlert"] * {
        color: white !important;
    }


</style>
""", unsafe_allow_html=True)


# ============================================================
# 3. LOAD MODEL AND PREPROCESSOR
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "churn_random_forest.pkl"
PREPROCESSOR_PATH = BASE_DIR / "models" / "preprocessor.pkl"


try:

    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)

except Exception as e:

    st.error("Model or preprocessor could not be loaded.")
    st.code(str(e))
    st.stop()


# ============================================================
# 4. TITLE / INTRODUCTION
# ============================================================

st.title("📊 Customer Churn Prediction")

st.write(
    """
    This application predicts whether a telecom customer is likely
    to churn based on their demographic information, services,
    contract details and billing information.
    """
)

st.divider()


# ============================================================
# 5. CUSTOMER INFORMATION
# ============================================================

st.header("👤 Customer Information")

col1, col2, col3 = st.columns(3)


with col1:

    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )


with col2:

    senior_citizen = st.selectbox(
        "Senior Citizen",
        [0, 1]
    )


with col3:

    partner = st.selectbox(
        "Partner",
        ["Yes", "No"]
    )


col1, col2, col3 = st.columns(3)


with col1:

    dependents = st.selectbox(
        "Dependents",
        ["Yes", "No"]
    )


with col2:

    tenure = st.number_input(
        "Tenure (Months)",
        min_value=0,
        max_value=72,
        value=12
    )


with col3:

    phone_service = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )


# ============================================================
# 6. SERVICE INFORMATION
# ============================================================

st.header("📡 Service Information")

col1, col2, col3 = st.columns(3)


with col1:

    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["Yes", "No", "No phone service"]
    )


with col2:

    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )


with col3:

    online_security = st.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"]
    )


col1, col2, col3 = st.columns(3)


with col1:

    online_backup = st.selectbox(
        "Online Backup",
        ["Yes", "No", "No internet service"]
    )


with col2:

    device_protection = st.selectbox(
        "Device Protection",
        ["Yes", "No", "No internet service"]
    )


with col3:

    tech_support = st.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"]
    )


col1, col2 = st.columns(2)


with col1:

    streaming_tv = st.selectbox(
        "Streaming TV",
        ["Yes", "No", "No internet service"]
    )


with col2:

    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No", "No internet service"]
    )


# ============================================================
# 7. CONTRACT / BILLING INFORMATION
# ============================================================

st.header("💳 Contract & Billing Information")

col1, col2, col3 = st.columns(3)


with col1:

    contract = st.selectbox(
        "Contract",
        [
            "Month-to-month",
            "One year",
            "Two year"
        ]
    )


with col2:

    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )


with col3:

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )


col1, col2 = st.columns(2)


with col1:

    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        max_value=200.0,
        value=70.0
    )


with col2:

    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        max_value=10000.0,
        value=1000.0
    )


# ============================================================
# 8. CREATE INPUT DATAFRAME
# ============================================================

input_data = pd.DataFrame({

    "gender": [gender],

    "SeniorCitizen": [senior_citizen],

    "Partner": [partner],

    "Dependents": [dependents],

    "tenure": [tenure],

    "PhoneService": [phone_service],

    "MultipleLines": [multiple_lines],

    "InternetService": [internet_service],

    "OnlineSecurity": [online_security],

    "OnlineBackup": [online_backup],

    "DeviceProtection": [device_protection],

    "TechSupport": [tech_support],

    "StreamingTV": [streaming_tv],

    "StreamingMovies": [streaming_movies],

    "Contract": [contract],

    "PaperlessBilling": [paperless_billing],

    "PaymentMethod": [payment_method],

    "MonthlyCharges": [monthly_charges],

    "TotalCharges": [total_charges]

})


# ============================================================
# 9. PREDICTION BUTTON
# ============================================================

st.divider()

predict_button = st.button(
    "🔮 Predict Customer Churn",
    use_container_width=True
)


# ============================================================
# 10. MAKE PREDICTION
# ============================================================

if predict_button:

    try:

        # ----------------------------------------------------
        # Apply preprocessing
        # ----------------------------------------------------

        transformed_data = preprocessor.transform(input_data)


        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        prediction = model.predict(transformed_data)[0]


        # ----------------------------------------------------
        # Prediction probabilities
        # ----------------------------------------------------

        probability = model.predict_proba(transformed_data)[0]

        churn_probability = probability[1]

        stay_probability = probability[0]


        # ====================================================
        # 11. DISPLAY RESULT
        # ====================================================

        st.divider()

        st.header("📌 Prediction Result")


        # ====================================================
        # RESULT MESSAGE
        # ====================================================

        if prediction == 1:

            st.markdown(
                """
                <div class="prediction-box high-risk">
                    ⚠️ High Churn Risk — Customer is likely to churn.
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                """
                <div class="prediction-box low-risk">
                    ✅ Low Churn Risk — Customer is likely to stay.
                </div>
                """,
                unsafe_allow_html=True
            )


        # ====================================================
        # 12. PROBABILITY DISPLAY
        # ====================================================

        col1, col2 = st.columns(2)


        with col1:

            st.metric(
                "Stay Probability",
                f"{stay_probability * 100:.2f}%"
            )


        with col2:

            st.metric(
                "Churn Probability",
                f"{churn_probability * 100:.2f}%"
            )


        # ====================================================
        # 13. CHURN RISK
        # ====================================================

        st.write("### Churn Risk")

        st.progress(float(churn_probability))


    except Exception as e:

        st.error("Prediction failed.")

        st.code(str(e))
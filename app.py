import joblib
import pandas as pd
import streamlit as st

FEATURES = [
    "number of bedrooms",
    "number of bathrooms",
    "living area",
    "condition of the house",
    "Number of schools nearby",
]

st.set_page_config(
    page_title="HomeValue | House Price Estimator",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { max-width: 1120px; padding-top: 2.2rem; padding-bottom: 3rem; }
    [data-testid="stMetric"] { background: #f4f7f5; border: 1px solid #e1e9e3; padding: 1rem 1.1rem; border-radius: 14px; }
    .result-card { background: #ffffff; border: 2px solid #176b4b; border-radius: 16px; padding: 1.35rem 1.5rem; margin: 1rem 0; box-shadow: 0 8px 24px rgba(23, 107, 75, .12); }
    .result-label { color: #176b4b; font-size: .82rem; font-weight: 800; letter-spacing: .08em; text-transform: uppercase; margin-bottom: .35rem; }
    .result-value { color: #10251c; font-size: clamp(2rem, 5vw, 3.2rem); font-weight: 800; line-height: 1.1; margin: 0; }
    .result-note { color: #41534a; font-size: .9rem; margin-top: .6rem; }
    .eyebrow { color: #31805c; font-weight: 700; letter-spacing: .11em; text-transform: uppercase; font-size: .78rem; }
    .muted { color: #63716a; }
    div.stButton > button[kind="primary"] { background: #176b4b; border-color: #176b4b; }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

with st.sidebar:
    st.markdown("## 🏡 HomeValue")
    st.caption("A simple, data-informed starting point for exploring a home’s estimated value.")
    st.divider()
    st.markdown("### About this estimate")
    st.write(
        "A machine-learning model uses five property details to produce an estimate. "
        "The model was trained on the included House Price India dataset."
    )
    st.info(
        "This is an educational estimate, not an appraisal or financial advice. "
        "The training data does not provide a reliable currency or location context, "
        "so the result is shown in the dataset’s original price units."
    )
    st.caption("Model: GridSearchCV · Inputs: 5 property features")

st.markdown('<div class="eyebrow">Property insights</div>', unsafe_allow_html=True)
st.title("Estimate a home’s value")
st.markdown(
    '<p class="muted">Enter a few property details to get a quick estimate from the trained model.</p>',
    unsafe_allow_html=True,
)

with st.form("property_details"):
    st.subheader("Property details")
    left, right = st.columns(2)
    with left:
        bedrooms = st.number_input("Bedrooms", min_value=1, max_value=33, value=3, step=1)
        bathrooms = st.number_input("Bathrooms", min_value=0.5, max_value=8.0, value=2.0, step=0.5)
        living_area = st.number_input("Living area (sq ft)", min_value=370, max_value=13540, value=2000, step=50)
    with right:
        condition = st.slider("House condition", min_value=1, max_value=5, value=3)
        nearby_schools = st.slider("Nearby schools", min_value=1, max_value=3, value=2)
        st.caption("Use the values reported for the property. These controls cover the observed training-data ranges.")
    submitted = st.form_submit_button("Calculate estimate", type="primary", use_container_width=True)

if submitted:
    values = pd.DataFrame(
        [[bedrooms, bathrooms, living_area, condition, nearby_schools]],
        columns=FEATURES,
    )
    estimate = float(model.predict(values)[0])
    st.divider()
    st.markdown(
        f"""
        <div class="result-card" role="status" aria-live="polite">
            <div class="result-label">Estimated value</div>
            <p class="result-value">{estimate:,.0f}</p>
            <div class="result-note">Model estimate · dataset price units</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("The estimate reflects patterns in the training data and may differ substantially from a real market valuation.")
else:
    st.divider()
    st.info("Complete the property details and select **Calculate estimate** to see a result.")

with st.expander("How to interpret this result"):
    st.write(
        "The model considers bedrooms, bathrooms, living area, house condition, and nearby schools. "
        "It does not account for exact address, local market changes, lot size, renovation, or many other factors. "
        "Use the result as a demonstration of a machine-learning workflow, not as a buying or selling recommendation."
    )


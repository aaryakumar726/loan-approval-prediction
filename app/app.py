import sys
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from data_preprocessing import full_pipeline_transform

BUNDLE_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'final_model_pipeline_v2.pkl')
REVIEW_BAND = 0.05

# Hard input caps: block obviously nonsensical entries. Set generously above the
# training data's max (income_annum max ~Rs.99,00,000/yr, loan_amount max ~Rs.3,95,00,000)
# so no realistic applicant is ever blocked.
MAX_INCOME_ANNUM = 100_000_000
MAX_LOAN_AMOUNT = 500_000_000

# Soft range: the actual min/max the model was trained on. Inputs inside the hard
# cap but outside this range are extrapolations and get flagged as such.
TRAINING_RANGE = {
    "Annual income": (200_000, 9_900_000),
    "Loan amount": (300_000, 39_500_000),
    "CIBIL score": (300, 900),
}


@st.cache_resource
def load_bundle():
    return joblib.load(BUNDLE_PATH)


def out_of_training_range(income_annum, loan_amount, cibil_score):
    checks = {"Annual income": income_annum, "Loan amount": loan_amount, "CIBIL score": cibil_score}
    flags = []
    for label, value in checks.items():
        lo, hi = TRAINING_RANGE[label]
        if value < lo or value > hi:
            flags.append(f"{label} ({value:,}) is outside the training data's observed range ({lo:,}-{hi:,})")
    return flags


def predict_one(applicant, bundle):
    df = pd.DataFrame([applicant])
    X = full_pipeline_transform(df, {'sentinel_stats': bundle['sentinel_stats'], 'scaler': bundle['scaler']})
    X = X[bundle['feature_order']]
    prob = bundle['model'].predict_proba(X)[:, 1][0]
    t = bundle['decision_threshold']
    if abs(prob - t) <= bundle['review_band']:
        decision = "Manual review recommended"
    elif prob >= t:
        decision = "Approved"
    else:
        decision = "Rejected"
    return prob, decision


def main():
    st.set_page_config(page_title="Loan Approval Predictor", page_icon=None)
    st.title("Loan Approval Predictor")
    st.caption(
        "Trained on the architsharma01 Loan-Approval-Prediction-Dataset (Kaggle, 4,269 rows). "
        "This dataset does not include gender, marital status, or property area, so no demographic "
        "fairness audit is possible for this version -- see project README for details. "
        "This is a decision-support score, not an automated approval."
    )

    bundle = load_bundle()

    with st.form("applicant_form"):
        col1, col2 = st.columns(2)
        with col1:
            no_of_dependents = st.selectbox("Number of dependents", [0, 1, 2, 3, 4, 5], index=0)
            education = st.selectbox("Education", ["Graduate", "Not Graduate"])
            self_employed = st.selectbox("Self employed", ["No", "Yes"])
            loan_term = st.selectbox("Loan term (years)", [2, 4, 6, 8, 10, 12, 14, 16, 18, 20], index=4)
        with col2:
            income_annum = st.number_input(
                "Applicant annual income (Rs.)", min_value=0, max_value=MAX_INCOME_ANNUM, value=5_000_000, step=100_000)
            loan_amount = st.number_input(
                "Loan amount (Rs.)", min_value=0, max_value=MAX_LOAN_AMOUNT, value=15_000_000, step=100_000)
            cibil_score = st.number_input("CIBIL score", min_value=300, max_value=900, value=650)

        st.markdown("**Assets (Rs.)**")
        col3, col4 = st.columns(2)
        with col3:
            residential_assets_value = st.number_input("Residential assets value", min_value=0, value=7_000_000, step=100_000)
            commercial_assets_value = st.number_input("Commercial assets value", min_value=0, value=5_000_000, step=100_000)
        with col4:
            luxury_assets_value = st.number_input("Luxury assets value", min_value=0, value=15_000_000, step=100_000)
            bank_asset_value = st.number_input("Bank asset value", min_value=0, value=5_000_000, step=100_000)

        submitted = st.form_submit_button("Get decision")

    if submitted:
        applicant = {
            "no_of_dependents": no_of_dependents, "education": education, "self_employed": self_employed,
            "income_annum": income_annum, "loan_amount": loan_amount, "loan_term": loan_term,
            "cibil_score": cibil_score, "residential_assets_value": residential_assets_value,
            "commercial_assets_value": commercial_assets_value, "luxury_assets_value": luxury_assets_value,
            "bank_asset_value": bank_asset_value,
        }

        range_flags = out_of_training_range(income_annum, loan_amount, cibil_score)
        if range_flags:
            st.warning(
                "**Input outside training data range -- treat this score as unreliable:**\n\n"
                + "\n".join(f"- {f}" for f in range_flags)
                + "\n\nThe model was trained on real applications within these ranges. Values far outside "
                "them are not extrapolated sensibly by a tree-based model."
            )

        prob, decision = predict_one(applicant, bundle)
        st.metric("Predicted probability of approval", f"{prob:.1%}")
        if decision == "Manual review recommended":
            st.warning(f"**{decision}** -- score falls within the review band around the {bundle['decision_threshold']:.0%} threshold.")
        elif decision == "Approved":
            st.success(decision)
        else:
            st.error(decision)


if __name__ == "__main__":
    main()

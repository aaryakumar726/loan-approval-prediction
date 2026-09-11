"""
HDFC Loan Approval Prediction -- Streamlit inference app.

Loads the single packaged pipeline bundle (models/final_model_pipeline.pkl)
containing: imputation stats, one-hot encoder, scaler, the tuned Random Forest
model, and the business-chosen decision threshold (0.84). All fit exclusively
on the training split -- this app does no fitting, only transforms.
"""
import sys
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from data_preprocessing import apply_imputation, engineer_features, encode_binary, apply_onehot, apply_scaler

BUNDLE_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'final_model_pipeline.pkl')

# Review-band width around the threshold: applicants scoring within this band
# get routed to manual review instead of an automatic decision -- this exists
# because Phase 7's SHAP analysis found a real applicant sitting essentially
# on the boundary (predicted P=0.840 vs. threshold 0.84) who was actually
# approved historically. A hard cutoff would auto-reject cases like that;
# the review band is what keeps a human in the loop for them.
REVIEW_BAND = 0.05


@st.cache_resource
def load_bundle():
    return joblib.load(BUNDLE_PATH)


def predict_one(applicant: dict, bundle: dict):
    df = pd.DataFrame([applicant])
    df = apply_imputation(df, bundle['imputation_stats'])
    df = engineer_features(df)
    df = encode_binary(df)
    df = apply_onehot(df, bundle['onehot_encoder'])
    df = apply_scaler(df, bundle['scaler'], bundle['numeric_scaled_cols'])
    X = df[bundle['feature_order']]
    prob = bundle['model'].predict_proba(X)[:, 1][0]

    t = bundle['decision_threshold']
    if abs(prob - t) <= REVIEW_BAND:
        decision = "Manual review recommended"
    elif prob >= t:
        decision = "Approved"
    else:
        decision = "Rejected"
    return prob, decision


def main():
    st.set_page_config(page_title="HDFC Loan Approval -- Decision Support", layout="centered")
    st.title("Loan Approval -- Decision Support")
    st.caption(
        "Model output is a decision-support score, not an automated final decision. "
        "Applications scoring near the threshold are routed to manual review."
    )

    bundle = load_bundle()

    with st.form("applicant_form"):
        col1, col2 = st.columns(2)
        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            married = st.selectbox("Married", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
            education = st.selectbox("Education", ["Graduate", "Not Graduate"])
            self_employed = st.selectbox("Self Employed", ["No", "Yes"])
        with col2:
            applicant_income = st.number_input("Applicant monthly income (Rs.)", min_value=0, value=4000)
            coapplicant_income = st.number_input("Coapplicant monthly income (Rs.)", min_value=0, value=0)
            loan_amount = st.number_input("Loan amount (in thousands, Rs.)", min_value=0, value=120)
            loan_term = st.selectbox("Loan term (months)", [360, 180, 120, 84, 60, 36, 12, 300, 240, 480])
            credit_history = st.selectbox("Credit history", ["Good", "Bad", "Unknown"])
        property_area = st.selectbox("Property area", ["Urban", "Semiurban", "Rural"])

        submitted = st.form_submit_button("Get decision")

    if submitted:
        applicant = {
            'Gender': gender, 'Married': married, 'Dependents': dependents,
            'Education': education, 'Self_Employed': self_employed,
            'ApplicantIncome': applicant_income, 'CoapplicantIncome': coapplicant_income,
            'LoanAmount': loan_amount, 'Loan_Amount_Term': loan_term,
            'Credit_History': {'Good': 1.0, 'Bad': 0.0, 'Unknown': np.nan}[credit_history],
            'Property_Area': property_area,
        }
        prob, decision = predict_one(applicant, bundle)

        st.metric("Predicted probability of approval", f"{prob:.1%}")
        if decision == "Manual review recommended":
            st.warning(f"**{decision}** -- score falls within the review band around the "
                       f"{bundle['decision_threshold']:.0%} threshold.")
        elif decision == "Approved":
            st.success(f"**{decision}**")
        else:
            st.error(f"**{decision}**")

        st.caption(
            "This is a decision-support score for underwriting, not a fully automated "
            "approval/rejection. See the model card / README for fairness audit results "
            "and limitations before using this in any real workflow."
        )


if __name__ == "__main__":
    main()

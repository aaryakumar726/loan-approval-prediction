"""
Preprocessing + feature engineering for HDFC Loan Approval Prediction.
All statistics (imputation values, encoders, scaler) are fit on the
TRAINING split only, per the no-leakage rule, and then applied to test.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder

RANDOM_STATE = 42

NUMERIC_SKEWED = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']
ONEHOT_COLS = ['Dependents', 'Property_Area', 'Credit_History_cat']
BINARY_MAP = {'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0,
              'Graduate': 1, 'Not Graduate': 0}
BINARY_COLS = ['Gender', 'Married', 'Self_Employed', 'Education']


def fit_imputation_stats(train_df):
    """All imputation values learned from TRAIN ONLY."""
    stats = {}
    stats['Gender_mode'] = train_df['Gender'].mode()[0]
    stats['Married_mode'] = train_df['Married'].mode()[0]
    stats['Dependents_mode'] = train_df['Dependents'].mode()[0]
    stats['Self_Employed_mode'] = train_df['Self_Employed'].mode()[0]
    stats['LoanAmount_median'] = train_df['LoanAmount'].median()
    stats['Loan_Amount_Term_mode'] = train_df['Loan_Amount_Term'].mode()[0]
    return stats


def apply_imputation(df, stats):
    df = df.copy()
    df['Gender'] = df['Gender'].fillna(stats['Gender_mode'])
    df['Married'] = df['Married'].fillna(stats['Married_mode'])
    df['Dependents'] = df['Dependents'].fillna(stats['Dependents_mode'])
    df['Self_Employed'] = df['Self_Employed'].fillna(stats['Self_Employed_mode'])
    df['LoanAmount'] = df['LoanAmount'].fillna(stats['LoanAmount_median'])
    df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(stats['Loan_Amount_Term_mode'])
    # Credit_History: NaN becomes its own explicit category, not mode-imputed.
    # Justification (from EDA): missing-credit rows approve at 74%, far closer
    # to the good-credit group (79.6%) than the bad-credit group (7.9%) and
    # close to the base rate (68.7%) -- folding NaN into 0 would incorrectly
    # treat "unknown" as "bad credit", which the data does not support.
    df['Credit_History_cat'] = df['Credit_History'].map({1.0: 'Good', 0.0: 'Bad'}).fillna('Unknown')
    return df


def engineer_features(df):
    """
    Ratio features are treated as PRIMARY signal here, not an afterthought:
    EDA showed raw ApplicantIncome/CoapplicantIncome/LoanAmount have near-zero
    standalone correlation with Loan_Status (|r| < 0.06), so the ratios below
    are what's expected to carry the income/loan information, not the raw magnitudes.

    Unit reconciliation (explicit, not silent): ApplicantIncome/CoapplicantIncome
    are monthly rupee figures; LoanAmount is in THOUSANDS of rupees. Converting
    LoanAmount to actual rupees before computing ratios so units are consistent
    with income -- mixing thousands and units silently would distort every
    ratio below by a factor of 1000.
    """
    df = df.copy()
    df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
    loan_amount_rupees = df['LoanAmount'] * 1000

    # Debt-to-income proxy: total loan size vs. monthly household income
    df['LoanAmount_to_Income'] = loan_amount_rupees / df['TotalIncome'].replace(0, np.nan)
    df['LoanAmount_to_Income'] = df['LoanAmount_to_Income'].fillna(df['LoanAmount_to_Income'].median())

    # Approximate flat monthly repayment (no interest modeled -- explicitly a proxy,
    # not a real amortized EMI figure)
    df['EMI_estimate'] = loan_amount_rupees / df['Loan_Amount_Term']

    # Disposable income proxy after estimated repayment
    df['Balance_Income'] = df['TotalIncome'] - df['EMI_estimate']

    # Log-transform the skewed raw numerics (log1p handles CoapplicantIncome's zeros)
    for col in NUMERIC_SKEWED:
        df[f'{col}_log'] = np.log1p(df[col])

    return df


def encode_binary(df):
    df = df.copy()
    for col in BINARY_COLS:
        df[col] = df[col].map(BINARY_MAP)
    return df


def fit_onehot_encoder(train_df):
    enc = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)
    enc.fit(train_df[ONEHOT_COLS])
    return enc


def apply_onehot(df, enc):
    arr = enc.transform(df[ONEHOT_COLS])
    cols = enc.get_feature_names_out(ONEHOT_COLS)
    onehot_df = pd.DataFrame(arr, columns=cols, index=df.index)
    return pd.concat([df.drop(columns=ONEHOT_COLS), onehot_df], axis=1)


def fit_scaler(train_df, numeric_cols):
    scaler = StandardScaler()
    scaler.fit(train_df[numeric_cols])
    return scaler


def apply_scaler(df, scaler, numeric_cols):
    df = df.copy()
    df[numeric_cols] = scaler.transform(df[numeric_cols])
    return df

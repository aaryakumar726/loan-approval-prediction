"""
Preprocessing + feature engineering for the architsharma01 Loan-Approval-Prediction-Dataset
(4,269 rows). Mirrors the fit-on-train-only discipline and feature-engineering philosophy
of the original 614-row-dataset pipeline, adapted to this dataset's actual schema.
"""
import numpy as np
import pandas as pd

RAW_NUMERIC = ['no_of_dependents','income_annum','loan_amount','loan_term','cibil_score',
               'residential_assets_value','commercial_assets_value','luxury_assets_value','bank_asset_value']
ASSET_COLS = ['residential_assets_value','commercial_assets_value','luxury_assets_value','bank_asset_value']


def clean_sentinel_negatives(df, stats=None, fit=False):
    """residential_assets_value has a small number of exact -100000 sentinel values
    (28 of 4269, all exactly -100000 -- not a real economically meaningful negative
    asset value). Treated as missing and median-imputed, fit on train only."""
    df = df.copy()
    if fit:
        median_val = df.loc[df.residential_assets_value >= 0, 'residential_assets_value'].median()
        stats = {'residential_assets_median': median_val}
    df.loc[df.residential_assets_value < 0, 'residential_assets_value'] = stats['residential_assets_median']
    return df, stats


def engineer_features(df):
    df = df.copy()
    df['total_assets'] = df[ASSET_COLS].sum(axis=1)
    df['loan_to_income'] = df['loan_amount'] / df['income_annum'].replace(0, np.nan)
    df['loan_to_assets'] = df['loan_amount'] / df['total_assets'].replace(0, np.nan)
    df['income_per_dependent'] = df['income_annum'] / (df['no_of_dependents'] + 1)
    # Flat-repayment EMI proxy (loan_term is in years here; no interest rate in the data,
    # same caveat as the original project's EMI_estimate -- illustrative, not a true amortized EMI)
    df['EMI_estimate'] = df['loan_amount'] / (df['loan_term'] * 12)
    df['balance_income_monthly'] = (df['income_annum'] / 12) - df['EMI_estimate']
    # fill any inf/nan from zero-division edge cases
    for c in ['loan_to_income','loan_to_assets']:
        df[c] = df[c].replace([np.inf, -np.inf], np.nan)
        df[c] = df[c].fillna(df[c].median())
    return df


LOG_COLS = ['income_annum','loan_amount','total_assets','residential_assets_value',
            'commercial_assets_value','luxury_assets_value','bank_asset_value']

def log_transform(df):
    df = df.copy()
    for c in LOG_COLS:
        df[c + '_log'] = np.log1p(df[c].clip(lower=0))
    return df


def encode_binary(df):
    df = df.copy()
    df['education_bin'] = (df['education'] == 'Graduate').astype(int)
    df['self_employed_bin'] = (df['self_employed'] == 'Yes').astype(int)
    return df


NUMERIC_TO_SCALE = ['no_of_dependents','loan_term','cibil_score',
                     'loan_to_income','loan_to_assets','income_per_dependent',
                     'EMI_estimate','balance_income_monthly',
                     'income_annum_log','loan_amount_log','total_assets_log',
                     'residential_assets_value_log','commercial_assets_value_log',
                     'luxury_assets_value_log','bank_asset_value_log']

FEATURE_ORDER = NUMERIC_TO_SCALE + ['education_bin','self_employed_bin']


def fit_scaler(df):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(df[NUMERIC_TO_SCALE])
    return scaler


def apply_scaler(df, scaler):
    df = df.copy()
    df[NUMERIC_TO_SCALE] = scaler.transform(df[NUMERIC_TO_SCALE])
    return df


def full_pipeline_fit(raw_df):
    df, sentinel_stats = clean_sentinel_negatives(raw_df, fit=True)
    df = engineer_features(df)
    df = log_transform(df)
    df = encode_binary(df)
    scaler = fit_scaler(df)
    df = apply_scaler(df, scaler)
    return df[FEATURE_ORDER], {'sentinel_stats': sentinel_stats, 'scaler': scaler}


def full_pipeline_transform(raw_df, artifacts):
    df, _ = clean_sentinel_negatives(raw_df, stats=artifacts['sentinel_stats'], fit=False)
    df = engineer_features(df)
    df = log_transform(df)
    df = encode_binary(df)
    df = apply_scaler(df, artifacts['scaler'])
    return df[FEATURE_ORDER]

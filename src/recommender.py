"""
Product recommender — maps a customer segment to a financial product recommendation.

Usage:
    from src.recommender import recommend, predict_segment_and_recommend
"""

import pandas as pd
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent

PRODUCT_MAP = {
    "Young Spender": {
        "product": "Premium Cashback Credit Card",
        "pitch": "Earn up to 5% cashback on every transaction. No annual fee for the first year.",
        "upsell": "Starter Savings Plan — build your emergency fund while you spend smart.",
    },
    "Stable Saver": {
        "product": "High-Yield Savings Account",
        "pitch": "Earn 4.8% APY on your savings, with no minimum balance requirement.",
        "upsell": "Fixed Deposit — lock in a higher rate for 12-24 months.",
    },
    "High Earner": {
        "product": "Wealth Management Portfolio",
        "pitch": "Personalized investment strategy with a dedicated advisor. Minimum £50k.",
        "upsell": "Private Banking Suite — exclusive benefits, priority service, travel perks.",
    },
    "Debt Stressed": {
        "product": "Debt Consolidation Loan",
        "pitch": "Combine your debts into one low-rate monthly payment. Up to 60-month terms.",
        "upsell": "Free Financial Health Check with our advisors.",
    },
}

FEATURES = [
    "age", "annual_income", "monthly_spending", "savings_balance",
    "credit_score", "loan_amount", "num_existing_products",
    "months_as_customer", "spending_ratio", "savings_ratio",
]


def recommend(segment: str) -> dict:
    """Return product recommendation for a named segment."""
    return PRODUCT_MAP.get(segment, {
        "product": "General Banking Package",
        "pitch": "Contact us to discuss your options.",
        "upsell": "",
    })


def load_models():
    """Load all saved model artifacts from data/."""
    preprocessing = joblib.load(BASE_DIR / "data" / "preprocessing_pipeline.pkl")
    pca           = joblib.load(BASE_DIR / "data" / "pca_model.pkl")
    kmeans        = joblib.load(BASE_DIR / "data" / "kmeans_model.pkl")
    return preprocessing, pca, kmeans


def _name_cluster(profile_row: pd.Series) -> str:
    income  = profile_row["annual_income"]
    savings = profile_row["savings_balance"]
    credit  = profile_row["credit_score"]
    loan    = profile_row["loan_amount"]
    age     = profile_row["age"]

    if income > 90_000 and savings > 60_000:
        return "High Earner"
    elif income < 35_000 and loan > 20_000 and credit < 620:
        return "Debt Stressed"
    elif savings > 20_000 and credit > 700 and age > 38:
        return "Stable Saver"
    else:
        return "Young Spender"


def predict_segment_and_recommend(customer: dict) -> dict:
    """
    End-to-end prediction for a single new customer.

    Parameters
    ----------
    customer : dict
        Keys: age, annual_income, monthly_spending, savings_balance,
              credit_score, loan_amount, num_existing_products, months_as_customer

    Returns
    -------
    dict with keys: cluster, segment, recommended_product, pitch, upsell
    """
    preprocessing, pca, kmeans = load_models()

    row = pd.DataFrame([customer])
    row["spending_ratio"] = row["monthly_spending"] / (row["annual_income"] / 12)
    row["savings_ratio"]  = row["savings_balance"] / row["annual_income"]

    X_scaled = preprocessing.transform(row[FEATURES])
    X_pca    = pca.transform(X_scaled)
    cluster_id = int(kmeans.predict(X_pca)[0])

    # Build cluster name from training data profiles
    df_clustered = pd.read_csv(BASE_DIR / "data" / "customers_clustered.csv")
    profile = df_clustered.groupby("cluster")[
        ["age", "annual_income", "savings_balance", "credit_score", "loan_amount"]
    ].mean().loc[cluster_id]
    segment = _name_cluster(profile)

    rec = recommend(segment)
    return {
        "cluster": cluster_id,
        "segment": segment,
        "recommended_product": rec["product"],
        "pitch": rec["pitch"],
        "upsell": rec["upsell"],
    }

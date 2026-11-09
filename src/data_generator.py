"""
Generates synthetic banking customer data for the BankSense project.

This creates realistic (but fake) customer profiles with financial attributes
that form natural clusters — perfect for learning unsupervised ML.
"""

import numpy as np
import pandas as pd
import os


def generate_customers(n_customers: int = 1000, seed: int = 42) -> pd.DataFrame:
    """
    Generate a DataFrame of synthetic bank customers.

    Each customer belongs loosely to one of 4 archetypes:
      - Young Spender:    low income, high spending, low savings, low credit score
      - Stable Saver:     mid income, low spending, high savings, good credit score
      - High Earner:      high income, mid spending, high savings, excellent credit
      - Debt Stressed:    low income, high loan, low savings, poor credit score

    These archetypes are blended with noise so clusters aren't perfectly clean —
    just like real data.

    Parameters
    ----------
    n_customers : int
        Number of customers to generate.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        One row per customer, columns described below.
    """
    rng = np.random.default_rng(seed)

    # Assign each customer to an archetype
    archetypes = rng.choice(
        ["young_spender", "stable_saver", "high_earner", "debt_stressed"],
        size=n_customers,
        p=[0.30, 0.30, 0.20, 0.20],
    )

    rows = []
    for archetype in archetypes:
        if archetype == "young_spender":
            age               = int(np.clip(rng.normal(26, 4), 18, 35))
            annual_income     = round(float(np.clip(rng.normal(32_000, 6_000), 18_000, 55_000)), 2)
            monthly_spending  = round(float(np.clip(rng.normal(2_200, 400), 800, 4_000)), 2)
            savings_balance   = round(float(np.clip(rng.normal(2_500, 1_200), 0, 10_000)), 2)
            credit_score      = int(np.clip(rng.normal(620, 40), 500, 720))
            loan_amount       = round(float(np.clip(rng.normal(8_000, 3_000), 0, 20_000)), 2)
            num_products      = int(rng.integers(1, 3))
            months_as_customer= int(np.clip(rng.normal(18, 10), 1, 60))

        elif archetype == "stable_saver":
            age               = int(np.clip(rng.normal(42, 8), 30, 60))
            annual_income     = round(float(np.clip(rng.normal(58_000, 10_000), 35_000, 90_000)), 2)
            monthly_spending  = round(float(np.clip(rng.normal(2_600, 500), 1_000, 4_500)), 2)
            savings_balance   = round(float(np.clip(rng.normal(35_000, 12_000), 5_000, 80_000)), 2)
            credit_score      = int(np.clip(rng.normal(730, 35), 650, 820))
            loan_amount       = round(float(np.clip(rng.normal(5_000, 4_000), 0, 20_000)), 2)
            num_products      = int(rng.integers(2, 5))
            months_as_customer= int(np.clip(rng.normal(72, 24), 12, 180))

        elif archetype == "high_earner":
            age               = int(np.clip(rng.normal(45, 9), 32, 65))
            annual_income     = round(float(np.clip(rng.normal(120_000, 25_000), 80_000, 220_000)), 2)
            monthly_spending  = round(float(np.clip(rng.normal(5_500, 1_200), 2_000, 12_000)), 2)
            savings_balance   = round(float(np.clip(rng.normal(90_000, 30_000), 20_000, 250_000)), 2)
            credit_score      = int(np.clip(rng.normal(790, 25), 720, 850))
            loan_amount       = round(float(np.clip(rng.normal(15_000, 10_000), 0, 60_000)), 2)
            num_products      = int(rng.integers(3, 7))
            months_as_customer= int(np.clip(rng.normal(96, 30), 24, 240))

        else:  # debt_stressed
            age               = int(np.clip(rng.normal(38, 10), 22, 58))
            annual_income     = round(float(np.clip(rng.normal(28_000, 5_000), 15_000, 45_000)), 2)
            monthly_spending  = round(float(np.clip(rng.normal(2_800, 500), 1_500, 5_000)), 2)
            savings_balance   = round(float(np.clip(rng.normal(1_200, 800), 0, 5_000)), 2)
            credit_score      = int(np.clip(rng.normal(560, 50), 400, 650))
            loan_amount       = round(float(np.clip(rng.normal(32_000, 8_000), 10_000, 60_000)), 2)
            num_products      = int(rng.integers(1, 3))
            months_as_customer= int(np.clip(rng.normal(48, 20), 6, 120))

        # Randomly introduce missing values (~3% chance per financial field)
        if rng.random() < 0.03:
            savings_balance = float("nan")
        if rng.random() < 0.03:
            loan_amount = float("nan")

        rows.append({
            "customer_id":        f"C{len(rows)+1:05d}",
            "age":                age,
            "annual_income":      annual_income,
            "monthly_spending":   monthly_spending,
            "savings_balance":    savings_balance,
            "credit_score":       credit_score,
            "loan_amount":        loan_amount,
            "num_existing_products": num_products,
            "months_as_customer": months_as_customer,
            "archetype":          archetype,   # kept for validation only, not used in ML
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = generate_customers(1000)
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/customers_raw.csv", index=False)
    print(f"Generated {len(df)} customers -> data/customers_raw.csv")
    print(df.head())

#!/usr/bin/env python3

"""
RSRC Calculator: Evaluating AI Model Efficiency in the Post-Scaling Era

This script calculates the Recursive Self-Referential Compression (RSRC) metric
for AI models, focusing on training efficiency (RSRC_t).

License: GNU General Public License v3 (GPLv3)
Repository: [Your GitHub Repository Link Here]
Paper: [Link to Your Paper Here]

Data is loaded from data/model_leaderboard.csv.
Calculated RSRC scores are printed to the console.

Note: Many data points in model_leaderboard.csv are approximations and
extrapolations, especially for cutting-edge models. See README.md for details.
"""

import pandas as pd
import numpy as np

# Define parameters for RSRC calculation (as per paper)
ALPHA = 0.5
BETA = 50
ARCH_SCORE = 1.0
MOE_PARAM_SCALE_FACTOR = 0.8
LEGACY_PENALTY_FACTOR = 1.2
MOE_EXPONENT = 0.48
DENSE_EXPONENT = 0.45
CURRENT_YEAR = 2024  # Year for age decay calculation


def calculate_flops_efficiency_score(df):
    """Calculates the FLOPs Efficiency Score for each model."""
    df["FLOPs Efficiency Score"] = (
        (df["Training FLOPs"] / (df["Parameters (B)"] ** ALPHA)) *
        (df["Energy/pFLOP"] / BETA)
    )
    return df


def apply_legacy_penalty(df):
    """Applies a FLOPs inefficiency penalty to older dense models."""
    df.loc[(df["Architecture"] == "Dense") & (df["Year"] < 2024), "FLOPs Efficiency Score"] *= LEGACY_PENALTY_FACTOR
    return df


def apply_flops_cap(df):
    """Applies a cap to Training FLOPs to prevent absurd scaling."""
    df["Training FLOPs"] = np.minimum(df["Training FLOPs"], df["Parameters (B)"] ** 0.8 * 1e3)
    return df


def scale_moe_parameters(df):
    """Scales parameter count for MoE models to reflect active parameters."""
    df["Active Parameters"] = df["Parameters (B)"].copy()
    df.loc[df["Architecture"] == "MoE", "Active Parameters"] *= MOE_PARAM_SCALE_FACTOR

    # Correct Active Parameters for specific MoE models based on provided active parameter counts (if available)
    df.loc[df["Model"] == "GPT-4o", "Active Parameters"] = 200  # Example, adjust as needed
    df.loc[df["Model"] == "DeepSeek V3", "Active Parameters"] = 37 # Example, adjust as needed
    df.loc[df["Model"] == "Mixtral 8x7B", "Active Parameters"] = 13 # Example, adjust as needed

    return df


def apply_age_decay(df):
    """Applies an age-based decay to the RSRC score."""
    df["Age Decay"] = 1 - ((CURRENT_YEAR - df["Year"]) * 0.05)
    df["Age Decay"] = df["Age Decay"].clip(lower=0.8)  # Ensure decay doesn't go below 0.8
    return df


def calculate_rsrc(df):
    """Calculates the RSRC training efficiency score."""
    df["New RSRC"] = (
        (np.log(df["Active Parameters"]) * ARCH_SCORE) /
        (df["FLOPs Efficiency Score"] ** df["Architecture"].apply(lambda x: MOE_EXPONENT if x == "MoE" else DENSE_EXPONENT))
    ) * df["Age Decay"]
    return df


def preprocess_dataframe(df):
    """Preprocesses the dataframe to handle data types and missing values."""
    # Convert 'N/A' and '~55' to NaN for numerical columns, and handle '~78%' and '88.7%' in MMLU
    df['MMLU (5-shot)'] = df['MMLU (5-shot)'].astype(str).str.replace('~', '', regex=False).str.replace('%', '', regex=False).replace('N/A', np.nan).astype(float)
    df['MMLU (5-shot)'] = df['MMLU (5-shot)'].fillna(df['MMLU (5-shot)'].mean()) # Fill NaN MMLU with mean
    df['Training FLOPs'] = df['Training FLOPs'].replace('N/A', np.nan).astype(float) # Convert 'N/A' to NaN in Training FLOPs
    return df


def load_model_data(csv_path="data/model_leaderboard.csv"):
    """Loads model data from CSV."""
    try:
        df = pd.read_csv(csv_path)
        return df
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}. Please ensure the file exists.")
        return None


def main():
    """Main function to load data, calculate RSRC, and display results."""
    model_df = load_model_data()
    if model_df is None:
        return

    model_df = preprocess_dataframe(model_df)
    model_df = calculate_flops_efficiency_score(model_df)
    model_df = apply_legacy_penalty(model_df)
    model_df = apply_flops_cap(model_df)
    model_df = scale_moe_parameters(model_df)
    model_df = apply_age_decay(model_df)
    model_df = calculate_rsrc(model_df)

    # Sort by New RSRC (descending) for leaderboard order
    df_sorted = model_df.sort_values(by="New RSRC", ascending=False)

    # Display leaderboard in console
    print("----------------------------------------------------------------------------------")
    print("                              RSRC Training Efficiency Leaderboard                 ")
    print("----------------------------------------------------------------------------------")
    print("{:<25} {:<15} {:<15}".format("Model", "Architecture", "New RSRC"))
    print("----------------------------------------------------------------------------------")
    for index, row in df_sorted.iterrows():
        print("{:<25} {:<15} {:<15.2f}".format(row["Model"], row["Architecture"], row["New RSRC"]))
    print("----------------------------------------------------------------------------------")
    print("\nNote: RSRC scores are based on approximated data. See README.md for details.")


if __name__ == "__main__":
    main()

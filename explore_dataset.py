"""
Dataset Exploration & Sampling Script
======================================
Problem: Predict whether a pair of offering and wanted ads are a semantic match
         in Sri Lankan classified marketplaces.
Dataset: Sri Lankan Classified Ads Matching Dataset v1 (54,489 ad pairs)
"""

import pandas as pd
import numpy as np

# ============================================================
# 1. Load the full dataset
# ============================================================
print("=" * 70)
print("LOADING DATASET")
print("=" * 70)

df = pd.read_csv("sri_lankan_classified_ads_matching_dataset_v1.csv")

print(f"\nDataset Shape : {df.shape[0]} rows x {df.shape[1]} columns")
print(f"\nColumn Names  :")
for col in df.columns:
    print(f"  - {col}")

# ============================================================
# 2. Data Types & Memory Usage
# ============================================================
print("\n" + "=" * 70)
print("DATA TYPES & MEMORY")
print("=" * 70)
print(df.dtypes)
print(f"\nMemory Usage  : {df.memory_usage(deep=True).sum() / 1e6:.2f} MB")

# ============================================================
# 3. Missing Values
# ============================================================
print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)
missing = df.isnull().sum()
missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
missing_df = pd.DataFrame({"Missing Count": missing, "Missing %": missing_pct})
print(missing_df)

# ============================================================
# 4. Sample Rows
# ============================================================
print("\n" + "=" * 70)
print("FIRST 3 ROWS (Truncated Text)")
print("=" * 70)
for i, row in df.head(3).iterrows():
    print(f"\n--- Row {i} ---")
    for col in df.columns:
        val = str(row[col])
        display = val[:120] + "..." if len(val) > 120 else val
        print(f"  {col}: {display}")

# ============================================================
# 5. Category Distribution
# ============================================================
print("\n" + "=" * 70)
print("MAIN CATEGORY (category_1) DISTRIBUTION")
print("=" * 70)
cat1 = df["category_1"].value_counts()
cat1_pct = df["category_1"].value_counts(normalize=True).mul(100).round(2)
cat1_summary = pd.DataFrame({"Count": cat1, "Percentage": cat1_pct})
print(cat1_summary)

print("\n" + "=" * 70)
print("SUBCATEGORY (category_2) DISTRIBUTION")
print("=" * 70)
cat2 = df["category_2"].value_counts()
cat2_pct = df["category_2"].value_counts(normalize=True).mul(100).round(2)
cat2_summary = pd.DataFrame({"Count": cat2, "Percentage": cat2_pct})
print(cat2_summary)

# ============================================================
# 6. Text Length Statistics
# ============================================================
print("\n" + "=" * 70)
print("TEXT LENGTH STATISTICS (character count)")
print("=" * 70)
text_cols = ["offering_ad", "wanted_ad", "offering_ad_title", "offering_ad_description",
             "wanted_ad_title", "wanted_ad_description"]
for col in text_cols:
    lengths = df[col].astype(str).str.len()
    print(f"\n  {col}:")
    print(f"    Min: {lengths.min()}, Max: {lengths.max()}, "
          f"Mean: {lengths.mean():.0f}, Median: {lengths.median():.0f}")

# ============================================================
# 7. Duplicate Check
# ============================================================
print("\n" + "=" * 70)
print("DUPLICATE CHECK")
print("=" * 70)
print(f"  Duplicate rows          : {df.duplicated().sum()}")
print(f"  Duplicate offering_ad   : {df['offering_ad'].duplicated().sum()}")
print(f"  Duplicate wanted_ad     : {df['wanted_ad'].duplicated().sum()}")

# ============================================================
# 8. Create Sampled Dataset (10,000 records)
# ============================================================
print("\n" + "=" * 70)
print("CREATING SAMPLED DATASET (10,000 records)")
print("=" * 70)

# Stratified sampling by category_1 to preserve distribution
np.random.seed(42)

sample_size = 10000
# Calculate proportional samples per main category
cat_counts = df["category_1"].value_counts(normalize=True)
print("\nOriginal distribution:")
print(cat_counts.mul(100).round(2))

sampled_dfs = []
for cat, proportion in cat_counts.items():
    n = int(round(proportion * sample_size))
    cat_df = df[df["category_1"] == cat]
    sampled = cat_df.sample(n=min(n, len(cat_df)), random_state=42)
    sampled_dfs.append(sampled)

df_sampled = pd.concat(sampled_dfs, ignore_index=True)

# Shuffle the sampled dataset
df_sampled = df_sampled.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\nSampled dataset shape: {df_sampled.shape}")
print("\nSampled distribution:")
sampled_cat = df_sampled["category_1"].value_counts(normalize=True).mul(100).round(2)
print(sampled_cat)

print("\nSampled Subcategory distribution:")
sampled_cat2 = df_sampled["category_2"].value_counts()
print(sampled_cat2)

# Save sampled dataset
output_file = "sampled_dataset_10k.csv"
df_sampled.to_csv(output_file, index=False)
print(f"\nSampled dataset saved to: {output_file}")
print(f"File size: {pd.io.common.file_exists(output_file)}")

print("\n" + "=" * 70)
print("EXPLORATION COMPLETE")
print("=" * 70)

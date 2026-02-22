"""
Ad Match Prediction Pipeline
=============================
Problem: Predict whether a pair of offering and wanted ads are a semantic match
         in Sri Lankan classified marketplaces.

Algorithm: LightGBM (Light Gradient Boosting Machine)

Steps:
  1. Load sampled dataset (10,000 matched pairs)
  2. Generate negative (non-matching) pairs
  3. Feature engineering (text similarity, overlap, category features)
  4. Train LightGBM classifier
  5. Evaluate with multiple metrics
  6. SHAP explainability analysis
"""

import pandas as pd
import numpy as np
import re
import os
import pickle
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)
import lightgbm as lgb
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Create output directories
os.makedirs("plots", exist_ok=True)
os.makedirs("models", exist_ok=True)

# ============================================================
# 1. LOAD DATASET
# ============================================================
print("=" * 70)
print("STEP 1: LOADING SAMPLED DATASET")
print("=" * 70)

df = pd.read_csv("sampled_dataset_10k.csv")
# Fill any missing descriptions with empty string
df["offering_ad_description"] = df["offering_ad_description"].fillna("")
df["wanted_ad_description"] = df["wanted_ad_description"].fillna("")
print(f"Loaded {len(df)} matched ad pairs.")

# ============================================================
# 2. GENERATE NEGATIVE PAIRS
# ============================================================
print("\n" + "=" * 70)
print("STEP 2: GENERATING NEGATIVE (NON-MATCHING) PAIRS")
print("=" * 70)

np.random.seed(42)

# All existing rows are positive matches (label = 1)
df["label"] = 1

# Generate negative pairs with 3 difficulty levels for a realistic task:
#   - 30% cross-category (easy): offering from one category, wanted from another
#   - 35% same-category, different subcategory (medium)
#   - 35% same-subcategory, different ad (hard): tests true semantic matching
negative_pairs = []
num_negatives = len(df)

n_cross = int(num_negatives * 0.30)
n_same_cat = int(num_negatives * 0.35)
n_same_sub = num_negatives - n_cross - n_same_cat

def make_neg_pair(offering_row, wanted_row):
    return {
        "offering_ad": offering_row["offering_ad"],
        "wanted_ad": wanted_row["wanted_ad"],
        "category_1": offering_row["category_1"],
        "category_2": offering_row["category_2"],
        "offering_ad_title": offering_row["offering_ad_title"],
        "offering_ad_description": offering_row["offering_ad_description"],
        "wanted_ad_title": wanted_row["wanted_ad_title"],
        "wanted_ad_description": wanted_row["wanted_ad_description"],
        "wanted_category_1": wanted_row["category_1"],
        "wanted_category_2": wanted_row["category_2"],
        "label": 0
    }

# --- Easy negatives: different main category ---
for i in range(n_cross):
    idx_offer = np.random.randint(0, len(df))
    offering_row = df.iloc[idx_offer]
    diff_cat = df[df["category_1"] != offering_row["category_1"]]
    wanted_row = diff_cat.iloc[np.random.randint(0, len(diff_cat))]
    negative_pairs.append(make_neg_pair(offering_row, wanted_row))

# --- Medium negatives: same main category, different subcategory ---
for i in range(n_same_cat):
    idx_offer = np.random.randint(0, len(df))
    offering_row = df.iloc[idx_offer]
    same_cat_diff_sub = df[
        (df["category_1"] == offering_row["category_1"]) &
        (df["category_2"] != offering_row["category_2"])
    ]
    if len(same_cat_diff_sub) == 0:
        same_cat_diff_sub = df[df["category_1"] != offering_row["category_1"]]
    wanted_row = same_cat_diff_sub.iloc[np.random.randint(0, len(same_cat_diff_sub))]
    negative_pairs.append(make_neg_pair(offering_row, wanted_row))

# --- Hard negatives: same subcategory, different ad ---
for i in range(n_same_sub):
    idx_offer = np.random.randint(0, len(df))
    offering_row = df.iloc[idx_offer]
    same_sub = df[
        (df["category_2"] == offering_row["category_2"]) &
        (df.index != idx_offer)
    ]
    if len(same_sub) == 0:
        same_sub = df[df.index != idx_offer]
    wanted_row = same_sub.iloc[np.random.randint(0, len(same_sub))]
    negative_pairs.append(make_neg_pair(offering_row, wanted_row))

df_neg = pd.DataFrame(negative_pairs)

# Add wanted category columns to positive pairs (same category)
df["wanted_category_1"] = df["category_1"]
df["wanted_category_2"] = df["category_2"]

# Combine positive and negative pairs
df_all = pd.concat([df, df_neg], ignore_index=True)
df_all = df_all.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Positive pairs: {(df_all['label'] == 1).sum()}")
print(f"Negative pairs: {(df_all['label'] == 0).sum()}")
print(f"Total dataset:  {len(df_all)}")

# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================
print("\n" + "=" * 70)
print("STEP 3: FEATURE ENGINEERING")
print("=" * 70)


def clean_text(text):
    """Basic text cleaning."""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def word_set(text):
    """Get set of words from text."""
    return set(clean_text(text).split())


def jaccard_similarity(text1, text2):
    """Jaccard similarity between two texts."""
    s1, s2 = word_set(text1), word_set(text2)
    if len(s1 | s2) == 0:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)


def word_overlap_count(text1, text2):
    """Count of overlapping words."""
    return len(word_set(text1) & word_set(text2))


def word_overlap_ratio(text1, text2):
    """Ratio of overlapping words to total unique words in wanted ad."""
    s1, s2 = word_set(text1), word_set(text2)
    if len(s2) == 0:
        return 0.0
    return len(s1 & s2) / len(s2)


def length_ratio(text1, text2):
    """Ratio of text lengths."""
    l1, l2 = len(str(text1)), len(str(text2))
    if max(l1, l2) == 0:
        return 0.0
    return min(l1, l2) / max(l1, l2)


print("Computing text features...")

# --- Text Similarity Features ---
df_all["jaccard_full"] = df_all.apply(
    lambda r: jaccard_similarity(r["offering_ad"], r["wanted_ad"]), axis=1
)
df_all["jaccard_title"] = df_all.apply(
    lambda r: jaccard_similarity(r["offering_ad_title"], r["wanted_ad_title"]), axis=1
)
df_all["jaccard_desc"] = df_all.apply(
    lambda r: jaccard_similarity(r["offering_ad_description"], r["wanted_ad_description"]), axis=1
)

# --- Word Overlap Features ---
df_all["word_overlap_full"] = df_all.apply(
    lambda r: word_overlap_count(r["offering_ad"], r["wanted_ad"]), axis=1
)
df_all["word_overlap_title"] = df_all.apply(
    lambda r: word_overlap_count(r["offering_ad_title"], r["wanted_ad_title"]), axis=1
)
df_all["word_overlap_ratio_full"] = df_all.apply(
    lambda r: word_overlap_ratio(r["offering_ad"], r["wanted_ad"]), axis=1
)
df_all["word_overlap_ratio_title"] = df_all.apply(
    lambda r: word_overlap_ratio(r["offering_ad_title"], r["wanted_ad_title"]), axis=1
)

# --- Length Features ---
df_all["offering_len"] = df_all["offering_ad"].astype(str).str.len()
df_all["wanted_len"] = df_all["wanted_ad"].astype(str).str.len()
df_all["len_ratio_full"] = df_all.apply(
    lambda r: length_ratio(r["offering_ad"], r["wanted_ad"]), axis=1
)
df_all["offering_title_len"] = df_all["offering_ad_title"].astype(str).str.len()
df_all["wanted_title_len"] = df_all["wanted_ad_title"].astype(str).str.len()

# --- Word Count Features ---
df_all["offering_word_count"] = df_all["offering_ad"].astype(str).apply(lambda x: len(x.split()))
df_all["wanted_word_count"] = df_all["wanted_ad"].astype(str).apply(lambda x: len(x.split()))

# --- Category Match Features ---
df_all["category_match"] = (df_all["category_1"] == df_all["wanted_category_1"]).astype(int)
df_all["subcategory_match"] = (df_all["category_2"] == df_all["wanted_category_2"]).astype(int)

# --- TF-IDF Cosine Similarity ---
print("Computing TF-IDF cosine similarity...")
# Fit TF-IDF on all ad texts
all_texts = pd.concat([
    df_all["offering_ad"].apply(clean_text),
    df_all["wanted_ad"].apply(clean_text)
]).reset_index(drop=True)

tfidf = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 2))
tfidf.fit(all_texts)

tfidf_offering = tfidf.transform(df_all["offering_ad"].apply(clean_text))
tfidf_wanted = tfidf.transform(df_all["wanted_ad"].apply(clean_text))

# Cosine similarity (row-wise dot product of normalized TF-IDF vectors)
from sklearn.preprocessing import normalize
tfidf_offering_norm = normalize(tfidf_offering)
tfidf_wanted_norm = normalize(tfidf_wanted)
df_all["tfidf_cosine_sim"] = np.array(
    tfidf_offering_norm.multiply(tfidf_wanted_norm).sum(axis=1)
).flatten()

# Also for titles
tfidf_title = TfidfVectorizer(max_features=2000, stop_words="english")
all_titles = pd.concat([
    df_all["offering_ad_title"].apply(clean_text),
    df_all["wanted_ad_title"].apply(clean_text)
]).reset_index(drop=True)
tfidf_title.fit(all_titles)

tfidf_off_title = tfidf_title.transform(df_all["offering_ad_title"].apply(clean_text))
tfidf_wan_title = tfidf_title.transform(df_all["wanted_ad_title"].apply(clean_text))
tfidf_off_title_norm = normalize(tfidf_off_title)
tfidf_wan_title_norm = normalize(tfidf_wan_title)
df_all["tfidf_cosine_sim_title"] = np.array(
    tfidf_off_title_norm.multiply(tfidf_wan_title_norm).sum(axis=1)
).flatten()

# --- Final Feature List ---
feature_cols = [
    "jaccard_full", "jaccard_title", "jaccard_desc",
    "word_overlap_full", "word_overlap_title",
    "word_overlap_ratio_full", "word_overlap_ratio_title",
    "offering_len", "wanted_len", "len_ratio_full",
    "offering_title_len", "wanted_title_len",
    "offering_word_count", "wanted_word_count",
    "category_match", "subcategory_match",
    "tfidf_cosine_sim", "tfidf_cosine_sim_title"
]

print(f"\nTotal features: {len(feature_cols)}")
print("Features:")
for f in feature_cols:
    print(f"  - {f}")

X = df_all[feature_cols]
y = df_all["label"]

print(f"\nFeature matrix shape: {X.shape}")
print(f"\nFeature statistics:")
print(X.describe().round(4).to_string())

# ============================================================
# 4. TRAIN / VALIDATION / TEST SPLIT
# ============================================================
print("\n" + "=" * 70)
print("STEP 4: TRAIN / VALIDATION / TEST SPLIT")
print("=" * 70)

# 70% train, 15% validation, 15% test
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.1765, random_state=42, stratify=y_train_val
)
# 0.1765 of 0.85 ≈ 0.15 of total

print(f"Training set  : {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"Validation set: {X_val.shape[0]} samples ({X_val.shape[0]/len(X)*100:.1f}%)")
print(f"Test set      : {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")
print(f"\nLabel distribution in train: {y_train.value_counts().to_dict()}")
print(f"Label distribution in val  : {y_val.value_counts().to_dict()}")
print(f"Label distribution in test : {y_test.value_counts().to_dict()}")

# ============================================================
# 5. MODEL TRAINING (LightGBM)
# ============================================================
print("\n" + "=" * 70)
print("STEP 5: TRAINING LightGBM MODEL")
print("=" * 70)

# Hyperparameters
params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 7,
    "num_leaves": 63,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "random_state": 42,
    "verbose": -1,
    "n_jobs": -1
}

print("Hyperparameters:")
for k, v in params.items():
    print(f"  {k}: {v}")

model = lgb.LGBMClassifier(**params)

# Train with early stopping using validation set
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[
        lgb.early_stopping(stopping_rounds=30, verbose=True),
        lgb.log_evaluation(period=50)
    ]
)

print(f"\nBest iteration: {model.best_iteration_}")

# ============================================================
# 6. MODEL EVALUATION
# ============================================================
print("\n" + "=" * 70)
print("STEP 6: MODEL EVALUATION")
print("=" * 70)

# Predictions
y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

print("\n--- Test Set Results ---")
print(f"  Accuracy  : {accuracy:.4f}")
print(f"  Precision : {precision:.4f}")
print(f"  Recall    : {recall:.4f}")
print(f"  F1 Score  : {f1:.4f}")
print(f"  AUC-ROC   : {auc:.4f}")

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred, target_names=["No Match", "Match"]))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# --- Plot 1: Confusion Matrix ---
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Match", "Match"],
            yticklabels=["No Match", "Match"], ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix")
plt.tight_layout()
plt.savefig("plots/confusion_matrix.png", dpi=150)
plt.close()
print("\nSaved: plots/confusion_matrix.png")

# --- Plot 2: ROC Curve ---
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"LightGBM (AUC = {auc:.4f})")
ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random Baseline")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig("plots/roc_curve.png", dpi=150)
plt.close()
print("Saved: plots/roc_curve.png")

# --- Plot 3: Feature Importance (LightGBM built-in) ---
importance = model.feature_importances_
feat_imp = pd.DataFrame({
    "Feature": feature_cols,
    "Importance": importance
}).sort_values("Importance", ascending=True)

fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(feat_imp["Feature"], feat_imp["Importance"], color="steelblue")
ax.set_xlabel("Feature Importance (Split Count)")
ax.set_title("LightGBM Feature Importance")
plt.tight_layout()
plt.savefig("plots/feature_importance.png", dpi=150)
plt.close()
print("Saved: plots/feature_importance.png")

# --- Plot 4: Metrics Summary Bar Chart ---
metrics_dict = {
    "Accuracy": accuracy, "Precision": precision,
    "Recall": recall, "F1 Score": f1, "AUC-ROC": auc
}
fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(metrics_dict.keys(), metrics_dict.values(), color=["#2196F3", "#4CAF50", "#FF9800", "#F44336", "#9C27B0"])
for bar, val in zip(bars, metrics_dict.values()):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{val:.4f}", ha="center", va="bottom", fontsize=10)
ax.set_ylim(0, 1.15)
ax.set_ylabel("Score")
ax.set_title("Model Performance Metrics")
plt.tight_layout()
plt.savefig("plots/metrics_summary.png", dpi=150)
plt.close()
print("Saved: plots/metrics_summary.png")

# ============================================================
# 7. EXPLAINABILITY (SHAP)
# ============================================================
print("\n" + "=" * 70)
print("STEP 7: SHAP EXPLAINABILITY ANALYSIS")
print("=" * 70)

# Use a sample for SHAP (for speed)
shap_sample_size = 500
X_shap = X_test.sample(n=shap_sample_size, random_state=42)

print(f"Computing SHAP values for {shap_sample_size} test samples...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_shap)

# For binary classification, shap_values may be a list [class_0, class_1]
if isinstance(shap_values, list):
    shap_vals = shap_values[1]  # SHAP values for positive class (Match)
else:
    shap_vals = shap_values

# --- Plot 5: SHAP Summary (Bee Swarm) ---
fig, ax = plt.subplots(figsize=(10, 7))
shap.summary_plot(shap_vals, X_shap, feature_names=feature_cols, show=False)
plt.title("SHAP Summary Plot (Impact on Match Prediction)")
plt.tight_layout()
plt.savefig("plots/shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: plots/shap_summary.png")

# --- Plot 6: SHAP Bar Plot (Mean Absolute SHAP) ---
fig, ax = plt.subplots(figsize=(9, 6))
shap.summary_plot(shap_vals, X_shap, feature_names=feature_cols, plot_type="bar", show=False)
plt.title("SHAP Feature Importance (Mean |SHAP Value|)")
plt.tight_layout()
plt.savefig("plots/shap_bar.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: plots/shap_bar.png")

# --- Plot 7: Partial Dependence Plots for top 3 features ---
print("\nGenerating Partial Dependence Plots...")
mean_abs_shap = np.abs(shap_vals).mean(axis=0)
top3_idx = np.argsort(mean_abs_shap)[-3:][::-1]
top3_features = [feature_cols[i] for i in top3_idx]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for k, (feat, ax) in enumerate(zip(top3_features, axes)):
    shap.dependence_plot(feat, shap_vals, X_shap, feature_names=feature_cols, ax=ax, show=False)
    ax.set_title(f"PDP: {feat}")
plt.suptitle("Partial Dependence Plots (Top 3 Features)", y=1.02)
plt.tight_layout()
plt.savefig("plots/pdp_top3.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: plots/pdp_top3.png")

# --- SHAP Interpretation Summary ---
print("\n--- SHAP Feature Importance Ranking ---")
shap_importance = pd.DataFrame({
    "Feature": feature_cols,
    "Mean |SHAP|": np.abs(shap_vals).mean(axis=0)
}).sort_values("Mean |SHAP|", ascending=False)
print(shap_importance.to_string(index=False))

# ============================================================
# 8. SAVE MODEL & ARTIFACTS
# ============================================================
print("\n" + "=" * 70)
print("STEP 8: SAVING MODEL & ARTIFACTS")
print("=" * 70)

# Save model
model.booster_.save_model("models/lightgbm_model.txt")
print("Saved: models/lightgbm_model.txt")

# Save as pickle for Streamlit app
with open("models/model.pkl", "wb") as f:
    pickle.dump(model, f)
print("Saved: models/model.pkl")

# Save TF-IDF vectorizers
with open("models/tfidf_full.pkl", "wb") as f:
    pickle.dump(tfidf, f)
with open("models/tfidf_title.pkl", "wb") as f:
    pickle.dump(tfidf_title, f)
print("Saved: models/tfidf_full.pkl, models/tfidf_title.pkl")

# Save feature columns list
with open("models/feature_cols.pkl", "wb") as f:
    pickle.dump(feature_cols, f)
print("Saved: models/feature_cols.pkl")

# Save results summary
results = {
    "accuracy": accuracy, "precision": precision,
    "recall": recall, "f1": f1, "auc": auc,
    "best_iteration": model.best_iteration_,
    "params": params
}
with open("models/results.pkl", "wb") as f:
    pickle.dump(results, f)
print("Saved: models/results.pkl")

print("\n" + "=" * 70)
print("PIPELINE COMPLETE!")
print("=" * 70)
print(f"\nFinal Results:")
print(f"  Accuracy  : {accuracy:.4f}")
print(f"  Precision : {precision:.4f}")
print(f"  Recall    : {recall:.4f}")
print(f"  F1 Score  : {f1:.4f}")
print(f"  AUC-ROC   : {auc:.4f}")
print(f"\nPlots saved in: plots/")
print(f"Model saved in: models/")

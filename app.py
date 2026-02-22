"""
Streamlit App - Ad Match Prediction
=====================================
A web interface for predicting whether an offering ad matches a wanted ad
in Sri Lankan classified marketplaces.

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
from sklearn.preprocessing import normalize

# ============================================================
# Load Model & Artifacts
# ============================================================
@st.cache_resource
def load_artifacts():
    with open("models/model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/tfidf_full.pkl", "rb") as f:
        tfidf_full = pickle.load(f)
    with open("models/tfidf_title.pkl", "rb") as f:
        tfidf_title = pickle.load(f)
    with open("models/feature_cols.pkl", "rb") as f:
        feature_cols = pickle.load(f)
    with open("models/results.pkl", "rb") as f:
        results = pickle.load(f)
    return model, tfidf_full, tfidf_title, feature_cols, results

# ============================================================
# Feature Engineering Functions
# ============================================================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def word_set(text):
    return set(clean_text(text).split())

def jaccard_similarity(t1, t2):
    s1, s2 = word_set(t1), word_set(t2)
    if len(s1 | s2) == 0:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)

def word_overlap_count(t1, t2):
    return len(word_set(t1) & word_set(t2))

def word_overlap_ratio(t1, t2):
    s1, s2 = word_set(t1), word_set(t2)
    if len(s2) == 0:
        return 0.0
    return len(s1 & s2) / len(s2)

def length_ratio(t1, t2):
    l1, l2 = len(str(t1)), len(str(t2))
    if max(l1, l2) == 0:
        return 0.0
    return min(l1, l2) / max(l1, l2)

def extract_features(off_title, off_desc, wan_title, wan_desc,
                     cat_match, subcat_match, tfidf_full, tfidf_title):
    offering_ad = off_title + " " + off_desc
    wanted_ad = wan_title + " " + wan_desc

    features = {
        "jaccard_full": jaccard_similarity(offering_ad, wanted_ad),
        "jaccard_title": jaccard_similarity(off_title, wan_title),
        "jaccard_desc": jaccard_similarity(off_desc, wan_desc),
        "word_overlap_full": word_overlap_count(offering_ad, wanted_ad),
        "word_overlap_title": word_overlap_count(off_title, wan_title),
        "word_overlap_ratio_full": word_overlap_ratio(offering_ad, wanted_ad),
        "word_overlap_ratio_title": word_overlap_ratio(off_title, wan_title),
        "offering_len": len(offering_ad),
        "wanted_len": len(wanted_ad),
        "len_ratio_full": length_ratio(offering_ad, wanted_ad),
        "offering_title_len": len(off_title),
        "wanted_title_len": len(wan_title),
        "offering_word_count": len(offering_ad.split()),
        "wanted_word_count": len(wanted_ad.split()),
        "category_match": cat_match,
        "subcategory_match": subcat_match,
    }

    # TF-IDF cosine similarity
    off_vec = normalize(tfidf_full.transform([clean_text(offering_ad)]))
    wan_vec = normalize(tfidf_full.transform([clean_text(wanted_ad)]))
    features["tfidf_cosine_sim"] = float(off_vec.multiply(wan_vec).sum())

    off_t_vec = normalize(tfidf_title.transform([clean_text(off_title)]))
    wan_t_vec = normalize(tfidf_title.transform([clean_text(wan_title)]))
    features["tfidf_cosine_sim_title"] = float(off_t_vec.multiply(wan_t_vec).sum())

    return features

# ============================================================
# Streamlit UI
# ============================================================
st.set_page_config(page_title="Ad Match Predictor", page_icon="🔍", layout="wide")

st.title("🔍 Ad Match Predictor")
st.markdown("""
Predict whether an **offering ad** matches a **wanted ad** in Sri Lankan classified marketplaces.  
Powered by **LightGBM** with explainable AI (SHAP).
""")

try:
    model, tfidf_full, tfidf_title, feature_cols, results = load_artifacts()
except FileNotFoundError:
    st.error("⚠️ Model not found. Please run `python train_model.py` first.")
    st.stop()

# --- Sidebar: Model Info ---
st.sidebar.header("📊 Model Performance")
st.sidebar.metric("Accuracy", f"{results['accuracy']:.4f}")
st.sidebar.metric("F1 Score", f"{results['f1']:.4f}")
st.sidebar.metric("AUC-ROC", f"{results['auc']:.4f}")
st.sidebar.metric("Precision", f"{results['precision']:.4f}")
st.sidebar.metric("Recall", f"{results['recall']:.4f}")

st.sidebar.markdown("---")
st.sidebar.header("ℹ️ About")
st.sidebar.markdown("""
- **Algorithm**: LightGBM
- **Features**: 18 engineered features
- **Dataset**: Sri Lankan Classified Ads
- **Task**: Binary classification (Match / No Match)
""")

# --- Main Input Area ---
st.header("Enter Ad Details")

categories = ["Electronics", "Vehicle", "Property"]
subcategories = {
    "Electronics": ["Electronic Home Appliances", "Mobile Phones & Tablets", "Computer Accessories",
                    "TVs", "Audio & MP3", "Computers", "Mobile Phone Accessories",
                    "Air Conditions & Electrical fittings", "Cameras & Camcorders"],
    "Vehicle": ["car", "van", "lorry_truck", "three-wheeler", "bike", "bicycle"],
    "Property": ["house", "land", "commercial property", "apartment", "room & annex"]
}

col1, col2 = st.columns(2)

with col1:
    st.subheader("📦 Offering Ad")
    off_cat = st.selectbox("Main Category (Offering)", categories, key="off_cat")
    off_subcat = st.selectbox("Subcategory (Offering)", subcategories[off_cat], key="off_subcat")
    off_title = st.text_input("Offering Ad Title", placeholder="e.g., Samsung Galaxy S21 for sale...")
    off_desc = st.text_area("Offering Ad Description", height=150,
                            placeholder="e.g., Brand new Samsung Galaxy S21, 128GB, with warranty...")

with col2:
    st.subheader("🔎 Wanted Ad")
    wan_cat = st.selectbox("Main Category (Wanted)", categories, key="wan_cat")
    wan_subcat = st.selectbox("Subcategory (Wanted)", subcategories[wan_cat], key="wan_subcat")
    wan_title = st.text_input("Wanted Ad Title", placeholder="e.g., Looking for a Samsung phone...")
    wan_desc = st.text_area("Wanted Ad Description", height=150,
                            placeholder="e.g., Need a Samsung phone under 100,000 LKR, good condition...")

# --- Predict Button ---
if st.button("🚀 Predict Match", type="primary", use_container_width=True):
    if not off_title or not wan_title:
        st.warning("Please enter at least the titles for both ads.")
    else:
        cat_match = 1 if off_cat == wan_cat else 0
        subcat_match = 1 if off_subcat == wan_subcat else 0

        features = extract_features(
            off_title, off_desc or "", wan_title, wan_desc or "",
            cat_match, subcat_match, tfidf_full, tfidf_title
        )

        X_input = pd.DataFrame([features])[feature_cols]
        proba = model.predict_proba(X_input)[0]
        prediction = model.predict(X_input)[0]

        st.markdown("---")
        st.header("Prediction Results")

        result_col1, result_col2, result_col3 = st.columns(3)
        with result_col1:
            if prediction == 1:
                st.success("✅ **MATCH**")
            else:
                st.error("❌ **NO MATCH**")

        with result_col2:
            st.metric("Match Probability", f"{proba[1]:.2%}")

        with result_col3:
            st.metric("No Match Probability", f"{proba[0]:.2%}")

        # --- Feature Breakdown ---
        st.subheader("📋 Feature Analysis")
        feat_df = pd.DataFrame({
            "Feature": feature_cols,
            "Value": [features[f] for f in feature_cols]
        })
        feat_df["Value"] = feat_df["Value"].round(4)
        st.dataframe(feat_df, use_container_width=True, hide_index=True)

        # --- Key Insights ---
        st.subheader("💡 Key Insights")
        insights = []
        if cat_match:
            insights.append("✅ Both ads are in the **same main category**.")
        else:
            insights.append("❌ Ads are in **different main categories** — unlikely match.")
        if subcat_match:
            insights.append("✅ Both ads are in the **same subcategory**.")
        if features["tfidf_cosine_sim"] > 0.3:
            insights.append(f"✅ High text similarity (TF-IDF cosine = {features['tfidf_cosine_sim']:.3f}).")
        elif features["tfidf_cosine_sim"] < 0.1:
            insights.append(f"⚠️ Low text similarity (TF-IDF cosine = {features['tfidf_cosine_sim']:.3f}).")
        if features["jaccard_full"] > 0.2:
            insights.append(f"✅ Good word overlap (Jaccard = {features['jaccard_full']:.3f}).")

        for insight in insights:
            st.markdown(insight)

# --- Gallery: Show plots if they exist ---
st.markdown("---")
st.header("📈 Model Analysis Plots")

import os
plot_files = {
    "Confusion Matrix": "plots/confusion_matrix.png",
    "ROC Curve": "plots/roc_curve.png",
    "Feature Importance": "plots/feature_importance.png",
    "Performance Metrics": "plots/metrics_summary.png",
    "SHAP Summary": "plots/shap_summary.png",
    "SHAP Feature Importance": "plots/shap_bar.png",
    "Partial Dependence Plots": "plots/pdp_top3.png"
}

available_plots = {k: v for k, v in plot_files.items() if os.path.exists(v)}

if available_plots:
    tabs = st.tabs(list(available_plots.keys()))
    for tab, (name, path) in zip(tabs, available_plots.items()):
        with tab:
            st.image(path, caption=name, use_container_width=True)
else:
    st.info("Run `python train_model.py` first to generate analysis plots.")

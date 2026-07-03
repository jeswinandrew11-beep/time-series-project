"""
Sales Forecasting — Simple Streamlit App
Uses ONNX runtime (no TensorFlow needed — works on any Python version)

Folder structure:
  your_project/
  ├── simple_app.py
  └── artefacts/
      ├── ann_model.onnx       ← converted from .keras using tf2onnx
      ├── feat_scaler.pkl
      ├── tgt_scaler.pkl
      ├── encoders.pkl
      └── feature_cols.pkl

Run: py -m streamlit run simple_app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Sales Predictor", layout="centered")

# ── Artefact paths ────────────────────────────────────────────────────────────
ARTEFACT_DIR = os.path.join(os.path.dirname(__file__), "artefacts")

# ── Load model once ───────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    import onnxruntime as ort
    session      = ort.InferenceSession(f"{ARTEFACT_DIR}/ann_model.onnx")
    feat_scaler  = joblib.load(f"{ARTEFACT_DIR}/feat_scaler.pkl")
    tgt_scaler   = joblib.load(f"{ARTEFACT_DIR}/tgt_scaler.pkl")
    encoders     = joblib.load(f"{ARTEFACT_DIR}/encoders.pkl")
    feature_cols = joblib.load(f"{ARTEFACT_DIR}/feature_cols.pkl")
    return session, feat_scaler, tgt_scaler, encoders, feature_cols

try:
    session, feat_scaler, tgt_scaler, encoders, feature_cols = load_model()
    model_ready = True
except Exception as e:
    model_ready = False
    st.error(f"Could not load model from `artefacts/` folder: {e}")
    st.stop()

# ── Date feature extraction ───────────────────────────────────────────────────
def extract_date_features(date: pd.Timestamp) -> dict:
    return {
        "day":                        date.strftime("%A").lower(),
        "month":                      date.strftime("%b").lower(),
        "year":                       date.year,
        "days_num":                   date.dayofyear,
        "weekday_or_weekends":        1 if date.weekday() >= 5 else 0,
        "count_of_week_acc_to_year":  date.isocalendar().week,
        "count_of_week_acc_to_month": (date.day - 1) // 7 + 1,
        "holiday(canada)":    0,
        "holiday(italy)":     0,
        "holiday(finland)":   0,
        "holiday(norway)":    0,
        "holiday(kenya)":     0,
        "holiday(singapore)": 0,
    }

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("Sticker Sales Predictor")
st.caption("Enter 4 values — the model does the rest.")
st.divider()

c1, c2 = st.columns(2)
with c1:
    input_date = st.date_input("Date", value=pd.Timestamp("2017-01-01"))
    country    = st.selectbox("Country",
                              ["Canada", "Finland", "Italy", "Kenya", "Norway", "Singapore"])
with c2:
    store   = st.selectbox("Store",
                           ["Discount Stickers", "Premium Sticker Mart", "Stickers for Less"])
    product = st.selectbox("🏷 Product",
                           ["Holographic Goose", "Kaggle", "Kaggle Tiers",
                            "Kerneler", "Kerneler Dark Mode"])

with st.expander("Mark as Public Holiday? (optional)"):
    is_holiday = st.checkbox(f"Yes, it's a public holiday in {country}")

with st.expander("Auto-extracted date features"):
    ts       = pd.Timestamp(input_date)
    features = extract_date_features(ts)
    st.dataframe(pd.DataFrame([features]), use_container_width=True)

st.divider()

LOOK_BACK = 15

if st.button("Predict num_sold", type="primary"):
    try:
        ts       = pd.Timestamp(input_date)
        features = extract_date_features(ts)

        country_holiday_map = {
            "Canada": "holiday(canada)", "Italy": "holiday(italy)",
            "Finland": "holiday(finland)", "Norway": "holiday(norway)",
            "Kenya": "holiday(kenya)", "Singapore": "holiday(singapore)",
        }
        if is_holiday:
            features[country_holiday_map[country]] = 1

        features["country"] = country
        features["store"]   = store
        features["product"] = product

        df_input = pd.DataFrame([features] * (LOOK_BACK + 1))

        for col in ["country", "store", "product", "day", "month"]:
            enc         = encoders[f"ohe_{col}"]
            transformed = enc.transform(df_input[[col]])
            df_input    = pd.concat([df_input.drop(columns=[col]), transformed], axis=1)

        df_input.fillna(0, inplace=True)

        for col in feature_cols:
            if col not in df_input.columns:
                df_input[col] = 0.0

        feat_scaled = feat_scaler.transform(df_input[feature_cols]).astype("float32")
        X           = feat_scaled[:LOOK_BACK].flatten().reshape(1, -1).astype("float32")

        # ONNX inference
        input_name = session.get_inputs()[0].name
        y_scaled   = session.run(None, {input_name: X})[0].flatten()
        y_pred     = tgt_scaler.inverse_transform(y_scaled.reshape(-1, 1)).flatten()[0]

        st.success("### ✅ Prediction Ready")
        r1, r2, r3 = st.columns(3)
        r1.metric("Predicted Sales", f"{y_pred:,.1f}")
        r2.metric("Rounded",         f"{round(y_pred):,} units")
        r3.metric("Date",            str(input_date))

        st.markdown(f"""
        | Field   | Value |
        |---------|-------|
        | Country | {country} |
        | Store   | {store} |
        | Product | {product} |
        | Day     | {ts.strftime('%A')} |
        | Month   | {ts.strftime('%B %Y')} |
        | Holiday | {"Yes ✅" if is_holiday else "No"} |
        """)

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.exception(e)

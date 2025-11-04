import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

APP_TITLE = "HealthyLiver — Cirrhosis Outcome Prediction"
MODEL_PATHS = [
    Path("models/model.pkl"),
    Path("artifacts/model.pkl"),
    Path("model.pkl")
    ]

FEATURE_SCHEMA = {
    "Age": ("number", 50.0),
    "Sex": ("category", ["M", "F"]),
    "Bilirubin": ("number", 3.0),
    "Albumin": ("number", 3.5),
    "Alk_Phos": ("number", 100.0),
    "SGOT": ("number", 80.0),
    "Ascites": ("category", ["No", "Yes"]),
    "Hepatomegaly": ("category", ["No", "Yes"]),
    "Spiders": ("category", ["No", "Yes"]),
    "Edema": ("category", ["No", "Yes"]),
    "Stage": ("category", ["1", "2", "3", "4"])
    }

CATEGORY_MAPS = {
    "Sex": {"M": 1, "F": 0},
    "Ascites": {"No": 0, "Yes": 1},
    "Hepatomegaly": {"No": 0, "Yes": 1},
    "Spiders": {"No": 0, "Yes": 1},
    "Edema": {"No": 0, "Yes": 1},
    "Stage": {"1": 1, "2": 2, "3": 3, "4": 4}
    }

TARGET_LABELS = ["C", "CL", "D"]

def load_model():
    for p in MODEL_PATHS:
        if p.exists():
            try:
                return joblib.load(p)
            except Exception as e:
                st.warning(f"Failed to load model from {p}: {e}")
    return None


def sidebar_inputs():
    st.sidebar.header("Input features (t₀ — initial assessment)")
    values = {}

    MAXS = {
        "Age": 120.0,
        "Bilirubin": 50.0,
        "Albumin": 6.0,
        "Alk_Phos": 3000.0,
        "SGOT": 5000.0
        }

    for name, (ftype, default) in FEATURE_SCHEMA.items():
        if ftype == "number":
            step = 0.1 if name not in ("Age", "SGOT", "Alk_Phos") else 1.0
            min_val = 0.0
            max_val = float(MAXS.get(name, 1e6))

            values[name] = st.sidebar.number_input(name,
                                                   value=float(default),
                                                   step=float(step),
                                                   min_value=float(min_val),
                                                   max_value=float(max_val)
                                                   )
        else:
            choices = list(default)
            values[name] = st.sidebar.selectbox(name, options=choices, index=0)

    return values


def preprocess(raw_dict: dict) -> pd.DataFrame:
    x = {}
    for k, v in raw_dict.items():
        if k in CATEGORY_MAPS:
            x[k] = CATEGORY_MAPS[k][v]
        else:
            x[k] = float(v)
    return pd.DataFrame([x])


def predict_proba_with_fallback(model, X: pd.DataFrame) -> np.ndarray:
    """Use ML model if available, otherwise apply a deterministic fallback. Fallback is purely for demo
    """
    if model is not None:
        if hasattr(model, "predict_proba"):
            try:
                return np.asarray(model.predict_proba(X))[0]
            except Exception:
                pass
        # decision_function -> softmax
        if hasattr(model, "decision_function"):
            try:
                scores = np.atleast_2d(model.decision_function(X))[0]
                if scores.ndim == 1 and scores.shape[0] == 3:
                    exps = np.exp(scores - np.max(scores))
                    return exps / exps.sum()
            except Exception:
                pass
        # or simply predict label
        try:
            label = int(model.predict(X)[0])
            probs = np.full(3, 0.0)
            probs[min(max(label, 0), 2)] = 1.0
            return probs
        except Exception:
            pass

    # minimal deterministic fallback - simple linear score
    s_d = (
        0.02 * (X["Age"].iloc[0] - 50)
        + 0.6 * (X["Bilirubin"].iloc[0] - 2)
        - 0.7 * (X["Albumin"].iloc[0] - 3.5)
        + 0.15 * (X["Stage"].iloc[0] - 2)
        + 0.3 * X["Ascites"].iloc[0]
        )

    s_cl = (
        0.2 * X["Edema"].iloc[0]
        + 0.1 * (X["SGOT"].iloc[0] - 30)
    )

    s_c = 0.0
    scores = np.array([s_c, s_cl, s_d])
    exps = np.exp(scores - scores.max())
    return exps / exps.sum()


st.set_page_config(page_title="HealthyLiver", page_icon="🧬", layout="centered")
st.title(APP_TITLE)
st.caption("""Predict outcomes for cirrhosis patients (C - alive, CL - transplant, D - deceased).
Features are collected at t0 (initial assessment)""")

with st.expander("ℹ️ About this demo", expanded=False):
    st.markdown(
        """**Educational MVP** by **BioMLabs**.
        Load your trained sklearn model into `models/model.pkl`"""
        )

raw = sidebar_inputs()
X = preprocess(raw)

model = load_model()
probas = predict_proba_with_fallback(model, X)

st.subheader("Prognosis")

proba_df = pd.DataFrame(
    {"Prognosis": TARGET_LABELS, "Probability": probas}
    ).sort_values("Probability", ascending=False).reset_index(drop=True)

left, right = st.columns([1, 1])

with left:
    st.dataframe(
        proba_df.assign(Probability=proba_df["Probability"] * 100),
        column_config={
            "Prognosis": st.column_config.TextColumn("Prognosis"),
            "Probability": st.column_config.NumberColumn("Probability", format="%.1f%%"),
        },
        hide_index=True,
        use_container_width=True,
    )

with right:
    fig, ax = plt.subplots(figsize=(5, 3))
    bars = ax.bar(proba_df["Prognosis"], proba_df["Probability"] * 100, width=0.6)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Probability")
    ax.set_xlabel("")
    ax.set_title("Probability distribution")
    ax.bar_label(bars, fmt="%.1f%%")
    st.pyplot(fig, clear_figure=True, use_container_width=True)

pred_label = proba_df.iloc[0]["Prognosis"]
st.success(f"Predicted class: **{pred_label}**")

st.markdown("---")
st.caption("BioMLabs — HealthyLiver. Educational use only. Not a medical device.")

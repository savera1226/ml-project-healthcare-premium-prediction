import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import math
import streamlit as st
import os  # <-- FIX 1: Imports os

_artifact_dir = "artifacts"

# --- helpers and loaders with caching so Streamlit dev reloads are fast ---
@st.cache_resource
def _safe_joblib_load(path: str):
    try:
        return joblib.load(path)
    except Exception:
        return None

@st.cache_resource
def load_artifacts() -> Dict[str, Any]:
    """Attempt to load models and scalers. Return metadata dict (never raises)."""
    info = {"models": {}, "scalers": {}, "loaded": False}
    # FIX 2: Uses os.path.join() for cross-platform paths
    my = _safe_joblib_load(os.path.join(_artifact_dir, "model_young.joblib"))
    mr = _safe_joblib_load(os.path.join(_artifact_dir, "model_rest.joblib"))
    sy = _safe_joblib_load(os.path.join(_artifact_dir, "scaler_young.joblib"))
    sr = _safe_joblib_load(os.path.join(_artifact_dir, "scaler_rest.joblib"))

    info["models"]["young"] = my
    info["models"]["rest"] = mr
    info["scalers"]["young"] = sy
    info["scalers"]["rest"] = sr
    info["loaded"] = any([my is not None, mr is not None])
    return info

def artifacts_info(loaded: Optional[Dict[str, Any]] = None, as_text: bool = True):
    """Return friendly summary about loaded artifacts. If as_text False returns JSON-like dict."""
    if loaded is None:
        loaded = load_artifacts()
    models = loaded.get("models", {})
    scalers = loaded.get("scalers", {})
    summary = {
        "model_young_loaded": models.get("young") is not None,
        "model_rest_loaded": models.get("rest") is not None,
        "scaler_young_loaded": scalers.get("young") is not None,
        "scaler_rest_loaded": scalers.get("rest") is not None,
    }
    if as_text:
        parts = []
        for k, v in summary.items():
            parts.append(f"{k}: {'yes' if v else 'no'}")
        return " | ".join(parts)
    else:
        # also include feature names if available (converted to Python str)
        def _feat_names(m):
            try:
                if hasattr(m, "feature_names_in_"):
                    return [str(x) for x in list(m.feature_names_in_)]
                return None
            except Exception:
                return None
        return {
            "summary": summary,
            "feature_names_young": _feat_names(models.get("young")),
            "feature_names_rest": _feat_names(models.get("rest"))
        }

def fallback_model_available(loaded_info: Optional[Dict[str, Any]] = None) -> bool:
    """If no artifacts present, the app will use a local fallback estimator. This returns True always (fallback exists)."""
    return True

# --- utility functions for preprocessing & feature-handling ---
def _clean_feature_names(arr) -> Optional[list]:
    try:
        return [str(x) for x in list(arr)]
    except Exception:
        return None

def _col_matches(cols, prefix, value):
    if value is None:
        return None
    target = f"{prefix}_{value}".lower().replace(" ", "_")
    for c in cols:
        if str(c).lower().replace(" ", "_") == target:
            return c
    for c in cols:
        s = str(c).lower().replace(" ", "_")
        if s.startswith(prefix.lower() + "_") and s.endswith(value.lower().replace(" ", "_")):
            return c
    return None

def calculate_normalized_risk(medical_history: str) -> float:
    if not medical_history:
        return 0.0
    risk_scores = {"diabetes": 6, "heart disease": 8, "high blood pressure": 6, "thyroid": 5, "no disease": 0, "none": 0}
    parts = [p.strip().lower() for p in str(medical_history).split("&") if p.strip()]
    total = sum(risk_scores.get(p, 0) for p in parts)
    max_score = 14
    return (total / max_score) if max_score > 0 else 0.0

def preprocess_input(input_dict: Dict[str, Any]) -> pd.DataFrame:
    """Create the model-ready DataFrame (unscaled)."""
    # choose model based on age, try to get feature names
    loaded = load_artifacts()
    age_val = int(input_dict.get("Age", 30))
    chosen_model = loaded["models"].get("young") if age_val <= 25 else loaded["models"].get("rest")
    columns = None
    if chosen_model is not None and hasattr(chosen_model, "feature_names_in_"):
        columns = _clean_feature_names(chosen_model.feature_names_in_)
    if not columns:
        # fallback list
        columns = [
            'age', 'number_of_dependants', 'income_lakhs', 'insurance_plan',
            'genetical_risk', 'normalized_risk_score',
            'gender_Male',
            'region_Northwest', 'region_Southeast', 'region_Southwest', 'region_Northeast',
            'marital_status_Unmarried',
            'bmi_category_Obesity', 'bmi_category_Overweight', 'bmi_category_Underweight',
            'smoking_status_Occasional', 'smoking_status_Regular', 'smoking_status_Not_Smoking',
            'employment_status_Salaried', 'employment_status_Self-Employed', 'employment_status_Freelancer'
        ]
    # ensure string names
    columns = [str(c) for c in columns]
    df = pd.DataFrame([[0.0] * len(columns)], columns=columns, dtype="float64")
    # numeric
    if 'age' in df.columns:
        df['age'] = float(input_dict.get('Age', 30))
    if 'number_of_dependants' in df.columns:
        df['number_of_dependants'] = float(input_dict.get('Number of Dependants', 0))
    if 'income_lakhs' in df.columns:
        df['income_lakhs'] = float(input_dict.get('Income in Lakhs', 10))
    if 'genetical_risk' in df.columns:
        df['genetical_risk'] = float(input_dict.get('Genetical Risk', 0))
    if 'insurance_plan' in df.columns:
        enc = {'Bronze': 1.0, 'Silver': 2.0, 'Gold': 3.0}
        df['insurance_plan'] = float(enc.get(input_dict.get('Insurance Plan', 'Bronze'), 1.0))
    if 'normalized_risk_score' in df.columns:
        df['normalized_risk_score'] = float(calculate_normalized_risk(input_dict.get('Medical History', 'No Disease')))

    # categorical one-hot tolerant setters
    gender = input_dict.get('Gender')
    col = _col_matches(df.columns, 'gender', gender)
    if col:
        df[col] = 1.0

    region = input_dict.get('Region')
    col = _col_matches(df.columns, 'region', region)
    if col:
        df[col] = 1.0

    mstatus = input_dict.get('Marital Status')
    col = _col_matches(df.columns, 'marital_status', mstatus)
    if col:
        df[col] = 1.0

    bmi = input_dict.get('BMI Category')
    col = _col_matches(df.columns, 'bmi_category', bmi)
    if col:
        df[col] = 1.0

    smoking = input_dict.get('Smoking Status')
    if smoking:
        for candidate in [smoking, smoking.replace(" ", "_"), smoking.replace(" ", "_").lower()]:
            col = _col_matches(df.columns, 'smoking_status', candidate)
            if col:
                df[col] = 1.0
                break

    employment = input_dict.get('Employment Status')
    col = _col_matches(df.columns, 'employment_status', employment)
    if col:
        df[col] = 1.0

    # ensure columns are strings and in expected order
    df.columns = df.columns.astype(str)
    df = df.reindex(columns=columns, fill_value=0.0)
    return df

# --- Scaler application (robust) ---
def _apply_scaler(scaler_obj, df, cols_to_scale):
    if scaler_obj is None or not hasattr(scaler_obj, "transform"):
        return df
    cols = [c for c in cols_to_scale if c in df.columns]
    if not cols:
        return df
    try:
        df[cols] = df[cols].astype(float)
        transformed = scaler_obj.transform(df[cols])
        df[cols] = pd.DataFrame(transformed, columns=cols, index=df.index)
    except Exception:
        # don't raise; scaling is optional
        pass
    return df

def handle_scaling(age: int, df: pd.DataFrame) -> pd.DataFrame:
    loaded = load_artifacts()
    scaler_obj = loaded["scalers"].get("young") if int(age) <= 25 else loaded["scalers"].get("rest")
    if scaler_obj is None:
        return df
    # if saved as dict {'cols_to_scale': [...], 'scaler': scaler}
    if isinstance(scaler_obj, dict):
        cols = scaler_obj.get("cols_to_scale", [])
        scaler = scaler_obj.get("scaler")
        return _apply_scaler(scaler, df, cols)
    # else scaler object (apply to common numeric columns)
    candidate = ['age', 'number_of_dependants', 'income_lakhs', 'genetical_risk', 'insurance_plan', 'normalized_risk_score']
    return _apply_scaler(scaler_obj, df, [c for c in candidate if c in df.columns])

# --- Fallback estimator (simple, deterministic, safe) ---
def fallback_estimator(input_df: pd.DataFrame) -> float:
    """Simple heuristic fallback — returns modest premium so UI can continue."""
    # Use a lightweight deterministic formula:
    # base = 5000; age factor + income factor + genetic & normalized risk penalties + comorbid penalties
    row = input_df.iloc[0]
    base = 4500.0
    age = float(row.get("age", 30))
    income = float(row.get("income_lakhs", 10.0))
    gen = float(row.get("genetical_risk", 0.0))
    norm_risk = float(row.get("normalized_risk_score", 0.0))
    dependants = float(row.get("number_of_dependants", 0.0))

    age_factor = max(0.0, (age - 25)) * 100.0
    income_factor = max(0.0, (income - 3.0)) * 50.0  # higher income -> slightly higher nominal premium
    genetic_factor = gen * 600.0
    normalized_factor = norm_risk * 8000.0
    dependants_factor = dependants * 150.0

    rough = base + age_factor + income_factor + genetic_factor + normalized_factor + dependants_factor
    # round to nearest 100 and ensure at least base
    return float(max(base, round(rough / 100.0) * 100.0))

# --- Main predict function ---
def predict(input_dict: Dict[str, Any], force_fallback: bool = False) -> int:
    """Return predicted premium as int.
    If artifacts missing or force_fallback True, uses fallback_estimator.
    Raises Exception only for unexpected internal errors.
    """
    try:
        # prepare input
        df = preprocess_input(input_dict)

        # attempt to scale
        df = handle_scaling(int(input_dict.get("Age", 30)), df)

        # choose model or fallback
        loaded = load_artifacts()
        chosen_model = None
        if not force_fallback:
            # FIX 3: Fixed the 'input_Ddict' typo
            chosen_model = loaded["models"].get("young") if int(input_dict.get("Age", 30)) <= 25 else loaded[ "models"].get("rest")
        # if model not present -> fallback
        if chosen_model is None:
            # fallback estimator ensures app remains responsive
            val = fallback_estimator(df)
            return int(math.ceil(val))

        # make sure columns are strings and in the order model expects (if available)
        if hasattr(chosen_model, "feature_names_in_"):
            expected = _clean_feature_names(chosen_model.feature_names_in_)
            if expected:
                df = df.reindex(columns=expected, fill_value=0.0)

        # final safe predict
        preds = chosen_model.predict(df)
        if isinstance(preds, (list, tuple, np.ndarray)):
            pred_val = float(preds[0])
        else:
            pred_val = float(preds)
        return int(math.ceil(pred_val))

    except Exception as e:
        # catch unexpected internal errors and return a helpful message
        raise Exception(f"Prediction error: {str(e)}")
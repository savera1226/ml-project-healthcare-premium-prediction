"""
Health Insurance Premium Predictor - Unified, polished main app
Requires prediction_helper.py to export:
 - load_artifacts()
 - predict(input_dict, force_fallback=False)
 - preprocess_input(input_dict)
 - artifacts_info(load_info, as_text=True/False)
 - fallback_model_available(load_info)   (optional, handled defensively)
"""
import streamlit as st
from typing import Dict, Any
import pandas as pd
from datetime import datetime

# Import from your prediction helper - it should implement robust loading & fallback
from prediction_helper import (
    load_artifacts,
    predict,
    preprocess_input,
    artifacts_info,
    fallback_model_available
)

# --- Page config
st.set_page_config(
    page_title="AI Health Insurance Premium Predictor | Instant Quotes",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS (modern 2025 look; fixes selectbox visibility & clipping) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
    * { font-family: 'Inter', system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial; }

    .header-container {
        background: linear-gradient(135deg, rgba(102,126,234,0.95) 0%, rgba(118,75,162,0.95) 100%);
        padding: 1.8rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.6rem;
        color: #fff;
        box-shadow: 0 10px 30px rgba(0,0,0,0.35);
    }
    .header-title { font-size: 1.9rem; font-weight:800; margin:0; }
    .header-subtitle { font-size:0.98rem; opacity:0.95; margin-top:6px; }

    .section-header { background: linear-gradient(90deg,#10b981 0%,#059669 100%); color:white; padding:0.85rem 1rem; border-radius:10px; margin:1rem 0; font-weight:700; }

    /* Selectbox: larger tap area, visible white text, caret white */
    .stSelectbox [data-baseweb="select"] > div[role="button"],
    .stSelectbox [data-baseweb="select"] > div[role="button"] > div {
        min-height:44px !important;
        padding:0.45rem 0.9rem !important;
        border-radius:10px !important;
        background:#0f1720 !important;
        border: 1px solid rgba(255,255,255,0.06) !important;
        box-sizing:border-box !important;
    }
    .stSelectbox [data-baseweb="select"] > div[role="button"] span,
    .stSelectbox [data-baseweb="select"] > div[role="button"] div {
        color:#ffffff !important;
        font-weight:600 !important;
        font-size:15px !important;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .stSelectbox [data-baseweb="select"] > div[role="button"] svg { fill:#fff !important; }

    /* Dropdown option list */
    .baseweb-select-dropdown, .baseweb-menu-option, [role="listbox"] > div {
        background: #0b1220 !important;
        color: #fff !important;
    }
    .baseweb-menu-option:hover, [role="option"]:hover { background:#0b8f66 !important; color:#fff !important; }

    /* Number inputs */
    .stNumberInput input { background:#0f1720 !important; color:#fff !important; padding:0.45rem !important; min-height:40px; border-radius:10px !important; border:1px solid rgba(255,255,255,0.06) !important; }

    /* Buttons */
    .stButton > button { background: linear-gradient(135deg,#10b981 0%,#059669 100%); color:#fff; font-weight:700; padding:0.85rem 1rem; border-radius:10px; }

    /* Info / result boxes */
    .modern-info-box { background: rgba(59,130,246,0.06); border-left:4px solid #3b82f6; padding:0.9rem; border-radius:8px; color:#dbeafe; margin:1rem 0; }
    .result-card { background: linear-gradient(135deg,#3b82f6 0%,#7c3aed 100%); color:#fff; padding:1.4rem; border-radius:14px; text-align:center; box-shadow: 0 12px 32px rgba(0,0,0,0.4); margin-top:1rem; }
    .result-amount { font-size:2.4rem; font-weight:800; margin:0.6rem 0; }

    /* Recommendations */
    .rec-card { background: linear-gradient(135deg, rgba(20,20,20,0.8), rgba(40,40,40,0.8)); border-left:4px solid #10b981; padding:1rem; border-radius:12px; margin:0.6rem 0; color:#e6eef8; }
    .rec-card .rec-title { font-weight:700; margin-bottom:0.35rem; }
    .rec-card .rec-desc { color:#9ca3af; font-size:0.95rem; }

    /* Footer */
    .footer { text-align:center; color:#9ca3af; padding:1.2rem 0; margin-top:1.6rem; border-top:1px solid rgba(255,255,255,0.03); }

    /* small responsive tweaks */
    @media (max-width: 640px) {
        .stSelectbox [data-baseweb="select"] > div[role="button"] { min-height:48px !important; font-size:14px !important; }
    }
    </style>
""", unsafe_allow_html=True)

# --- Session state init
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

# --- Header
st.markdown("""
    <div class="header-container">
        <div style="display:flex;justify-content:space-between;align-items:center;">
            <div>
                <div class="header-title">🏥 Health Insurance Premium Predictor</div>
                <div class="header-subtitle">AI-powered premium estimation • Instant results • Personalized insights</div>
            </div>
            <div style="text-align:right;color:#fff;font-size:0.85rem;">v2025.1</div>
        </div>
    </div>
""", unsafe_allow_html=True)

# --- Load artifacts (safe)
with st.spinner("Loading AI models & artifacts..."):
    load_info = load_artifacts()

# determine model availability robustly
models_loaded_flag = False
try:
    # artifacts_info may return string or dict, but load_info usually contains model objects
    if isinstance(load_info, dict):
        models = load_info.get("models") or {}
        models_loaded_flag = any(v is not None for v in models.values()) if isinstance(models, dict) else False
    else:
        # fallback: inspect artifacts_info text
        txt = str(artifacts_info(load_info))
        models_loaded_flag = "yes" in txt.lower()
except Exception:
    models_loaded_flag = False

# top controls
status_col1, status_col2 = st.columns([3, 1])
with status_col1:
    if models_loaded_flag:
        st.markdown('<span style="display:inline-block;padding:0.35rem 0.8rem;border-radius:999px;background:rgba(16,185,129,0.12);color:#10b981;border:1px solid rgba(16,185,129,0.18);font-weight:700;">✓ Models Loaded</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span style="display:inline-block;padding:0.35rem 0.8rem;border-radius:999px;background:rgba(239,68,68,0.08);color:#ef4444;border:1px solid rgba(239,68,68,0.16);font-weight:700;">⚠ Fallback Mode</span>', unsafe_allow_html=True)
with status_col2:
    use_fallback = st.checkbox("Dev Mode (force fallback)", value=False)

# --- categorical options
CATEGORICAL_OPTIONS = {
    'Gender': ['Male', 'Female'],
    'Marital Status': ['Unmarried', 'Married'],
    'BMI Category': ['Normal', 'Obesity', 'Overweight', 'Underweight'],
    'Smoking Status': ['No Smoking', 'Regular', 'Occasional', 'Not Smoking'],
    'Employment Status': ['Salaried', 'Self-Employed', 'Freelancer'],
    'Region': ['Northwest', 'Southeast', 'Northeast', 'Southwest'],
    'Medical History': [
        'No Disease', 'Diabetes', 'High blood pressure',
        'Diabetes & High blood pressure', 'Thyroid', 'Heart disease',
        'High blood pressure & Heart disease', 'Diabetes & Thyroid',
        'Diabetes & Heart disease'
    ],
    'Insurance Plan': ['Bronze', 'Silver', 'Gold']
}

# --- Inputs UI
st.markdown('<div class="section-header">👤 Personal Information</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
with c1:
    age = st.number_input('🎂 Age', min_value=18, max_value=100, value=30, step=1)
with c2:
    gender = st.selectbox('👥 Gender', CATEGORICAL_OPTIONS['Gender'])
with c3:
    marital_status = st.selectbox('💑 Marital Status', CATEGORICAL_OPTIONS['Marital Status'])

st.markdown('<div class="section-header">👨‍👩‍👧‍👦 Family & Financial Details</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
with c1:
    number_of_dependants = st.number_input('👶 Number of Dependants', min_value=0, max_value=20, value=0, step=1)
with c2:
    income_lakhs = st.number_input('💰 Annual Income (Lakhs ₹)', min_value=0.0, max_value=200.0, value=10.0, step=0.5, format="%.2f")
with c3:
    employment_status = st.selectbox('💼 Employment Status', CATEGORICAL_OPTIONS['Employment Status'])

st.markdown('<div class="section-header">🏥 Health Information</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
with c1:
    bmi_category = st.selectbox('⚖️ BMI Category', CATEGORICAL_OPTIONS['BMI Category'])
with c2:
    smoking_status = st.selectbox('🚭 Smoking Status', CATEGORICAL_OPTIONS['Smoking Status'])
with c3:
    medical_history = st.selectbox('📋 Medical History', CATEGORICAL_OPTIONS['Medical History'])

st.markdown('<div class="section-header">📊 Insurance & Risk Assessment</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
with c1:
    insurance_plan = st.selectbox('📑 Insurance Plan', CATEGORICAL_OPTIONS['Insurance Plan'])
with c2:
    genetical_risk = st.number_input('🧬 Genetical Risk Score', min_value=0, max_value=5, value=0, step=1)
with c3:
    region = st.selectbox('📍 Region', CATEGORICAL_OPTIONS['Region'])

# --- Build input dict
input_dict: Dict[str, Any] = {
    'Age': int(age),
    'Number of Dependants': int(number_of_dependants),
    'Income in Lakhs': float(income_lakhs),
    'Genetical Risk': int(genetical_risk),
    'Insurance Plan': insurance_plan,
    'Employment Status': employment_status,
    'Gender': gender,
    'Marital Status': marital_status,
    'BMI Category': bmi_category,
    'Smoking Status': smoking_status,
    'Region': region,
    'Medical History': medical_history
}

# --- Advanced & Debug tools
with st.expander("🔧 Advanced Options & Debug Tools", expanded=False):
    df_debug_toggle = st.checkbox("Show Model Input (preprocessed)", value=False)
    show_feature_names = st.checkbox("Show Model Feature Names", value=False)
    show_scaler_info = st.checkbox("Show Scaler Info", value=False)

    if show_feature_names:
        try:
            info = artifacts_info(load_info, as_text=False)
            st.markdown("**Model artifact info / feature names**")
            st.json(info)
        except Exception as e:
            st.warning(f"Could not show features: {e}")

    if df_debug_toggle:
        try:
            df_pre = preprocess_input(input_dict)
            st.markdown("**Preprocessed model input (single-row)**")
            st.dataframe(df_pre.T.rename(columns={0: "Value"}), use_container_width=True)
        except Exception as e:
            st.warning(f"Could not build preprocessed input: {e}")

# info box
st.markdown('<div class="modern-info-box">ℹ️ <strong>Note:</strong> All fields are required for accurate prediction. Use debug tools if you encounter a model mismatch.</div>', unsafe_allow_html=True)

# simple validation
validation_errors = []
if income_lakhs <= 0:
    validation_errors.append("Income must be greater than zero.")
if not (18 <= age <= 100):
    validation_errors.append("Age must be between 18 and 100.")
if validation_errors:
    for err in validation_errors:
        st.error(f"❌ {err}")

# Predict button
st.markdown("")  # spacing
b1, b2, b3 = st.columns([1, 2, 1])
with b2:
    predict_btn = st.button('🔮 CALCULATE PREMIUM', use_container_width=True)

# Prediction flow
if predict_btn:
    if validation_errors:
        st.error("Please fix validation issues before predicting.")
    else:
        try:
            with st.spinner("🤖 AI is analyzing your profile..."):
                # call predict with fallback toggle
                premium = predict(input_dict, force_fallback=use_fallback)
                # store history
                st.session_state.prediction_history.append({
                    'timestamp': datetime.now(),
                    'premium': premium,
                    'inputs': input_dict.copy()
                })

            # result card
            st.markdown("---")
            st.markdown(f"""
                <div class="result-card">
                    <div style="font-weight:700;font-size:1.05rem;">💰 Your Estimated Annual Premium</div>
                    <div class="result-amount">₹ {int(premium):,}</div>
                    <div style="opacity:0.9;font-size:0.95rem;">Calculated with {"Fallback Estimator" if use_fallback or not models_loaded_flag else "AI Model"}</div>
                </div>
            """, unsafe_allow_html=True)

            # metrics
            st.markdown("### 📊 Premium Breakdown")
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Monthly Premium", f"₹ {premium/12:,.0f}")
            with m2:
                st.metric("Daily Cost", f"₹ {premium/365:,.0f}")
            with m3:
                st.metric("Risk Level", "High" if premium > 50000 else "Medium" if premium > 25000 else "Low")
            with m4:
                income_pct = (premium / (income_lakhs * 100000)) * 100 if income_lakhs > 0 else 0
                st.metric("Income Ratio", f"{income_pct:.1f}%")

            # recommendations (cards)
            st.markdown("---")
            st.markdown("### 💡 Personalized Recommendations")
            recommendations = []
            if smoking_status in ['Regular', 'Occasional']:
                recommendations.append({'icon':'🚭','title':'Quit Smoking','desc':'Quitting smoking can reduce your premium by ~20-30%','impact':'High Impact'})
            if bmi_category in ['Obesity', 'Overweight']:
                recommendations.append({'icon':'🏃','title':'Weight Management','desc':'Achieving a healthy BMI can lower long-term costs','impact':'High Impact'})
            if medical_history != 'No Disease':
                recommendations.append({'icon':'🏥','title':'Regular Health Monitoring','desc':'Consistent check-ups help manage conditions','impact':'Medium Impact'})
            if genetical_risk > 3:
                recommendations.append({'icon':'🧬','title':'Genetic Counseling','desc':'Consider preventive strategies if genetic risk is high','impact':'Medium Impact'})
            if insurance_plan == 'Bronze' and premium > 40000:
                recommendations.append({'icon':'⬆️','title':'Consider Plan Upgrade','desc':'Silver/Gold may offer better value for your risk profile','impact':'Medium Impact'})
            if income_pct > 10:
                recommendations.append({'icon':'💰','title':'Premium Burden','desc':f'Premium is {income_pct:.1f}% of income — consider cost-saving measures','impact':'High Impact'})

            if recommendations:
                for i, rec in enumerate(recommendations):
                    # arrange 2-column layout for cards
                    colA, colB = st.columns(2)
                    # decide column
                    chosen_col = colA if i % 2 == 0 else colB
                    with chosen_col:
                        st.markdown(f"""
                            <div class="rec-card">
                                <div class="rec-title" style="font-size:1.05rem;">{rec['icon']} {rec['title']}</div>
                                <div class="rec-desc" style="margin-bottom:6px;">{rec['desc']}</div>
                                <div style="display:inline-block;padding:0.22rem 0.55rem;border-radius:10px;background:rgba(16,185,129,0.12);color:#10b981;font-weight:700;font-size:0.85rem;">{rec['impact']}</div>
                            </div>
                        """, unsafe_allow_html=True)
            else:
                st.success("✅ Excellent health profile — no immediate recommendations.")

            # model info & history
            st.markdown("---")
            st.info("📊 " + ("Young Adult Model used" if age <= 25 else "Standard Model used"))

            if len(st.session_state.prediction_history) > 0:
                st.markdown("---")
                st.markdown("### 📈 Recent Predictions")
                hist = st.session_state.prediction_history[-6:][::-1]
                hist_df = pd.DataFrame([{
                    "Timestamp": h["timestamp"].strftime("%Y-%m-%d %H:%M"),
                    "Premium": f"₹ {h['premium']:,}",
                    "Age": h['inputs']["Age"],
                    "Plan": h['inputs']["Insurance Plan"]
                } for h in hist])
                st.dataframe(hist_df, use_container_width=True)

        except Exception as e:
            st.error("❌ Prediction error:")
            st.code(str(e), language="text")
            st.info("Tip: enable 'Show Model Input' in Advanced Options to debug feature mismatches.")

# footer
st.markdown("---")
st.markdown(""" 
    <div class="footer">
        <div><strong>⚠️ Disclaimer:</strong> Educational estimator only — real insurer quotes may vary.</div>
        <div style="margin-top:6px;">🔒 Your data is processed locally and not stored by this app.</div>
        <div style="margin-top:8px;">Made with ❤️ using <a href="https://streamlit.io" target="_blank">Streamlit</a> | © 2025</div>
    </div>
""", unsafe_allow_html=True)
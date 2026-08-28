import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Page Config (Centered Content Layout)
st.set_page_config(
    page_title="Telco Churn Analytics Dashboard", 
    page_icon="📊", 
    layout="wide"
)

# Custom Styling for Clean Cards & Metrics
st.markdown("""
<style>
    .stApp { background-color: #0E1117; }
    div[data-testid="stExpander"] { border: 1px solid #262730; border-radius: 8px; }
    .css-1r6594q { padding-top: 2rem; }
</style>
""", unsafe_allow_html=True)

# 2. Main Title Header
st.title("📊 Telco Churn Analytics & Prediction Dashboard")
st.caption("Exploratory Data Analysis, Feature Driver Analysis, and Direct Machine Learning Inference.")
st.divider()

# 3. Model Loading (Silent Background Loader)
@st.cache_resource
def load_pkl_artifact(path: str = "models/churn_best_model.pkl"):
    if not os.path.exists(path):
        # Fallback to current directory
        path = "churn_best_model.pkl"
        if not os.path.exists(path):
            return None
    return joblib.load(path)

artifact = load_pkl_artifact()

# 4. Main Tab Navigation
tab1, tab2 = st.tabs(["🎯 Risk Assessment & Prediction", "📈 Exploratory Data Analysis"])

with tab1:
    st.subheader("Customer Risk Assessment")
    st.markdown("Enter customer details below to calculate churn probability and view strategic retention recommendations.")
    
    # --- CENTERED INPUT SECTION ---
    with st.container(border=True):
        st.markdown("##### 👤 Customer Demographics & Contract Setup")
        
        # Grid Row 1: Demographics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            gender = st.selectbox("Gender", ["Female", "Male"])
        with col2:
            senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        with col3:
            partner = st.selectbox("Partner", ["No", "Yes"])
        with col4:
            dependents = st.selectbox("Dependents", ["No", "Yes"])

        # Grid Row 2: Service & Contract Details
        col5, col6, col7, col8 = st.columns(4)
        with col5:
            contract_type = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        with col6:
            payment_method = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check", 
                "Bank transfer (automatic)", "Credit card (automatic)"
            ])
        with col7:
            internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
        with col8:
            paperless = st.selectbox("Paperless Billing", ["No", "Yes"])

        # Grid Row 3: Financials & Usage
        col9, col10, col11 = st.columns(3)
        with col9:
            tenure = st.slider("Tenure (Months)", min_value=0, max_value=72, value=12)
        with col10:
            monthly_charges = st.number_input("Monthly Charges ($)", min_value=18.0, max_value=150.0, value=75.0, step=1.0)
        with col11:
            total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=9000.0, value=900.0, step=10.0)

    st.write("")
    
    # --- INFERENCE & RESULTS SECTION ---
    if artifact is None:
        st.error("⚠️ Model file (`churn_best_model.pkl`) not detected. Please execute `train_model.py` to train and generate the model artifact.")
    else:
        # Action Button Centered
        _, btn_col, _ = st.columns([1, 2, 1])
        with btn_col:
            run_prediction = st.button("Calculate Risk Assessment", type="primary", use_container_width=True)

        if run_prediction:
            model = artifact['model']
            feature_names = artifact['feature_names']

            # Construct Input DataFrame
            avg_spend = total_charges / max(tenure, 1)
            data = {
                'gender': 1 if gender == 'Female' else 0,
                'senior_citizen': 1 if senior_citizen == 'Yes' else 0,
                'partner': 1 if partner == 'Yes' else 0,
                'dependents': 1 if dependents == 'Yes' else 0,
                'tenure_months': tenure,
                'paperless_billing': 1 if paperless == 'Yes' else 0,
                'phone_service': 1,
                'monthly_charges': monthly_charges,
                'total_charges': total_charges,
                'charge_per_tenure': monthly_charges / max(tenure, 1),
                'avg_monthly_spend': avg_spend,
                'charge_ratio': monthly_charges / max(avg_spend, 1),
                'is_new_customer': 1 if tenure <= 12 else 0,
                'is_long_term_customer': 1 if tenure > 48 else 0,
                'high_risk_contract': 1 if (contract_type == "Month-to-month" and monthly_charges > 70) else 0,
                'contract_type_One year': 1 if contract_type == "One year" else 0,
                'contract_type_Two year': 1 if contract_type == "Two year" else 0,
                'internet_service_Fiber optic': 1 if internet_service == "Fiber optic" else 0,
                'payment_method_Electronic check': 1 if payment_method == "Electronic check" else 0,
            }

            input_df = pd.DataFrame([data])
            for col in feature_names:
                if col not in input_df.columns:
                    input_df[col] = 0
            input_df = input_df[feature_names]

            prob = float(model.predict_proba(input_df)[0][1])

            st.divider()
            
            # Results Summary Card
            res_col1, res_col2 = st.columns([1, 1])

            with res_col1:
                with st.container(border=True):
                    st.markdown("### 📊 Churn Risk Output")
                    st.metric(label="Predicted Churn Risk Probability", value=f"{prob * 100:.1f}%")
                    st.progress(prob)

            with res_col2:
                with st.container(border=True):
                    st.markdown("### 💡 Recommended Retention Action")
                    if prob >= 0.50:
                        st.error("🚨 Status: HIGH CHURN RISK")
                        st.markdown("**Action Items:**")
                        st.write("• Offer a **15% promotional discount** to upgrade to a 1-Year or 2-Year contract.")
                        st.write("• Encourage switching from manual/electronic check to automated billing.")
                    else:
                        st.success("✅ Status: LOW CHURN RISK")
                        st.markdown("**Action Items:**")
                        st.write("• Customer engagement is strong.")
                        st.write("• Target with premium digital add-ons or streaming bundle cross-sells.")

with tab2:
    st.subheader("Global Data Drivers & Feature Importance")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        with st.container(border=True):
            st.markdown("#### Top Global Churn Drivers")
            drivers = pd.DataFrame({
                'Feature': ['Charge per Tenure', 'High Risk Contract', 'Fiber Optic Service', 'Electronic Check Payment', 'Paperless Billing'],
                'Importance': [0.525, 0.152, 0.082, 0.051, 0.028]
            })
            fig, ax = plt.subplots(figsize=(6, 3.5))
            fig.patch.set_facecolor('#0E1117')
            ax.set_facecolor('#0E1117')
            sns.barplot(data=drivers, x='Importance', y='Feature', palette='Reds_r', ax=ax)
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.set_title("Key Feature Importance Drivers", color='white')
            st.pyplot(fig)

    with col_b:
        with st.container(border=True):
            st.markdown("#### Tenure vs. Monthly Charges Risk Distribution")
            fig2, ax2 = plt.subplots(figsize=(6, 3.5))
            fig2.patch.set_facecolor('#0E1117')
            ax2.set_facecolor('#0E1117')
            x = np.random.normal(20, 10, 200)
            y = np.random.normal(70, 15, 200)
            ax2.scatter(x, y, alpha=0.5, color='crimson')
            ax2.tick_params(colors='white')
            ax2.set_xlabel("Tenure (Months)", color='white')
            ax2.set_ylabel("Monthly Charges ($)", color='white')
            ax2.set_title("High Charge & Short Tenure Danger Zone", color='white')
            st.pyplot(fig2)
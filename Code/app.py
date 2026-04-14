import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import base64

def generate_dynamic_ai_summary(risk_level, probability, top_features_df, input_data):

    explanation = "🧠 **Clinical Risk Interpretation Report**\n\n"

    explanation += (
        f"Based on the provided clinical parameters, the predicted "
        f"probability of diabetes is **{probability*100:.2f}%**, "
        f"which falls under the **{risk_level} RISK** category.\n\n"
    )

    explanation += "### 🔍 Key Clinical Findings:\n"

    for _, row in top_features_df.iterrows():
        feature = row["Feature"]
        impact = row["Impact"]

        value = input_data[feature].values[0] if feature in input_data.columns else None

        # HbA1c (FIXED NAME ✅)
        if feature == "HbA1c_level":
            if value >= 6.5:
                explanation += f"- **HbA1c Level: {value}%** → Diabetic range (≥6.5%), strong risk factor.\n"
            elif value >= 5.7:
                explanation += f"- **HbA1c Level: {value}%** → Prediabetic range (5.7–6.4%).\n"
            else:
                explanation += f"- **HbA1c Level: {value}%** → Normal range.\n"

        # Blood Glucose
        elif feature == "blood_glucose_level":
            if value >= 200:
                explanation += f"- **Blood Glucose: {value} mg/dL** → Very high, strong diabetes indicator.\n"
            elif value >= 140:
                explanation += f"- **Blood Glucose: {value} mg/dL** → Moderately high.\n"
            else:
                explanation += f"- **Blood Glucose: {value} mg/dL** → Normal.\n"

        # BMI
        elif feature == "bmi":
            if value >= 30:
                explanation += f"- **BMI: {value}** → Obese, high risk.\n"
            elif value >= 25:
                explanation += f"- **BMI: {value}** → Overweight.\n"
            else:
                explanation += f"- **BMI: {value}** → Normal.\n"

        # Age
        elif feature == "age":
            if value >= 45:
                explanation += f"- **Age: {value} years** → Increased age-related risk.\n"
            else:
                explanation += f"- **Age: {value} years** → Lower age risk.\n"

        # Hypertension
        elif feature == "hypertension":
            if value == 1:
                explanation += "- **Hypertension Present** → Increases diabetes risk.\n"

        # Heart Disease
        elif feature == "heart_disease":
            if value == 1:
                explanation += "- **Heart Disease Present** → Indicates metabolic issues.\n"

        # One-hot encoded features
        else:
            if "gender_Male" in feature and value == 1:
                explanation += "- **Male Gender** → Slightly higher diabetes risk.\n"

            elif "smoking_history_current" in feature and value == 1:
                explanation += "- **Current Smoker** → Increases metabolic risk.\n"

            elif "smoking_history_former" in feature and value == 1:
                explanation += "- **Former Smoker** → Moderate risk contribution.\n"

            else:
                if impact > 0:
                    explanation += f"- **{feature}** increases risk.\n"
                else:
                    explanation += f"- **{feature}** has minimal impact.\n"

    explanation += "\n### 📌 Clinical Recommendation:\n"

    if risk_level == "HIGH":
        explanation += "Immediate medical consultation is strongly recommended."

    elif risk_level == "MEDIUM":
        explanation += "Lifestyle changes and monitoring are advised."

    else:
        explanation += "Maintain a healthy lifestyle."

    return explanation
def generate_personalized_recommendations(input_data, risk_level):

    rec = "🥗 **Comprehensive Personalized Recommendations**\n\n"

    age = input_data["age"].values[0]
    bmi = input_data["bmi"].values[0]
    hba1c = input_data["HbA1c_level"].values[0]
    glucose = input_data["blood_glucose_level"].values[0]
    hypertension = input_data["hypertension"].values[0]
    heart = input_data["heart_disease"].values[0]

    # Detect gender
    gender = "Male" if "gender_Male" in input_data.columns and input_data["gender_Male"].values[0] == 1 else "Female"

    # Detect smoking
    if "smoking_history_current" in input_data.columns and input_data["smoking_history_current"].values[0] == 1:
        smoking = "current"
    elif "smoking_history_former" in input_data.columns and input_data["smoking_history_former"].values[0] == 1:
        smoking = "former"
    else:
        smoking = "never"

    # =============================
    # 🔹 HEALTH RECOMMENDATIONS
    # =============================
    rec += "### 🏥 Feature-wise Recommendations\n\n"
    # 1. Age
    rec += f"🔹 **Age ({age} years):** "
    if age >= 45:
        rec += "Higher risk age group → Regular screening every 6 months recommended.\n"
    else:
        rec += "Lower risk age → Maintain active lifestyle and periodic check-ups.\n"

    # 2. BMI
    rec += f"\n🔹 **BMI ({bmi}):** "
    if bmi >= 30:
        rec += "Obese → Strong weight reduction plan required.\n"
    elif bmi >= 25:
        rec += "Overweight → Moderate weight loss advised.\n"
    else:
        rec += "Healthy → Maintain current lifestyle.\n"

    # 3. HbA1c
    rec += f"\n🔹 **HbA1c ({hba1c}%):** "
    if hba1c >= 6.5:
        rec += "Diabetic range → Immediate medical attention.\n"
    elif hba1c >= 5.7:
        rec += "Prediabetic → Control sugar intake and monitor.\n"
    else:
        rec += "Normal → Continue healthy habits.\n"

    # 4. Glucose
    rec += f"\n🔹 **Blood Glucose ({glucose} mg/dL):** "
    if glucose >= 200:
        rec += "Very high → Urgent medical consultation.\n"
    elif glucose >= 140:
        rec += "Elevated → Dietary control required.\n"
    else:
        rec += "Normal → Maintain balanced diet.\n"

    # 5. Hypertension
    rec += f"\n🔹 **Hypertension ({'Yes' if hypertension==1 else 'No'}):** "
    if hypertension == 1:
        rec += "Present → Reduce salt intake and monitor BP.\n"
    else:
        rec += "Not present → Maintain heart-healthy lifestyle.\n"

    # 6. Heart Disease
    rec += f"\n🔹 **Heart Disease ({'Yes' if heart==1 else 'No'}):** "
    if heart == 1:
        rec += "Present → Follow strict medical supervision.\n"
    else:
        rec += "Not present → Maintain cardiovascular fitness.\n"

    # 7. Gender
    rec += f"\n🔹 **Gender ({gender}):** "
    if gender == "Male":
        rec += "Slightly higher diabetes risk → Focus on lifestyle management.\n"
    else:
        rec += "Maintain hormonal and metabolic balance with healthy habits.\n"

    # 8. Smoking
    rec += f"\n🔹 **Smoking ({smoking}):** "
    if smoking == "current":
        rec += "High risk → Strongly advised to quit immediately.\n"
    elif smoking == "former":
        rec += "Good progress → Continue staying smoke-free.\n"
    else:
        rec += "Excellent → Avoid smoking to maintain health.\n"
    # =============================
    # 🥗 FOOD RECOMMENDATIONS
    # =============================
    rec += "\n### 🥗 Food Recommendations\n\n"

    # HIGH RISK DIET
    if risk_level == "HIGH":

        rec += "**🚨 Strict Diabetic Diet Plan:**\n"
        rec += "- Eat: Leafy vegetables, whole grains (brown rice, oats), legumes, nuts\n"
        rec += "- Protein: Boiled eggs, grilled chicken, fish, paneer\n"
        rec += "- Fruits: Apple, guava, berries (low GI fruits)\n"
        rec += "- Avoid: Sugar, white rice, sweets, bakery items, soft drinks\n"
        rec += "- Drink: Plenty of water, green tea\n"

    # MEDIUM RISK DIET
    elif risk_level == "MEDIUM":

        rec += "**⚠️ Balanced Diet Plan:**\n"
        rec += "- Eat: Whole grains, vegetables, sprouts, dal\n"
        rec += "- Include fiber-rich foods to control glucose spikes\n"
        rec += "- Limit: Fried foods, sugar, processed snacks\n"
        rec += "- Prefer: Home-cooked meals over outside food\n"

    # LOW RISK DIET
    else:

        rec += "**✅ Healthy Maintenance Diet:**\n"
        rec += "- Maintain balanced diet with carbs, protein, and fats\n"
        rec += "- Include fruits, vegetables, dairy, nuts\n"
        rec += "- Avoid excessive junk food and sugary drinks\n"

    # Final summary
    rec += "\n### 📌 Overall Recommendation:\n"

    if risk_level == "HIGH":
        rec += "🚨 Immediate doctor consultation + strict lifestyle changes required."
    elif risk_level == "MEDIUM":
        rec += "⚠️ Lifestyle modifications and regular monitoring needed."
    else:
        rec += "✅ Maintain current healthy lifestyle."

    return rec
# --------------------------------
# Page configuration
# --------------------------------
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    layout="wide"
)

# --------------------------------
# Background
# --------------------------------
def set_background(image_file):
    with open(image_file, "rb") as img:
        encoded = base64.b64encode(img.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background: 
                linear-gradient(rgba(0,0,0,0.6), rgba(0,0,0,0.6)),
                url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background("images/capsule.jpg")
st.markdown("""
<style>

/* ================= GLOBAL TEXT ================= */
html, body, [class*="css"]  {
    color: white !important;
}

/* All headings */
h1, h2, h3, h4, h5, h6 {
    color: white !important;
}

/* Only text elements, NOT all div/span */
p, label {
    color: white !important;
}

/* ================= SIDEBAR ================= */
[data-testid="stSidebar"] {
    background: rgba(0,0,0,0.6);
}

/* Sidebar text */
[data-testid="stSidebar"] * {
    color: white !important;
}

/* ================= MARKDOWN FIX (VERY IMPORTANT) ================= */
[data-testid="stMarkdownContainer"] {
    color: white !important;
}

[data-testid="stMarkdownContainer"] * {
    color: white !important;
}

/* Bullet points */
[data-testid="stMarkdownContainer"] li {
    color: white !important;
}

/* ================= RESULT CARD ================= */
.result-card {
    background-color: rgba(255,255,255,0.95);
    color: black !important;
    padding: 35px;
    border-radius: 15px;
    box-shadow: 0px 8px 25px rgba(0,0,0,0.4);
    text-align: center;
    font-size: 24px;
    margin: 20px auto;
    width: 80%;   /* 🔥 makes it broad */
}

/* Risk colors */
.high-risk {
    color: red !important;
    font-weight: bold;
}

.medium-risk {
    color: orange !important;
    font-weight: bold;
}

.low-risk {
    color: green !important;
    font-weight: bold;
}

/* ================= TABS ================= */
button[data-baseweb="tab"] {
    font-size: 16px;
    margin-right: 10px;
    padding: 10px 18px;
    border-radius: 8px;
}

/* Active tab */
button[data-baseweb="tab"][aria-selected="true"] {
    background-color: #4CAF50;
    color: white !important;
}

/* Hover */
button[data-baseweb="tab"]:hover {
    background-color: #ddd;
    color: black !important;
}

/* ================= INPUT BOXES ================= */
input, textarea, select {
    background-color: rgba(255,255,255,0.9) !important;
    color: black !important;
}

/* ================= BUTTON ================= */
.stButton button {
    background-color: #4CAF50;
    color: white;
    border-radius: 8px;
}

/* ================= REMOVE DULLNESS ================= */
.block-container {
    background: transparent;
}
/* ================= SELECTBOX FIX (FINAL) ================= */

/* Main select box text */
div[data-baseweb="select"] * {
    color: black !important;
}

/* Selected value */
div[data-baseweb="select"] span {
    color: black !important;
}

/* Dropdown box */
div[data-baseweb="select"] {
    background-color: rgba(255,255,255,0.95) !important;
    border-radius: 8px;
}

/* Dropdown options */
ul[role="listbox"] li {
    color: black !important;
    background-color: white !important;
}

/* Hover */
ul[role="listbox"] li:hover {
    background-color: #f0f0f0 !important;
    color: black !important;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------
# Load artifacts (UPDATED PATH)
# --------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("models/diabetes_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    feature_names = joblib.load("models/feature_names.pkl")
    return model, scaler, feature_names

model, scaler, feature_names = load_artifacts()

# --------------------------------
# SHAP explainer
# --------------------------------
@st.cache_resource
def load_shap_explainer(_model):
    return shap.TreeExplainer(_model)

explainer = load_shap_explainer(model)

# --------------------------------
# Title
# --------------------------------
st.markdown(
    """
    <h1 style="text-align:center;">🩺 Diabetes Risk Predictor</h1>
    <p style="text-align:center; color:gray;">
    Explainable Machine Learning Dashboard
    </p>
    """,
    unsafe_allow_html=True
)

# --------------------------------
# Sidebar inputs
# --------------------------------
col1, col2, col3 = st.sidebar.columns([1,2,1]) 
with col2: 
    st.image("images/patient_icon.png", width=120)
st.sidebar.header("🧑‍⚕️ Patient Inputs")

age = st.sidebar.slider("Age", 1, 100, 40)
bmi = st.sidebar.slider("BMI", 10.0, 60.0, 25.0)
hba1c = st.sidebar.slider("HbA1c Level", 3.0, 15.0, 5.5)
glucose = st.sidebar.slider("Blood Glucose Level", 50, 300, 120)

hypertension = st.sidebar.selectbox("Hypertension", ["No", "Yes"])
heart_disease = st.sidebar.selectbox("Heart Disease", ["No", "Yes"])

gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
smoking = st.sidebar.selectbox("Smoking History", ["never", "former", "current"])

# --------------------------------
# Prepare input dataframe (FIXED)
# --------------------------------
input_data = pd.DataFrame(0, index=[0], columns=feature_names)

input_data["age"] = age
input_data["bmi"] = bmi
input_data["HbA1c_level"] = hba1c   # ✅ FIXED
input_data["blood_glucose_level"] = glucose
input_data["hypertension"] = 1 if hypertension == "Yes" else 0
input_data["heart_disease"] = 1 if heart_disease == "Yes" else 0

# One-hot encoding (SAFE)
for col in feature_names:
    if col == "gender_Male" and gender == "Male":
        input_data[col] = 1

    if col == "smoking_history_former" and smoking == "former":
        input_data[col] = 1

    if col == "smoking_history_current" and smoking == "current":
        input_data[col] = 1

# --------------------------------
# Prediction button
# --------------------------------
if st.button("🔍 Predict Diabetes Risk"):

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    shap_values = explainer.shap_values(input_scaled)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    st.session_state["probability"] = probability
    st.session_state["shap_values"] = shap_values

    if probability >= 0.15:
        st.session_state["risk"] = "HIGH"
    elif probability >= 0.05:
        st.session_state["risk"] = "MEDIUM"
    else:
        st.session_state["risk"] = "LOW"

# --------------------------------
# OUTPUT SECTION (SAME STRUCTURE)
# --------------------------------
if "risk" in st.session_state:

    st.markdown("---")
    st.subheader("⚕️Prediction Result")

    risk = st.session_state["risk"]
    probability = st.session_state["probability"]

    if risk == "HIGH":
        css_class = "high-risk"
        message = f"🚨 HIGH RISK — Probability: {probability*100:.2f}%"
    elif risk == "MEDIUM":
        css_class = "medium-risk"
        message = f"⚠️ MEDIUM RISK — Probability: {probability*100:.2f}%"
    else:
        css_class = "low-risk"
        message = f"✅ LOW RISK — Probability: {probability*100:.2f}%"

    st.markdown(
        f"""
        <div class="result-card">
            <p class="{css_class}">{message}</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    # Create Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Global Analysis",
        "🔍 SHAP Explanation",
        "📄 Clinical Summary",
        "🥗 Recommendations"
    ])

    # ---------------- TAB 1 ----------------
    with tab1:

        #st.markdown("## 📊 Model & Patient Analysis")

        # ==============================
        # 🔹 GLOBAL DATASET ANALYSIS
        # ==============================
        st.markdown("### 📊 Global Feature Importance (Dataset Level)")

        importance = model.feature_importances_

        imp_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importance
        }).sort_values(by="Importance", ascending=False).head(8)

        # Clean names
        imp_df["Feature"] = imp_df["Feature"].str.replace("_", " ").str.title()

        fig1, ax1 = plt.subplots(figsize=(7,4))

        ax1.barh(
            imp_df["Feature"],
            imp_df["Importance"],
            color="#4C72B0"
        )

        ax1.invert_yaxis()
        ax1.set_xlabel("Importance Score")
        ax1.set_title("Top Features Influencing the Model")

        st.pyplot(fig1)

        st.info("This shows which features are generally important across the dataset.")

        st.divider()

        # ==============================
        # 🔹 PATIENT RISK ANALYSIS
        # ==============================
        st.markdown("### 🏥 Patient-Specific Risk Drivers")

        shap_vals = st.session_state["shap_values"][0]

        explain_df = pd.DataFrame({
            "Feature": feature_names,
            "Impact": shap_vals
        })

        # Rank by impact
        explain_df["abs_impact"] = explain_df["Impact"].abs()
        explain_df = explain_df.sort_values(by="abs_impact", ascending=False).head(8)

        # Clean names
        explain_df["Feature"] = explain_df["Feature"].str.replace("_", " ").str.title()

        fig2, ax2 = plt.subplots(figsize=(7,4))

        # Color logic (risk vs protective)
        colors = ["#d62728" if x > 0 else "#2ca02c" for x in explain_df["Impact"]]

        ax2.barh(
            explain_df["Feature"],
            explain_df["Impact"],
            color=colors
        )

        ax2.invert_yaxis()
        ax2.set_xlabel("Impact on Prediction")
        ax2.set_title("Top Features Affecting This Patient")

        st.pyplot(fig2)

        st.caption("🔴 Increases Risk  |  🟢 Reduces Risk")
        # importance = model.feature_importances_

        # imp_df = pd.DataFrame({
        #     "Feature": feature_names,
        #     "Importance": importance
        # }).sort_values(by="Importance", ascending=False).head(10)

        # st.bar_chart(imp_df.set_index("Feature"))
        

    # ---------------- TAB 2 ----------------
    with tab2:
        shap_vals = st.session_state["shap_values"]

        fig, ax = plt.subplots()
        shap.plots.bar(
            shap.Explanation(
                values=shap_vals[0],
                base_values=explainer.expected_value,
                data=input_data.iloc[0],
                feature_names=feature_names
            ),
            show=False
        )
        st.pyplot(fig)

    # ---------------- TAB 3 ----------------
    with tab3:

        shap_array = st.session_state["shap_values"][0]

        explain_df = pd.DataFrame({
            "Feature": feature_names,
            "Impact": shap_array
        })

        # Sort by absolute impact
        explain_df["abs_impact"] = explain_df["Impact"].abs()

        top_features = explain_df.sort_values(
            "abs_impact", ascending=False
        ).head(3)

        clinical_summary = generate_dynamic_ai_summary(
            risk,
            probability,
            top_features,
            input_data
        )

        st.markdown(clinical_summary)
    # ---------------- TAB 4 ----------------
    with tab4:
        recommendations = generate_personalized_recommendations(input_data, risk)
        st.markdown(recommendations)
else:  
    st.info("👉 Enter patient details and click Predict to view results.")
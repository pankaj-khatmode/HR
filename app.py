import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Set page config
st.set_page_config(
    page_title="Employee Attrition Analysis",
    page_icon="👨‍💼",
    layout="wide"
)

# Load dataset
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/pankaj-khatmode/HR/main/WA_Fn-UseC_-HR-Employee-Attrition.csv"
    df = pd.read_csv(url, quotechar='"', on_bad_lines='skip')
    df.columns = df.columns.str.strip()
    return df

# Encode categorical columns
@st.cache_data
def preprocess_data(df):
    df_encoded = df.copy()
    label_encoders = {}
    for col in df_encoded.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        label_encoders[col] = le
    return df_encoded, label_encoders

# Train model
@st.cache_resource
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return model, accuracy

# Load and preprocess data
with st.spinner("Loading data and training model..."):
    df = load_data()
    if "Attrition" not in df.columns:
        st.error("'Attrition' column not found!")
        st.stop()
    df_encoded, label_encoders = preprocess_data(df)
    X = df_encoded.drop("Attrition", axis=1)
    y = df_encoded["Attrition"]
    model, accuracy = train_model(X, y)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Data Analysis", "Predict Attrition"])

# Home Page
if page == "Home":
    st.title("👨‍💼 Employee Attrition Analysis & Prediction")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Employees", len(df))
    with col2:
        attrition_rate = df['Attrition'].value_counts(normalize=True).get('Yes', 0)*100
        st.metric("Attrition Rate", f"{attrition_rate:.1f}%")
    with col3:
        st.metric("Model Accuracy", f"{accuracy:.2f}")
    st.write("Use the sidebar to explore data or predict attrition risk.")

# Data Analysis Page
elif page == "Data Analysis":
    st.title("Data Analysis")
    st.subheader("Dataset Preview")
    st.dataframe(df.head())
    st.subheader("Attrition Distribution")
    fig, ax = plt.subplots()
    df['Attrition'].value_counts().plot(kind='bar', ax=ax, color=['#4CAF50', '#F44336'])
    st.pyplot(fig)

    st.subheader("Age Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df['Age'], bins=20, kde=True, ax=ax)
    st.pyplot(fig)

    st.subheader("Correlation Heatmap")
    corr = df_encoded.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
    st.pyplot(fig)

# Prediction Page
else:
    st.title("Predict Employee Attrition")
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.slider("Age", 18, 60, 35)
            job_satisfaction = st.slider("Job Satisfaction (1-4)", 1, 4, 3)
            monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=20000, value=5000)
            years_at_company = st.slider("Years at Company", 0, 40, 5)
        with col2:
            job_level = st.slider("Job Level (1-5)", 1, 5, 3)
            work_life_balance = st.slider("Work Life Balance (1-4)", 1, 4, 3)
            years_since_last_promotion = st.slider("Years Since Last Promotion", 0, 15, 2)
        submitted = st.form_submit_button("Predict")

    if submitted:
        input_data = X.mean().to_dict()
        input_data.update({
            'Age': age,
            'JobSatisfaction': job_satisfaction,
            'MonthlyIncome': monthly_income,
            'YearsAtCompany': years_at_company,
            'JobLevel': job_level,
            'WorkLifeBalance': work_life_balance,
            'YearsSinceLastPromotion': years_since_last_promotion
        })
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0]

        st.subheader("Prediction Result")
        if prediction == 1:
            st.error(f"⚠️ High Risk of Attrition ({prediction_proba[1]*100:.1f}%)")
        else:
            st.success(f"🎉 Low Risk of Attrition ({prediction_proba[0]*100:.1f}%)")

        st.subheader("Key Influential Features")
        coef_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.coef_[0]
        }).sort_values('Importance', key=abs, ascending=False).head(5)
        st.bar_chart(coef_df.set_index('Feature'))

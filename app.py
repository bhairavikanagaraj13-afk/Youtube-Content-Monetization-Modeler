import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

st.title("YouTube Content Monetization Modeler")
st.write("Predict YouTube ad revenue and explore content performance insights")

# -----------------------------
# Load Model + Scaler
# -----------------------------
@st.cache_resource
def load_model():
    model = joblib.load("C:\\Users\\Bhairavi\\OneDrive\\DataScience\\Youtube Content Monetization\\linear_regression_model.pkl")
    scaler = joblib.load("C:\\Users\\Bhairavi\\OneDrive\\DataScience\\Youtube Content Monetization\\scaler.pkl")
    return model, scaler

model, scaler = load_model()

# -----------------------------
# Load Dataset (for analytics)
# -----------------------------
df = pd.read_csv("C:\\Users\\Bhairavi\\OneDrive\\GUVI- DSA\\Projects\\Project 3 - Youtube monetization\\youtube_ad_revenue_dataset.csv")

# -----------------------------
# Sidebar Navigation
# -----------------------------
menu = st.sidebar.selectbox(
    "Navigation",
    ["Revenue Prediction", "Visual Analytics", "Model Insights"]
)

# =====================================================
# 1️⃣ REVENUE PREDICTION
# =====================================================
if menu == "Revenue Prediction":

    st.header("Predict Ad Revenue")

    views = st.number_input("Views", min_value=0)
    likes = st.number_input("Likes", min_value=0)
    comments = st.number_input("Comments", min_value=0)
    watch_time_minutes = st.number_input("Watch Time (minutes)", min_value=0)
    video_length_minutes = st.number_input("Video Length (minutes)", min_value=0)
    subscribers = st.number_input("Subscribers", min_value=0)

    category = st.selectbox("Category",
                            ["Entertainment","Gaming","Lifestyle","Music","Tech"])

    device = st.selectbox("Device",
                          ["Mobile","TV","Tablet"])

    country = st.selectbox("Country",
                           ["CA","DE","IN","UK","US"])

    # Feature Engineering
    engagement_rate = (likes + comments) / views if views != 0 else 0
    like_rate = likes / views if views != 0 else 0
    watch_time_per_view = watch_time_minutes / views if views != 0 else 0
    subscriber_engagement = (likes + comments) / subscribers if subscribers != 0 else 0

    # One Hot Encoding
    category_Entertainment = 1 if category=="Entertainment" else 0
    category_Gaming = 1 if category=="Gaming" else 0
    category_Lifestyle = 1 if category=="Lifestyle" else 0
    category_Music = 1 if category=="Music" else 0
    category_Tech = 1 if category=="Tech" else 0

    device_Mobile = 1 if device=="Mobile" else 0
    device_TV = 1 if device=="TV" else 0
    device_Tablet = 1 if device=="Tablet" else 0

    country_CA = 1 if country=="CA" else 0
    country_DE = 1 if country=="DE" else 0
    country_IN = 1 if country=="IN" else 0
    country_UK = 1 if country=="UK" else 0
    country_US = 1 if country=="US" else 0

    if st.button("Predict Revenue"):

        input_data = np.array([[

            views,
            likes,
            comments,
            watch_time_minutes,
            video_length_minutes,
            subscribers,

            engagement_rate,
            like_rate,
            watch_time_per_view,
            subscriber_engagement,

            category_Entertainment,
            category_Gaming,
            category_Lifestyle,
            category_Music,
            category_Tech,

            device_Mobile,
            device_TV,
            device_Tablet,

            country_CA,
            country_DE,
            country_IN,
            country_UK,
            country_US

        ]])

        input_scaled = scaler.transform(input_data)

        prediction = model.predict(input_scaled)

        st.success(f"Estimated Ad Revenue: ${prediction[0]:.2f}")

# =====================================================
# 2️⃣ VISUAL ANALYTICS
# =====================================================
elif menu == "Visual Analytics":

    st.header("YouTube Performance Analytics")

    # Views vs Revenue
    st.subheader("Views vs Ad Revenue")
    fig, ax = plt.subplots()
    ax.scatter(df["views"], df["ad_revenue_usd"])
    ax.set_xlabel("Views")
    ax.set_ylabel("Ad Revenue")
    st.pyplot(fig)

    # Category Distribution
    st.subheader("Content Category Distribution")
    st.bar_chart(df["category"].value_counts())

    # Device Usage
    st.subheader("Device Usage")
    st.bar_chart(df["device"].value_counts())

# =====================================================
# 3️⃣ MODEL INSIGHTS
# =====================================================
elif menu == "Model Insights":

    st.header("Model Performance Insights")

    st.write("Best Performing Model: Linear Regression")

    metrics = pd.DataFrame({
        "Model":["Linear Regression","Polynomial Regression","Decision Tree","Random Forest","KNN"],
        "R2":[0.9526,0.9524,0.8965,0.9500,0.8927],
        "RMSE":[13.48,13.49,19.91,13.83,20.27],
        "MAE":[3.11,3.41,5.39,3.55,14.03]
    })

    st.table(metrics)

    st.write("""
    Key Insights:
    - Views and engagement metrics strongly influence revenue.
    - Linear Regression achieved the highest R² score (~95%).
    - Higher watch time increases monetization potential.
    """)
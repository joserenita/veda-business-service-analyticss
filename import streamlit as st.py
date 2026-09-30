import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Veda Business Analytics", layout="wide")

st.title("💼 Veda Technology Business & Service Analytics Dashboard")
st.markdown("Interactive machine learning application predicting client service conversion and lead engagement.")

@st.cache_data
def train_model():
    np.random.seed(42)
    n = 600
    df = pd.DataFrame({
        'website_visits': np.random.randint(1, 50, n),
        'time_on_site_mins': np.random.exponential(scale=10.0, size=n).clip(0.5, 45),
        'past_inquiries': np.random.poisson(lam=1.5, size=n),
        'high_intent_channel': np.random.choice([0, 1], n, p=[0.5, 0.5])
    })
    df['engagement_score'] = (df['website_visits'] * 0.4) + (df['time_on_site_mins'] * 0.6)
    df['inquiry_intensity'] = df['past_inquiries'] * df['engagement_score']
    score = (0.03 * df['website_visits'] + 0.04 * df['time_on_site_mins'] + 0.3 * df['high_intent_channel'])
    df['converted'] = (score > score.median()).astype(int)
    X = df[['website_visits', 'time_on_site_mins', 'past_inquiries', 'engagement_score', 'inquiry_intensity', 'high_intent_channel']]
    y = df['converted']
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X, y)
    return model, df

model, df = train_model()

st.sidebar.header("Input Customer Metrics")
visits = st.sidebar.slider("Website Visits", 1, 50, 15)
time_site = st.sidebar.slider("Time on Site (mins)", 1.0, 45.0, 18.5)
past_inq = st.sidebar.slider("Past Inquiries", 0, 10, 2)
channel = st.sidebar.selectbox("Marketing Channel", ['LinkedIn', 'Google Search', 'Instagram', 'Direct', 'Referral'])

high_intent = 1 if channel in ['LinkedIn', 'Referral'] else 0
eng_score = (visits * 0.4) + (time_site * 0.6)
intensity = past_inq * eng_score

features = np.array([[visits, time_site, past_inq, eng_score, intensity, high_intent]])
prob = model.predict_proba(features)[0][1]
prediction = 1 if prob >= 0.50 else 0

st.subheader("📊 Real-Time Lead Scoring Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("Engagement Score", f"{eng_score:.2f}")
col2.metric("Conversion Probability", f"{prob*100:.1f}%")
col3.metric("Predicted Status", "🔥 High-Value Client" if prediction == 1 else "⚡ Standard Lead")

st.markdown("---")

col_left, col_right = st.columns(2)
with col_left:
    st.subheader("📈 Website Engagement vs Conversion")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(data=df, x='time_on_site_mins', y='engagement_score', hue='converted', palette={0: 'crimson', 1: 'teal'}, alpha=0.7, ax=ax)
    ax.scatter([time_site], [eng_score], color='yellow', s=150, edgecolor='black', label='Current Input')
    ax.set_title("User Position on Engagement Boundary")
    ax.legend()
    st.pyplot(fig)

with col_right:
    st.subheader("🎯 Random Forest Feature Importance")
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    feat_names = ['Visits', 'Time on Site', 'Past Inquiries', 'Engagement', 'Intensity', 'High-Intent Channel']
    sns.barplot(x=model.feature_importances_, y=feat_names, palette='viridis', ax=ax2)
    ax2.set_title("Key Drivers of Business Conversion")
    st.pyplot(fig2)

st.success("App running smoothly on Streamlit Community Cloud!")
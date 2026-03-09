import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI Google Trends Dashboard", layout="wide")

st.title("🤖 AI Powered Google Trends Keyword Analysis")

# =============================
# Load Dataset
# =============================
# Load CSV
df = pd.read_csv(r"C:\Users\karan\Downloads\multiTimeline.csv", skiprows=2, header=None)

# Manually assign column names
df.columns = [
    "Date",
    "AI",
    "Machine Learning",
    "Data Science",
    "Python",
    "Deep Learning",
    "Big Data",
    "Neural Network",
    "NLP"
]

# Convert Date column
df["Date"] = pd.to_datetime(df["Date"])

# Convert numeric columns
df.iloc[:,1:] = df.iloc[:,1:].apply(pd.to_numeric, errors="coerce")

# Fill missing values
df = df.ffill()
# =============================
# Sidebar Keyword Selection
# =============================

keywords = list(df.columns[1:])

selected_keywords = st.sidebar.multiselect(
    "Choose Keywords",
    keywords,
    default=[keywords[0]]
)

# =============================
# KPI Cards
# =============================

st.subheader("📌 Key Metrics")

col1, col2, col3 = st.columns(3)

col1.metric("Total Records", len(df))
col2.metric("Keywords Available", len(keywords))
col3.metric("Latest Date", str(df["Date"].max().date()))

# =============================
# Trend Comparison Graph
# =============================

st.subheader("📈 Keyword Trend Comparison")

fig = px.line(
    df,
    x="Date",
    y=selected_keywords,
    title="Google Search Trend Comparison"
)

st.plotly_chart(fig, use_container_width=True)

# =============================
# Correlation Heatmap
# =============================

st.subheader("🔥 Keyword Correlation Heatmap")

corr = df[keywords].corr()

fig2, ax = plt.subplots(figsize=(8,6))

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    ax=ax
)

st.pyplot(fig2)

# =============================
# AI Prediction
# =============================

if selected_keywords:

    keyword = selected_keywords[0]

    st.subheader(f"🤖 AI Prediction for '{keyword}'")

    values = df[keyword].dropna().values.reshape(-1,1)

    if len(values) < 20:

        st.warning("Not enough data to train AI model")

    else:

        scaler = MinMaxScaler()

        scaled_data = scaler.fit_transform(values)

        X = []
        y = []

        sequence_length = 10

        for i in range(sequence_length, len(scaled_data)):
            X.append(scaled_data[i-sequence_length:i])
            y.append(scaled_data[i])

        X = np.array(X)
        y = np.array(y)

    from tensorflow.keras import Input

    model = Sequential([
            Input(shape=(X.shape[1],1)),
            LSTM(50, activation='relu'),
            Dense(1)
         ])

    model.compile(optimizer='adam', loss='mse')
    if len(X) == 0:
     st.warning("Not enough data to train the model.")
else:
    model.fit(X, y, epochs=5, verbose=0)

    last_sequence = scaled_data[-sequence_length:]

    last_sequence = last_sequence.reshape(1, sequence_length, 1)

    prediction = model.predict(last_sequence)

    prediction = scaler.inverse_transform(prediction)

    st.success(
        f"Predicted next search interest for {keyword}: {round(prediction[0][0],2)}"
        )

# =============================
# Top Search Days
# =============================

st.subheader("🏆 Highest Search Interest Days")

top = df.sort_values(
    by=selected_keywords[0],
    ascending=False
).head(10)

st.dataframe(top)

# =============================
# Download Dataset
# =============================

st.subheader("⬇ Download Dataset")

st.download_button(
    "Download CSV",
    df.to_csv(index=False),
    "google_trends_data.csv",
    "text/csv"
)
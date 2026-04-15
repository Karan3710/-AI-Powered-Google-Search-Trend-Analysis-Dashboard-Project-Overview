import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from prophet import Prophet
import sqlite3
import streamlit_authenticator as stauth
import random

st.set_page_config(page_title="AI Trends SaaS", layout="wide")

# =============================
# LOGIN SYSTEM
# =============================
# =============================
# SIMPLE LOGIN SYSTEM
# =============================
st.sidebar.title("🔐 Login")

username = st.sidebar.text_input("Username")
password = st.sidebar.text_input("Password", type="password")

if username != "karan" or password != "1234":
    st.warning("Enter correct username and password")
    st.stop()

st.sidebar.success("Logged in successfully")
# =============================
# DATABASE
# =============================
conn = sqlite3.connect("user.db")
c = conn.cursor()

c.execute("""
CREATE TABLE IF NOT EXISTS preferences (
    user TEXT,
    keyword TEXT
)
""")

def save_pref(user, keyword):
    c.execute("INSERT INTO preferences VALUES (?,?)", (user, keyword))
    conn.commit()

# =============================
# LOAD DATA
# =============================
df = pd.read_csv("multiTimeline.csv", skiprows=2, header=None)

df.columns = [
    "Date","AI","Machine Learning","Data Science","Python",
    "Deep Learning","Big Data","Neural Network","NLP"
]

df["Date"] = pd.to_datetime(df["Date"])
df.iloc[:,1:] = df.iloc[:,1:].apply(pd.to_numeric, errors="coerce")
df = df.ffill()

keywords = list(df.columns[1:])

# =============================
# SIDEBAR
# =============================
st.sidebar.title("⚙ Controls")

menu = ["Dashboard","Forecast","Settings"]
choice = st.sidebar.selectbox("Menu", menu)

selected_keywords = st.sidebar.multiselect(
    "Select Keywords", keywords, default=[keywords[0]]
)

start_date = st.sidebar.date_input("Start Date", df["Date"].min())
end_date = st.sidebar.date_input("End Date", df["Date"].max())

filtered_df = df[
    (df["Date"] >= pd.to_datetime(start_date)) &
    (df["Date"] <= pd.to_datetime(end_date))
]

# =============================
# DASHBOARD
# =============================
if choice == "Dashboard":

    st.title("📊 AI Trends Dashboard")

    # KPI
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Records", len(filtered_df))
    col2.metric("Keywords", len(keywords))
    col3.metric("Latest", filtered_df["Date"].max().strftime("%Y-%m-%d"))
    col4.metric("Peak", int(filtered_df[keywords].max().max()))

    # Trend
    st.subheader("📈 Trends")

    fig = px.line(
        filtered_df,
        x="Date",
        y=selected_keywords,
        template="plotly_dark"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Layout
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔥 Correlation")
        corr = filtered_df[keywords].corr()
        fig2, ax = plt.subplots()
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig2)

    with col2:
        st.subheader("🏆 Top Days")
        if selected_keywords:

         top = filtered_df.sort_values(
         by=selected_keywords[0],
         ascending=False
         ).head(10)

         st.dataframe(top)

        else:
         st.warning("⚠ Please select at least one keyword")

    # Monthly Heatmap
    st.subheader("📊 Monthly Heatmap")

    temp = filtered_df.copy()
    temp["Month"] = temp["Date"].dt.month
    temp["Year"] = temp["Date"].dt.year

    pivot = temp.pivot_table(
        values=selected_keywords[0],
        index="Month",
        columns="Year",
        aggfunc="mean"
    )

    fig3, ax = plt.subplots()
    sns.heatmap(pivot, cmap="coolwarm", annot=True, ax=ax)
    st.pyplot(fig3)

    # =============================
    # TREND SCORE (MOVE INSIDE)
    # =============================
    st.subheader("🎯 Trend Score")

    scores = {}

    for key in keywords:
        recent_avg = filtered_df[key].tail(5).mean()
        overall_avg = filtered_df[key].mean()

        if overall_avg == 0:
            score = 0
        else:
            score = round((recent_avg / overall_avg) * 100, 2)

        scores[key] = score

    score_df = pd.DataFrame({
        "Keyword": scores.keys(),
        "Score": scores.values()
    }).sort_values(by="Score", ascending=False)

    st.dataframe(score_df)

    st.success(f"🔥 Top: {score_df.iloc[0]['Keyword']}")

    # =============================
    # COUNTRY VIEW (MOVE INSIDE)
    # =============================
    st.subheader("🌍 Country View")

    countries = ["India","USA","UK","Germany","Canada"]

    country_df = pd.DataFrame({
        "Country": countries,
        "Interest": [random.randint(50,100) for _ in countries]
    })

    st.plotly_chart(
        px.bar(country_df, x="Country", y="Interest", template="plotly_dark")
    )
# =============================
# FORECAST
# =============================
elif choice == "Forecast":

    st.title("🔮 Forecast (30 Days)")

    if len(selected_keywords) > 3:
        st.warning("⚠ Select max 3 keywords for faster forecast")

    for keyword in selected_keywords:

        st.markdown(f"### 📊 Forecast for {keyword}")

        prophet_df = filtered_df[["Date", keyword]].rename(
            columns={"Date": "ds", keyword: "y"}
        )

        prophet_df = prophet_df.dropna()

        if len(prophet_df) < 10:
            st.warning(f"Not enough data for {keyword}")
            continue

        model = Prophet()
        model.fit(prophet_df)

        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)

        fig = px.line(
            forecast,
            x="ds",
            y="yhat",
            title=f"30-Day Forecast for {keyword}",
            template="plotly_dark"
        )

        st.plotly_chart(fig, use_container_width=True)

# =============================
# SETTINGS
# =============================
elif choice == "Settings":

    st.title("⚙ Settings")

    if st.button("Save Preference"):
        save_pref(username, selected_keywords[0])
        st.success("Saved!")

# =============================
# DOWNLOAD
# =============================
st.sidebar.download_button(
    "Download Data",
    filtered_df.to_csv(index=False),
    "trends.csv",
    "text/csv"
)

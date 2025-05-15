import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from model.arima_model import run_arima
from model.prophet_model import run_prophet
from model.lstm_model import run_lstm
from utils.preprocessing import load_and_preprocess

st.set_page_config(page_title="Stock Market Forecasting", layout="wide")

# --- SIDEBAR ---
st.sidebar.title("Navigation")
theme = st.sidebar.radio("Select Theme", ["Light", "Dark"])
model = st.sidebar.selectbox("Choose Model", ["ARIMA", "Prophet", "LSTM"])

st.sidebar.markdown("---")
uploaded_file = st.sidebar.file_uploader("Upload Stock Data (CSV)", type="csv")

# --- MAIN LAYOUT ---
st.title("Stock Market Time Series Forecasting Dashboard")

# Apply theme
if theme == "Dark":
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #0e1117;
            color: white;
        }
        .css-1v0mbdj p, .css-1v0mbdj h1, .css-1v0mbdj h2, .css-1v0mbdj h3 {
            color: white;
        }
        </style>
        """, unsafe_allow_html=True)

# Load data
if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv("data/Stock_data.csv")

df = load_and_preprocess(df)

# Tabs for visualization and modeling
tab1, tab2 = st.tabs(["**Data Overview**", "**Model Forecasting**"])

with tab1:
    st.subheader("Stock Closing Price")
    st.line_chart(df['close'])

    st.subheader("Raw Data")
    st.dataframe(df.tail(10), use_container_width=True)

with tab2:
    st.subheader(f"{model} Forecast")

    if model == "ARIMA":
        with st.expander("What is ARIMA?", expanded=False):
            st.write("ARIMA is a statistical model that combines autoregression, differencing, and moving averages.")
        forecast = run_arima(df)

    elif model == "Prophet":
        with st.expander("What is Prophet?", expanded=False):
            st.write("Prophet is a time series model developed by Facebook that is robust to missing data and trend shifts.")
        forecast = run_prophet(df)

    elif model == "LSTM":
        with st.expander("What is LSTM?", expanded=False):
            st.write("LSTM (Long Short-Term Memory) is a deep learning model good at learning long-term dependencies in time series data.")
        forecast = run_lstm(df)

    st.line_chart(pd.concat([df['close'].iloc[-30:], forecast], axis=0))

    st.subheader("Forecast Table")
    st.dataframe(forecast.to_frame(name="Forecast"), use_container_width=True)

# Footer
st.markdown("---")
st.markdown("**Developed by [Your Name] | Data Analytics Intern Project**")
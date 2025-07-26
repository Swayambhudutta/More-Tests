# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense

# Set page config
st.set_page_config(page_title="Power Demand Forecasting", layout="wide")

# Title
st.title("🔌 Power Demand Forecasting Dashboard")
st.markdown("Upload an Excel file with 15-minute block data for power demand across any Indian state.")

# File uploader
uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])

# Function to preprocess data
def preprocess_data(df):
    df["Datetime"] = pd.to_datetime(df["Date"].astype(str) + " " + df["Time"].astype(str))
    df.sort_values("Datetime", inplace=True)
    demand_series = df["Power Demand (MW)"].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(demand_series)
    return scaled_data, scaler

# Function to create sequences
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

# Function to build and train model
def train_model(model_type, X_train, y_train):
    model = Sequential()
    if model_type == "LSTM":
        model.add(LSTM(50, activation='relu', input_shape=(X_train.shape[1], 1)))
    else:
        model.add(GRU(50, activation='relu', input_shape=(X_train.shape[1], 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=50, verbose=0)
    return model

# Main logic
if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)
        required_columns = ["State", "Date", "Time", "Power Demand (MW)"]
        if not all(col in df.columns for col in required_columns):
            st.error("Excel file must contain columns: State, Date, Time, Power Demand (MW)")
        elif len(df) < 100:
            st.error("Excel file must contain at least 100 data points.")
        else:
            st.success("✅ File uploaded and validated successfully!")

            # Preprocess
            scaled_data, scaler = preprocess_data(df)
            train_data = scaled_data[:70]
            test_data = scaled_data[70:100]

            # Create sequences
            X_train, y_train = create_sequences(train_data, seq_length=10)
            X_test, y_test = create_sequences(test_data, seq_length=10)

            # Reshape for model
            X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
            X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

            # Train models
            lstm_model = train_model("LSTM", X_train, y_train)
            gru_model = train_model("GRU", X_train, y_train)

            # Predict
            lstm_preds = lstm_model.predict(X_test)
            gru_preds = gru_model.predict(X_test)

            # Inverse transform
            lstm_preds_inv = scaler.inverse_transform(lstm_preds)
            gru_preds_inv = scaler.inverse_transform(gru_preds)
            y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

            # Error metrics
            lstm_mae = mean_absolute_error(y_test_inv, lstm_preds_inv)
            lstm_rmse = np.sqrt(mean_squared_error(y_test_inv, lstm_preds_inv))
            gru_mae = mean_absolute_error(y_test_inv, gru_preds_inv)
            gru_rmse = np.sqrt(mean_squared_error(y_test_inv, gru_preds_inv))

            # Plot error metrics
            st.subheader("📈 Error Metrics Comparison")
            fig, ax = plt.subplots()
            metrics = ["MAE", "RMSE"]
            lstm_scores = [lstm_mae, lstm_rmse]
            gru_scores = [gru_mae, gru_rmse]
            x = np.arange(len(metrics))
            ax.bar(x - 0.2, lstm_scores, width=0.4, label='LSTM')
            ax.bar(x + 0.2, gru_scores, width=0.4, label='GRU')
            ax.set_xticks(x)
            ax.set_xticklabels(metrics)
            ax.set_ylabel("Error")
            ax.set_title("LSTM vs GRU Error Metrics")
            ax.legend()
            st.pyplot(fig)

            # Model recommendation
            better_model = "LSTM" if (lstm_mae + lstm_rmse) < (gru_mae + gru_rmse) else "GRU"
            st.success(f"✅ Based on error metrics, **{better_model}** is more suitable for forecasting this dataset.")

    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")

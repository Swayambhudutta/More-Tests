import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("📈 Power Demand Trend Viewer")

# Upload Excel file
uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx"])

if uploaded_file:
    try:
        # Read the Excel file
        df = pd.read_excel(uploaded_file, engine='openpyxl')

        # Check for required columns
        required_columns = ["Date", "Time", "Power Demand (MW)"]
        if not all(col in df.columns for col in required_columns):
            st.error("Excel file must contain columns: Date, Time, Power Demand (MW)")
        else:
            # Combine Date and Time into a single datetime column
            df["Datetime"] = pd.to_datetime(df["Date"].astype(str) + " " + df["Time"].astype(str))
            df.sort_values("Datetime", inplace=True)

            # Plot the trend line
            st.subheader("Power Demand Trend")
            fig, ax = plt.subplots()
            ax.plot(df["Datetime"], df["Power Demand (MW)"], color='blue', linewidth=2)
            ax.set_xlabel("Time")
            ax.set_ylabel("Power Demand (MW)")
            ax.set_title("Power Demand Over Time")
            plt.xticks(rotation=45)
            st.pyplot(fig)

    except Exception as e:
        st.error(f"Error reading Excel file: {e}")

import streamlit as st
import pandas as pd

st.title("📊 Excel Viewer")

# Upload Excel file
uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx"])

# Display the contents of the Excel file
if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file, engine='openpyxl')
        st.success("File uploaded successfully!")
        st.dataframe(df)
    except Exception as e:
        st.error(f"Error reading Excel file: {e}")

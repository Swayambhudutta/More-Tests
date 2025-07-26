import streamlit as st
import pandas as pd

st.title("✅ Streamlit Excel Viewer")

# File uploader
uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx"])

# Display contents
if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file, engine='openpyxl')
        st.success("File uploaded successfully!")
        st.dataframe(df)
    except Exception as e:
        st.error(f"Error reading Excel file: {e}")

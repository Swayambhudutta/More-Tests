
# test_app.py

import streamlit as st
import pandas as pd

st.title("✅ Streamlit Test App")
st.write("If you see this, your environment is working!")

uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.write(df.head())

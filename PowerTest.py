import streamlit as st

st.title("✅ Streamlit Test App")
st.write("If you see this, your environment is working!")

uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx"])
if uploaded_file:
    st.success("File uploaded successfully!")

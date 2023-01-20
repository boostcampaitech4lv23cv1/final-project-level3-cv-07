import streamlit as st

import requests

backend = "http://115.85.182.51:30002"
def app():
    uploaded_file = st.file_uploader("Choose a Video file")
    if uploaded_file is not None:
        bytes_data = uploaded_file.read()
        st.write("filename:", uploaded_file.name)
        requests.post(f"{backend}/save_video", bytes_data)
        # st.write(bytes_data)

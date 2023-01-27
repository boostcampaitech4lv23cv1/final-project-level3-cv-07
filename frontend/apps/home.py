import streamlit as st

import requests

backend = "http://115.85.182.51:30002"


def apply_model(encoded_video):
    requests.post(f"{backend}/save_video", encoded_video)


def app():
    uploaded_file = st.file_uploader("Choose a Video file")
    if uploaded_file is not None:
        encoded_video = uploaded_file.read()
        st.video(encoded_video)

        st.button("Apply model", "/", on_click=apply_model, args=(encoded_video,))

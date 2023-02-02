import streamlit as st

import requests

backend = "http://115.85.182.51:30002"

def apply_model(encoded_video, encoded_image):
    with st.spinner('Wait for it...'):
        requests.post(f"{backend}/upload/video", encoded_video)
        requests.post(f"{backend}/upload/image", encoded_image)
        requests.get(f"{backend}/req_infer")


def app():
    uploaded_video = st.file_uploader("Choose a Video file")
    if uploaded_video is not None:
        encoded_video = uploaded_video.read()
        st.video(encoded_video)
        
    choose_image = st.file_uploader("Choose a Image file")
    if choose_image is not None:
        encoded_image = choose_image.read()
        st.image(encoded_image)

        st.button("Apply model", "/", on_click=apply_model, args=(encoded_video, encoded_image))

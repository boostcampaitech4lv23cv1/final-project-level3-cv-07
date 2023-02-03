import streamlit as st
from streamlit_option_menu import option_menu

from apps import blank

st.set_page_config(page_title="Streamlit Geospatial", layout="wide")

# A dictionary of apps in the format of {"App title": "App icon"}
# More icons can be found here: https://icons.getbootstrap.com

import requests

backend = "http://115.85.182.51:30002"

def nextpage(): st.session_state.page += 1

if __name__ == "__main__":    
    if "page" not in st.session_state:
        st.session_state.page = 0

    placeholder = st.empty()
    
    if st.session_state.page == 0:
        with placeholder.container():
            uploaded_video = st.file_uploader("Choose a Video file")
            if uploaded_video is not None:
                encoded_video = uploaded_video.read()
                st.video(encoded_video)
                st.session_state.encoded_video = encoded_video
                
            choose_image = st.file_uploader("Choose a Image file")
            if choose_image is not None:
                encoded_image = choose_image.read()
                st.image(encoded_image)
                st.session_state.encoded_image = encoded_image
            st.button("Apply model", on_click=nextpage) #, on_click=apply_model, args=(encoded_video, encoded_image)):

    
    elif st.session_state.page == 1:
        with st.spinner('Wait for it...'):
            requests.post(f"{backend}/upload/video", st.session_state.encoded_video)
            requests.post(f"{backend}/upload/image", st.session_state.encoded_image)
            requests.get(f"{backend}/req_infer")
                
        st.button("결과 확인", on_click=nextpage)
    
    elif st.session_state.page == 2:
        cartoon_video = open("database/cartoonized_video/cartoonized.mp4", "rb")
        video_bytes = cartoon_video.read()
        st.video(video_bytes)   # 읽지만 영상 로드 안됨.
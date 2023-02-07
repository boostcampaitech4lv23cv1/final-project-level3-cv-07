import streamlit as st
from streamlit_option_menu import option_menu


st.set_page_config(page_title="Streamlit Geospatial", layout="wide")

# A dictionary of apps in the format of {"App title": "App icon"}
# More icons can be found here: https://icons.getbootstrap.com

import requests

backend = "http://115.85.182.51:30002"

if __name__ == "__main__":
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # 비디오 업로드
        uploaded_video = st.file_uploader("Choose a Video file")
        if uploaded_video is not None:
            encoded_video = uploaded_video.read()
            st.video(encoded_video)
            encoded_video = encoded_video
        
        # 이미지 업로드
        choose_image = st.file_uploader("Choose a Image file")
        if choose_image is not None:
            encoded_image = choose_image.read()
            st.image(encoded_image, width = 300)
            encoded_image = encoded_image
    
    
    with col2:
        if st.button("Apply model"):
            # 순차 Request
            with st.spinner():
                requests.post(f"{backend}/upload/video", encoded_video)
                requests.post(f"{backend}/upload/image", encoded_image)
                requests.get(f"{backend}/req_infer")
            
            # 결과 영상 도출
            with col3:
                cartoon_video = open("database/cartoonized_video/cartoonized.mp4", "rb")
                video_bytes = cartoon_video.read()
                st.video(video_bytes)   # 읽지만 영상 로드 안됨.
            
                st.download_button(
                    label="Download Video",
                    data=video_bytes,
                    file_name='video.mp4',
                    mime="video/mp4",
                )

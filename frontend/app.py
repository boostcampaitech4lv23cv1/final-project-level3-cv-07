import streamlit as st
from streamlit_option_menu import option_menu
import time


st.set_page_config(page_title="Streamlit Geospatial", layout="wide")

# A dictionary of apps in the format of {"App title": "App icon"}
# More icons can be found here: https://icons.getbootstrap.com

import requests

backend = "http://49.50.160.138:30002"

if __name__ == "__main__":
    
    empty0, col0 = st.columns([0.1,0.95])
    res_empty0, res_col, res_empty1 = st.columns([0.1, 0.85, 0.1])
    empty1, col1, empty2, col2, empty3 = st.columns([0.1, 0.4, 0.05, 0.4, 0.1])
    empty4, message_col, col3, col4 = st.columns([0.1, 0.5, 0.25, 0.2])
    
    with empty0:
        st.empty()

    with col0:
        st.title(":blue[CAFE]: :blue[CA]rtoonize :blue[F]or :blue[E]xtra faces :art:")

    with empty1:
        st.empty()

    with col1:
        # 비디오 업로드
        uploaded_video = st.file_uploader(":one: **:red[Choose your Video file]** :video_camera:")
        if uploaded_video is not None:
            encoded_video = uploaded_video.read()
            st.video(encoded_video)
            encoded_video = encoded_video
    
    with empty2:
        st.empty()

    with col2:
        # 이미지 업로드
        choose_image = st.file_uploader(":two: **:red[Choose your Target image]** :camera_with_flash:")
        if choose_image is not None:
            encoded_image = choose_image.read()
            st.image(encoded_image, width = 300)
            encoded_image = encoded_image

    with empty3:
        st.empty()

    with empty4:
        st.empty()

    with col3:
        st.write(":three: **:red[Push the button for applying model]** 	:arrow_right:")

    with col4:
        if st.button("Apply model"):
            txt = open("database/sentences.txt", "r")
            txt_list = txt.readlines()
            cnt = 1
            with message_col:
                while(cnt <= len(txt_list)):
                    st.text(txt_list[cnt - 1])
                    time.sleep(7)
                    cnt += 1

            # 순차 Request
            with st.spinner():
                requests.post(f"{backend}/upload/video", encoded_video)
                requests.post(f"{backend}/upload/image", encoded_image)
                requests.get(f"{backend}/req_infer")

            
            # 결과 영상 도출
            txt.close()
            with res_empty0:
                st.empty()

            with res_col:
                st.balloons()
                cartoon_video = open("database/cartoonized_video/cartoonized.mp4", "rb")
                video_bytes = cartoon_video.read()
                st.video(video_bytes)   # 읽지만 영상 로드 안됨.
            
                st.download_button(
                    label="Download Video",
                    data=video_bytes,
                    file_name='video.mp4',
                    mime="video/mp4",
                )

            with res_empty1:
                st.empty()

import streamlit as st
from streamlit_option_menu import option_menu
import time
import asyncio
import aiohttp
import random

st.set_page_config(page_title="Streamlit Geospatial", layout="wide")

# A dictionary of apps in the format of {"App title": "App icon"}
# More icons can be found here: https://icons.getbootstrap.com

import requests

backend = "http://115.85.182.51:30002"

infer_trigger = False


async def task_sentence(message_col):
    global infer_trigger
    txt = open("database/sentences_rdm.txt", "r")
    txt_list = txt.readlines()

    with message_col:
        placeholder = st.empty()
        while not infer_trigger:
            idx = random.randint(0, len(txt_list) - 1)
            with placeholder:
                st.markdown(txt_list[idx])

            await asyncio.sleep(4)
    txt.close()


async def req_inference():
    global infer_trigger

    async def task(session, url):
        async with session.get(url) as response:
            return await response.text()

    async with aiohttp.ClientSession() as session:
        await task(session, f"{backend}/req_infer")
        infer_trigger = True


async def multi_task(message_col):
    task1 = asyncio.create_task(task_sentence(message_col))
    task2 = asyncio.create_task(req_inference())
    await task1
    await task2


if __name__ == "__main__":

    empty0, col0 = st.columns([0.1, 0.95])
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
        uploaded_video = st.file_uploader(
            ":one: **:red[Choose your Video file]** :video_camera:"
        )
        if uploaded_video is not None:
            encoded_video = uploaded_video.read()
            st.video(encoded_video)
            encoded_video = encoded_video

    with empty2:
        st.empty()

    with col2:
        # 이미지 업로드
        choose_image = st.file_uploader(
            ":two: **:red[Choose your Target image]** :camera_with_flash:"
        )
        if choose_image is not None:
            encoded_image = choose_image.read()
            st.image(encoded_image, width=300)
            encoded_image = encoded_image

    with empty3:
        st.empty()

    with empty4:
        st.empty()

    with col3:
        st.write(":three: **:red[Push the button for applying model]** 	:arrow_right:")

    with col4:
        if st.button("Apply model"):

            # 문장 호출 및 inference 요청
            requests.post(f"{backend}/upload/video", encoded_video)
            requests.post(f"{backend}/upload/image", encoded_image)
            with st.spinner():
                asyncio.run(multi_task(message_col))

            # 결과 영상 도출
            with res_empty0:
                st.empty()

            with res_col:
                st.balloons()
                cartoon_video = open("database/cartoonized_video/video.mp4", "rb")
                video_bytes = cartoon_video.read()
                st.video(video_bytes)  # 읽지만 영상 로드 안됨.

                st.download_button(
                    label="Download Video",
                    data=video_bytes,
                    file_name="video.mp4",
                    mime="video/mp4",
                )

            with res_empty1:
                st.empty()

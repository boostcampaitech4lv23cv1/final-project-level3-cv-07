import streamlit as st

import requests


def app():
    st.title("Home")
    
    if st.button("request"):
        res = requests.get("http://115.85.182.51:30002")
        st.write(f"{res.json()}")
    else:
        st.write("hi")
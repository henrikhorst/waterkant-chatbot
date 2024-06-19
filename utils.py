
import os
import streamlit as st

def get_api_secret():
    secret = os.environ.get('OPENAI_API_KEY')
    
    if not secret:
        secret = st.secrets["OPENAI_API_KEY"]
    
    return secret
    
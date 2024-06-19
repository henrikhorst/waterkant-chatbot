from openai import OpenAI
import streamlit as st
import helper_appv1
from utils import get_api_secret

st.set_page_config(page_title="🌊 Waterkant Chatbot🏄")

st.title("🌊 Waterkant Chatbot🏄")


key =  get_api_secret()
client = OpenAI(api_key=key)


if "openai_model" not in st.session_state:
    st.session_state["model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": helper_appv1.system_prompt},
      ]

for message in st.session_state.messages[1:]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if question := st.chat_input("What is up?"):
    
    with st.chat_message("user"):
        st.markdown(question)
    
    


    with st.chat_message("assistant"):
        answer = helper_appv1.get_answer(question, st.session_state.messages)
        st.markdown(answer)
    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.messages.append({"role": "assistant", "content": answer})
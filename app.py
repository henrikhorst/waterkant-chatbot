from openai import OpenAI
import streamlit as st
import helper_app
from utils import get_api_secret

st.set_page_config(page_title="🌊 Waterkant Chatbot🏄")

st.title("🌊 Waterkant Chatbot🏄")


key =  get_api_secret()
client = OpenAI(api_key=key)


if "openai_model" not in st.session_state:
    st.session_state["model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": helper_app.system_prompt},
      ]

for message in st.session_state.messages[1:]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if question := st.chat_input("What is up?"):
    
    with st.chat_message("user"):
        st.markdown(question)
    
    


    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model=st.session_state["model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ]+[{"role": "user", "content": helper_app.get_prompt(question, k=6)}],
            temperature=0.0,
            stream=True,
        )
        response = st.write_stream(stream)
    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.messages.append({"role": "assistant", "content": response})
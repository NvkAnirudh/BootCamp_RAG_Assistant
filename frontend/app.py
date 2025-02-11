import streamlit as st
import requests

st.title("Talk to the BootCamp!")
st.caption("Powered by o1-mini and Hybrid RAG")

if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.chat_input("Ask about your documents...")

if user_input:
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Get API response
    response = requests.post(
        "http://localhost:8000/chat",
        json={"message": user_input, "history": st.session_state.history}
    ).json()
    
    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(response["response"])
        with st.expander("View sources"):
            st.write("\n".join(response["context_sources"]))
    
    # Update history
    st.session_state.history.append((user_input, response["response"]))
    
    # Keep history bounded
    if len(st.session_state.history) > 3:
        st.session_state.history = st.session_state.history[-3:]
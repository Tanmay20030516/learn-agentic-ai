import streamlit as st
from backend import create_chatbot, initial_state, HumanMessage
import time

st.title("Simple Chatbot with LangGraph")

CONFIG = {"configurable": {"thread_id": "simple-chatbot-thread"}}

def stream_with_delay(stream):
    for message_chunk, metadata in stream:
        if message_chunk.content:
            yield message_chunk.content
            time.sleep(0.1)

if "chatbot" not in st.session_state:
    st.session_state.chatbot = create_chatbot()
    # Initialize the chatbot state
    st.session_state.chatbot.invoke(initial_state, config=CONFIG)  # type: ignore

if "chat_history" not in st.session_state:
    # each element is {"role": "user"/"assistant", "content": "message content" }
    st.session_state.chat_history = []

# display the history
for chat in st.session_state["chat_history"]:
    with st.chat_message(chat["role"]):
        st.markdown(chat["content"])

user_query = st.chat_input("Ask me anything!")

if user_query:
    st.session_state["chat_history"].append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)
    # let us stream the response i.e. show tokens as they are generated
    with st.chat_message("assistant"):
        stream = st.session_state["chatbot"].stream(
            input={"messages": [HumanMessage(content=user_query)]},
            config=CONFIG,
            stream_mode="messages",
        )
        ai_message = st.write_stream(
            # (message_chunk.content, time.sleep(0.1) for message_chunk, metadata in stream), # generator expression parenthesized
            stream_with_delay(stream),
            cursor="▌ ",
        )
    
    st.session_state['chat_history'].append({'role': 'assistant', 'content': ai_message})
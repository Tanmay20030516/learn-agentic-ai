import streamlit as st
from backend import chatbot, initial_state, HumanMessage

st.title("Simple Chatbot with LangGraph")

CONFIG = {"configurable": {"thread_id": "simple-chatbot-thread"}}

if "chatbot" not in st.session_state:
    st.session_state.chatbot = chatbot
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
    # Call the backend chatbot to get a response
    response = st.session_state["chatbot"].invoke({"messages": [HumanMessage(content=user_query)]}, config=CONFIG)
    ai_response = response["messages"][-1].content
    st.session_state["chat_history"].append({"role": "assistant", "content": ai_response})
    with st.chat_message("assistant"):
        st.markdown(ai_response)
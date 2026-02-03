import streamlit as st
from backend import (
    title_generator, chatbot, initial_state,
    HumanMessage, BaseMessage, AIMessage, SystemMessage,
)
import time
import uuid

# *************************************** Utility functions ***********************************************

def generate_thread_id():
    thread_id = uuid.uuid4()
    return str(thread_id)

def create_new_chat():
    thread_id = generate_thread_id() # generate new thread ID
    st.session_state['thread_id'] = thread_id # set thread ID
    save_thread(thread_id)
    st.session_state['chat_history'] = [] # clear previous thread conversation history

def save_thread(thread_id):
    """
    - add `thread_id` to `chat_threads`
    - will be called when we reload the app or while we create a new chat
    """
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)

def load_conversation(thread_id: str) -> list[BaseMessage]:
    state = st.session_state["chatbot"].get_state(config={"configurable": {"thread_id": thread_id}})
    return state.values.get("messages", []) # just in case no messages [new conversation], return empty list

def convert_messages_format(messages: list[BaseMessage]) -> list[dict]:
    """ convert from `[BaseMessage, ...]` to `[{"role": ..., "content": ...}, ...]` """
    new_format = []
    for message in messages:
        role = ""
        if isinstance(message, HumanMessage):
            role = "user"
        elif isinstance(message, AIMessage):
            role = "assistant"
        elif isinstance(message, SystemMessage):
            role = "system"
        new_format.append({"role": role, "content": message.content})
    return new_format

def stream_with_delay(stream):
    for message_chunk, metadata in stream:
        if message_chunk.content:
            yield message_chunk.content
            time.sleep(0.05)

# *************************************** Session setup ***************************************************

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = [] # set() is unordered, can mess up chat thread sequences
save_thread(st.session_state["thread_id"]) # needed when we reload our page/app

if "chat_history" not in st.session_state:
    # each element is {"role": "user"/"assistant", "content": "message content" }
    st.session_state["chat_history"] = []

if "chatbot" not in st.session_state:
    # st.session_state["chatbot"] = create_chatbot()
    st.session_state["chatbot"] = chatbot
    # Initialize the chatbot state
    # st.session_state["chatbot"].invoke(initial_state, config={"configurable": {"thread_id": st.session_state["thread_id"]}})  # type: ignore

if "chat_titles" not in st.session_state:
    st.session_state["chat_titles"] = {} # thread_id -> title

# *************************************** Sidebar UI ******************************************************

with st.sidebar:
    st.title("⚛️ ChatLG")
    
    st.header("⌛️ Past Conversations")
    
    new_chat = st.button("➕ New Chat")
    if new_chat: # if clicked
        create_new_chat()

    for thread_id in reversed(st.session_state["chat_threads"]): # so that recent chat is on top
        if st.button(f"{st.session_state["chat_titles"].get(thread_id, thread_id[:8])}", key=thread_id):
            # update the session thread ID
            st.session_state["thread_id"] = thread_id
            messages = load_conversation(thread_id)
            st.session_state["chat_history"] = convert_messages_format(messages)

# *************************************** Main UI *********************************************************

# display the history [in the chat window]
for chat in st.session_state["chat_history"]:
    if chat["role"] == "system": continue # we do not wish to print the system prompt
    with st.chat_message(chat["role"]):
        st.markdown(chat["content"])

CONFIG = {"configurable": {"thread_id": st.session_state["thread_id"]}}

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
    # so that title is generated only once per chat thread
    if (st.session_state['thread_id'] not in st.session_state["chat_titles"]) and len(st.session_state["chat_history"]) > 0:
        chat_title = title_generator.invoke({"context": st.session_state["chat_history"][:1]})["title"]
        st.session_state["chat_titles"][st.session_state['thread_id']] = chat_title # cache it
        st.rerun()
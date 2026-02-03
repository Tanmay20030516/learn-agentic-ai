import streamlit as st
from backend_w_database import (
    chatbot, initial_state, retrieve_all_threads,
    title_generator, save_chat_title, retrieve_all_titles,
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
    st.session_state["chat_threads"] = retrieve_all_threads() # fetch all existing threads from DB
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
    # st.session_state["chat_titles"] = {} # thread_id -> title
    st.session_state["chat_titles"] = retrieve_all_titles() # thread_id -> title

# *************************************** Sidebar UI ******************************************************

with st.sidebar:
    st.title(":material/network_intel_node: ChatLG")

    st.header(":material/history_2: Past Conversations")

    if st.button(":material/add_circle: New Chat", type="primary", use_container_width=True):
        create_new_chat()

    st.divider()

    for thread_id in reversed(st.session_state["chat_threads"]):

        title = st.session_state["chat_titles"].get(thread_id, thread_id[:8])

        col_title, col_menu = st.columns([0.85, 0.15], gap="small")

        # ---- RIGHT: Menu button ----
        with col_menu:
            click = st.button(
                ":material/more_horiz:",
                key=f"menu_{thread_id}",
                type="secondary",
            )
            # if click:
            #     st.session_state["active_menu_thread"] = thread_id
            #     # later: show rename / delete / export menu
        # ---- LEFT: Chat title button ----
        with col_title:
            if st.button(
                title,
                key=f"open_{thread_id}",
                type="secondary",
                use_container_width=True,
            ) or click:
                st.session_state["thread_id"] = thread_id
                messages = load_conversation(thread_id)
                st.session_state["chat_history"] = convert_messages_format(messages)


# *************************************** Main UI *********************************************************

# display the history [in the chat window]
for chat in st.session_state["chat_history"]:
    if chat["role"] == "system": continue # we do not wish to print the system prompt
    with st.chat_message(chat["role"]):
        st.markdown(chat["content"])


user_query = st.chat_input("Ask me anything!")

if user_query:
    st.session_state["chat_history"].append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)
    
    # CONFIG = {"configurable": {"thread_id": st.session_state["thread_id"]}}
    CONFIG = {
        "configurable": {"thread_id": st.session_state["thread_id"]},
        "metadata": {
            "thread_id": st.session_state["thread_id"]
        },
        # "run_name": "chat_turn", # for better tracking in LangSmith
    }

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
        save_chat_title(st.session_state['thread_id'], chat_title) # save to DB
        st.session_state["chat_titles"][st.session_state['thread_id']] = chat_title # cache it
        st.rerun()

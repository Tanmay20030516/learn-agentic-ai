import os
import sqlite3
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Any, Generator, Iterator
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver # pip install langgraph-checkpoint-sqlite
from langchain_groq.chat_models import ChatGroq

load_dotenv()

LLM = ChatGroq(model=os.getenv("MODEL_NAME", ""))

# *************************************** Chatbot Graph ***************************************************
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState) -> ChatState:
    response = LLM.invoke(state["messages"])
    return {"messages": [response]}

def create_chatbot():
    graph = StateGraph(ChatState)

    # nodes
    graph.add_node("chat", chat_node)

    # edges
    graph.add_edge(START, "chat")
    graph.add_edge("chat", END)
    return graph.compile(checkpointer=CHECKPOINTER)

def retrieve_all_threads() -> list[str]:
    ''' get all unique thread IDs from the database '''
    all_threads = set()
    for checkpoint in CHECKPOINTER.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id']) # type: ignore

    return list(all_threads)

# *************************************** Title Generator Graph ***************************************************
class ChatTitle(TypedDict):
    ''' maybe once try to include this as a conditional edge in the main chatbot graph '''
    title: str
    context: list[dict[str, str]]

def gen_title(state: ChatTitle):
    context = ""
    for msg in state["context"]:
        role = msg["role"]
        text = msg["content"]
        context += f"{role}: {text}\n"
    prompt = f"""
    You are an expert chat analysis expert. Analyze the below chat and give a descriptive 3-4 word title to it.
    Chats: {context}
    - STRICTLY return ONLY the TITLE nothing else.
    """
    response: str | Any = LLM.invoke(prompt).content
    return {"title": response.strip()}

def create_title_generator():
    title_graph = StateGraph(ChatTitle)

    # nodes
    title_graph.add_node("generate_title", gen_title)

    # edges
    title_graph.add_edge(START, "generate_title")
    title_graph.add_edge("generate_title", END)

    return title_graph.compile()

def _create_titles_table():
    ''' creates the titles table if it doesn't exist '''
    cursor = CONN.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS chat_titles (
        thread_id TEXT PRIMARY KEY,
        title TEXT NOT NULL
    )
    """)
    CONN.commit()

def save_chat_title(thread_id: str, title: str):
    ''' insert or update the chat title for a given thread_id '''
    cursor = CONN.cursor()
    cursor.execute(
        "INSERT OR REPLACE INTO chat_titles (thread_id, title) VALUES (?, ?)",
        (thread_id, title)
    )
    CONN.commit()

def retrieve_all_titles() -> dict[str, str]:
    ''' fetch all chat titles from the database '''
    cursor = CONN.cursor()
    cursor.execute("SELECT thread_id, title FROM chat_titles")
    rows = cursor.fetchall()
    return {row[0]: row[1] for row in rows}

# *************************************** Initialization ***************************************************
CONN = sqlite3.connect(database=os.getenv("DATABASE_URL", ""), check_same_thread=False)
CHECKPOINTER = SqliteSaver(conn=CONN)
_create_titles_table()
chatbot = create_chatbot()
title_generator = create_title_generator()
initial_state = {"messages": [SystemMessage(content="You are a helpful assistant. ONLY answer the question asked and end your response with a GenZ hashtag. DO NOT REPLY ANYTHING ELSE.")]}
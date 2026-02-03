from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Any
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langchain_groq.chat_models import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

LLM = ChatGroq(model=os.getenv("MODEL_NAME", ""))


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

class ChatTitle(TypedDict):
    ''' maybe once try to include this as a conditional edge in the main chatbot graph '''
    title: str
    context: list[dict[str, str]]

def chat_node(state: ChatState) -> ChatState:
    response = LLM.invoke(state["messages"])
    return {"messages": [response]}

def gen_title(state: ChatTitle):
    context = ""
    for msg in state["context"]:
        role = msg["role"]
        text = msg["content"]
        context += f"{role}: {text}\n"
    prompt = f""" You are an expert chat analysis expert. Analyze the below chat and give a 3-4 word title to it.
    Chats: {context}
    - STRICTLY return ONLY the TITLE nothing else.
    - If the chats context is generic, return a simple 2-3 word title.
    """
    response: str | Any = LLM.invoke(prompt).content
    return {"title": response.strip()}


def create_chatbot():
    graph = StateGraph(ChatState)
    checkpointer = InMemorySaver()

    # nodes
    graph.add_node("chat", chat_node)

    # edges
    graph.add_edge(START, "chat")
    graph.add_edge("chat", END)
    return graph.compile(checkpointer=checkpointer)

def create_title_generator():
    title_graph = StateGraph(ChatTitle)

    # nodes
    title_graph.add_node("generate_title", gen_title)

    # edges
    title_graph.add_edge(START, "generate_title")
    title_graph.add_edge("generate_title", END)

    return title_graph.compile()

chatbot = create_chatbot()
title_generator = create_title_generator()
initial_state = {"messages": [SystemMessage(content="You are a helpful assistant. ONLY answer the question asked and end your response with a GenZ hashtag. DO NOT REPLY ANYTHING ELSE.")]}